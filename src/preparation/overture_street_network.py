import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue

import psycopg2

from src.config.config import Config
from src.core.config import settings
from src.db.db import Database
from src.preparation.overture_street_network_helper import (
    ComputeImpedance,
    ProcessSegments,
)
from src.utils.utils import (
    print_error,
    print_info,
)


class OvertureStreetNetworkPreparation:

    def __init__(self, db: Database, db_rd: Database, region: str):
        self.db = db
        self.db_rd = db_rd
        self.region = region
        self.config = Config("overture_street_network", region)

        self.NUM_THREADS = 16


    def initialize_connectors_table(self):
        """Create table for storing final processed connectors data."""

        sql_create_connectors_table = """
            DROP TABLE IF EXISTS basic.connector CASCADE;
            CREATE TABLE basic.connector (
                id serial NOT NULL UNIQUE,
                overture_id text NULL UNIQUE,
                geom public.geometry(point, 4326) NOT NULL,
                h3_3 integer NOT NULL,
                h3_6 integer NOT NULL,
                CONSTRAINT connector_pkey PRIMARY KEY (id, h3_3)
            );
            CREATE INDEX idx_connector_overture_id on basic.connector (overture_id);
            CREATE INDEX idx_connector_geom ON basic.connector USING gist (geom);
        """
        self.db.perform(sql_create_connectors_table)


    def initialize_segments_table(self):
        """Create table for storing final processed segments data."""

        sql_create_segments_table = """
            DROP TABLE IF EXISTS basic.segment;
            CREATE TABLE basic.segment (
                id serial NOT NULL,
                overture_id text NOT NULL,
                length_m float8 NOT NULL,
                length_3857 float8 NOT NULL,
                class_ text NOT NULL,
                subclass text NULL,
                impedance_slope float8 NULL,
                impedance_slope_reverse float8 NULL,
                impedance_surface float8 NULL,
                coordinates_3857 json NOT NULL,
                maxspeed_forward integer NULL,
                maxspeed_backward integer NULL,
                "source" integer NOT NULL,
                target integer NOT NULL,
                geom public.geometry(linestring, 4326) NOT NULL,
                h3_3 integer NOT NULL,
                h3_6 integer NOT NULL,
                CONSTRAINT segment_pkey PRIMARY KEY (id, h3_3),
                CONSTRAINT segment_source_fkey FOREIGN KEY ("source") REFERENCES basic.connector(id),
                CONSTRAINT segment_target_fkey FOREIGN KEY (target) REFERENCES basic.connector(id)
            );
            CREATE INDEX idx_segment_geom ON basic.segment USING gist (geom);
            CREATE INDEX ix_basic_segment_source ON basic.segment USING btree (source);
            CREATE INDEX ix_basic_segment_target ON basic.segment USING btree (target);
        """
        self.db.perform(sql_create_segments_table)


    def compute_region_h3_grid(self):
        """Use the h3 grid function to create a h3 grid for our region."""

        sql_get_region_geometry = f"""
            SELECT ST_AsText(geom) AS geom
            FROM ({self.config.preparation["region"]}) sub
        """
        region_geom = self.db_rd.select(sql_get_region_geometry)[0][0]

        sql_create_region_h3_3_grid = f"""
            DROP TABLE IF EXISTS basic.h3_3_grid;
            CREATE TABLE basic.h3_3_grid AS
                SELECT * FROM
                basic.fill_polygon_h3(ST_GeomFromText('{region_geom}', 4326), 3);
            ALTER TABLE basic.h3_3_grid ADD CONSTRAINT h3_3_grid_pkey PRIMARY KEY (h3_index);
            CREATE INDEX ON basic.h3_3_grid USING GIST (h3_boundary);
            CREATE INDEX ON basic.h3_3_grid USING GIST (h3_geom);
        """
        self.db.perform(sql_create_region_h3_3_grid)

        sql_create_region_h3_6_grid = f"""
            DROP TABLE IF EXISTS basic.h3_6_grid;
            CREATE TABLE basic.h3_6_grid AS
                SELECT * FROM
                basic.fill_polygon_h3(ST_GeomFromText('{region_geom}', 4326), 6);
            ALTER TABLE basic.h3_6_grid ADD CONSTRAINT h3_6_grid_pkey PRIMARY KEY (h3_index);
            CREATE INDEX ON basic.h3_6_grid USING GIST (h3_boundary);
            CREATE INDEX ON basic.h3_6_grid USING GIST (h3_geom);
        """
        self.db.perform(sql_create_region_h3_6_grid)

        sql_compute_h3_short_index = """
            ALTER TABLE basic.h3_3_grid ADD COLUMN h3_short integer;
            UPDATE basic.h3_3_grid
            SET h3_short = basic.to_short_h3_3(h3_index::bigint);

            ALTER TABLE basic.h3_6_grid ADD COLUMN h3_short integer;
            UPDATE basic.h3_6_grid
            SET h3_short = basic.to_short_h3_6(h3_index::bigint);
        """
        self.db.perform(sql_compute_h3_short_index)

        print_info(f"Computed H3 grid for region: {self.region}.")


    def open_database_connections(self, num_connections: int):
        """Open multiple database connections for parallel processing."""

        connections = []
        for _ in range(num_connections):
            connection_string = f"dbname={settings.POSTGRES_DB} user={settings.POSTGRES_USER} \
                                 password={settings.POSTGRES_PASSWORD} host={settings.POSTGRES_HOST} \
                                 port={settings.POSTGRES_PORT}"
            conn = psycopg2.connect(connection_string)
            connections.append(conn)
        return connections


    def close_database_connections(self, connections: list):
        """Close multiple database connections."""

        [conn.close() for conn in connections]


    def initiate_segment_processing(self):
        """Utilize multithreading to process segments in parallel."""

        # Load user-configured impedance coefficients for various surface types
        cycling_surfaces = json.dumps(self.config.preparation["cycling_surfaces"])

        # Load user-configured default speed limits for various road classes
        default_speed_limits = json.dumps(self.config.preparation["default_speed_limits"])

        # Create separate DB connections for each thread
        db_connections = self.open_database_connections(self.NUM_THREADS)

        print_info(f"Starting {self.NUM_THREADS} threads for processing segments.")
        start_time = time.time()

        # Start threads
        try:
            with ThreadPoolExecutor(max_workers=self.NUM_THREADS) as executor:
                # Process segments & connectors
                h3_3_queue = self.get_h3_3_index_queue()
                futures = [
                    executor.submit(
                        ProcessSegments(
                            thread_id=thread_id,
                            db_connection=db_connections[thread_id],
                            get_next_h3_index=lambda: self.get_next_h3_3_index(h3_3_queue),
                            cycling_surfaces=cycling_surfaces,
                            default_speed_limits=default_speed_limits,
                        ).run
                    )
                    for thread_id in range(self.NUM_THREADS)
                ]
                [future.result() for future in as_completed(futures)]
        except Exception as e:
            print_error(e)
            raise e
        finally:
            # Clean up DB connections
            self.close_database_connections(db_connections)

        # Create index on h3_6 column to support further processing
        self.db.perform("CREATE INDEX ON basic.segment (h3_6);")

        print_info(f"Finished processing segments in {round((time.time() - start_time) / 60)} minutes.")


    def compute_slope_impedance(self):
        """Utilize multithreading to process segments in parallel."""

        # Create separate DB connections for each thread
        db_connections = self.open_database_connections(self.NUM_THREADS)

        print_info(f"Starting {self.NUM_THREADS} threads for computing slope impedance.")
        start_time = time.time()

        # Start threads
        try:
            with ThreadPoolExecutor(max_workers=self.NUM_THREADS) as executor:
                # Compute segment impedance values
                h3_6_queue = self.get_h3_6_index_queue()
                futures = [
                    executor.submit(
                        ComputeImpedance(
                            thread_id=thread_id,
                            db_connection=db_connections[thread_id],
                            get_next_h3_index=lambda: self.get_next_h3_6_index(h3_6_queue),
                        ).run
                    )
                    for thread_id in range(self.NUM_THREADS)
                ]
                [future.result() for future in as_completed(futures)]
        except Exception as e:
            print_error(e)
            raise e
        finally:
            # Clean up DB connections
            self.close_database_connections(db_connections)

        # Fix remaining segments where impedance values could not be computed
        print_info("Fixing remaining segments where impedance values could not be computed.")
        sql_fix_remaining_segments = """
            UPDATE basic.segment
            SET impedance_slope = 0, impedance_slope_reverse = 0
            WHERE impedance_slope IS NULL;
        """
        self.db.perform(sql_fix_remaining_segments)

        print_info(f"Finished computing slope impedance in {round((time.time() - start_time) / 60)} minutes.")


    def get_h3_3_index_queue(self):
        """Get queue of H3 indexes to be processed by threads."""

        sql_get_h3_indexes = """
            SELECT h3_short
            FROM basic.h3_3_grid
            ORDER BY h3_short;
        """
        h3_indexes = self.db.select(sql_get_h3_indexes)
        h3_index_queue = Queue()
        for h3_index in h3_indexes:
            h3_index_queue.put(h3_index[0])
        return h3_index_queue


    def get_next_h3_3_index(self, h3_3_queue: Queue):
        """Get next H3_3 cell index to be processed by a thread."""

        if h3_3_queue.empty():
            return None
        next_h3_3_index = h3_3_queue.get()
        print_info(f"Processing H3_3 cell {next_h3_3_index}, remaining: {h3_3_queue.qsize()}")
        return next_h3_3_index


    def get_h3_6_index_queue(self):
        """Get queue of H3 indexes to be processed by threads."""

        sql_get_h3_indexes = """
            SELECT h3_short
            FROM basic.h3_6_grid
            ORDER BY h3_short;
        """
        h3_indexes = self.db.select(sql_get_h3_indexes)
        h3_index_queue = Queue()
        for h3_index in h3_indexes:
            h3_index_queue.put(h3_index[0])
        return h3_index_queue


    def get_next_h3_6_index(self, h3_6_queue: Queue):
        """Get next H3_6 cell index to be processed by a thread."""

        if h3_6_queue.empty():
            return None
        next_h3_6_index = h3_6_queue.get()
        if len(h3_6_queue.queue) % 1000 == 0:
            print_info(f"Processing H3_6 cell {next_h3_6_index}, remaining: {h3_6_queue.qsize()}")
        return next_h3_6_index


    def run(self):
        """Run Overture network preparation."""

        self.initialize_connectors_table()
        self.initialize_segments_table()

        self.compute_region_h3_grid()

        self.initiate_segment_processing()
        self.compute_slope_impedance()


def prepare_overture_street_network(region: str):
    print_info(f"Prepare Overture network data for region: {region}.")
    db = Database(settings.LOCAL_DATABASE_URI)
    db_rd = Database(settings.RAW_DATABASE_URI)

    try:
        OvertureStreetNetworkPreparation(
            db=db,
            db_rd=db_rd,
            region=region
        ).run()
        db.close()
        db_rd.close()
        print_info("Finished Overture network preparation.")
    except Exception as e:
        print_error(e)
        raise e
    finally:
        db.close()
        db_rd.close()
