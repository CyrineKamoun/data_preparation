import json
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue

import psycopg2

from src.config.config import Config
from src.core.config import settings
from src.db.db import Database
from src.preparation.network_overture_parallelism import (
    ComputeImpedance,
    ProcessSegments,
)
from src.utils.utils import (
    delete_dir,
    download_link,
    make_dir,
    print_error,
    print_info,
)


class OvertureNetworkPreparation:

    def __init__(self, db: Database, db_rd: Database, region: str):
        self.db = db
        self.db_rd = db_rd
        self.region = region
        self.config = Config("network_overture", region)

        self.NUM_THREADS = os.cpu_count() - 2


    def initialize_dem_table(self):
        """Initialize digital elevation model (DEM) raster table."""

        sql_create_dem_table = """
            DROP TABLE IF EXISTS public.dem;
            CREATE TABLE public.dem (
                rid serial4 NOT NULL,
                rast public.raster NULL,
                filename text NULL,
                CONSTRAINT dem_pkey PRIMARY KEY (rid),
                CONSTRAINT enforce_num_bands_rast CHECK ((ST_NumBands(rast) = 1)),
                CONSTRAINT enforce_srid_rast CHECK ((ST_SRID(rast) = 4326))
            );
            CREATE INDEX dem_st_convexhull_idx ON public.dem USING gist (ST_ConvexHull(rast));
        """
        self.db.perform(sql_create_dem_table)


    def import_dem_tiles(self):
        """Import digital elevation model (DEM) used for calculating slopes."""

        print_info("Importing DEM tiles.")

        # Create directory for storing DEM related files
        dem_dir = os.path.join(settings.INPUT_DATA_DIR, "network_overture", "dem")
        delete_dir(dem_dir)
        make_dir(dem_dir)

        # Get list of source URLs to fetch DEM tiles
        dem_source_list_file_path = os.path.join(
            settings.INPUT_DATA_DIR,
            "network_overture",
            self.config.preparation["dem_source_list"]
        )
        dem_source_list = []
        with open(dem_source_list_file_path, "r") as file:
            for line in file:
                dem_source_list.append(line.strip())

        # Download & import relevant DEM tiles
        for index in range(len(dem_source_list)):
            download_link(
                link=dem_source_list[index],
                directory=dem_dir
            )
            try:
                subprocess.run(
                    f"raster2pgsql -s 4326 -M -a {os.path.join(dem_dir, os.path.basename(dem_source_list[index]))} -F -t auto public.dem | " \
                    f"PGPASSWORD='{settings.POSTGRES_PASSWORD}' psql -h {settings.POSTGRES_HOST} -U {settings.POSTGRES_USER} -d {settings.POSTGRES_DB}",
                    stdout = subprocess.DEVNULL,
                    shell=True,
                    check=True,
                )
            except subprocess.CalledProcessError as e:
                print_error(e)
                raise e

            print_info(f"Imported DEM tile: {index + 1} of {len(dem_source_list)}.")


    def initialize_connectors_table(self):
        """Create table for storing final processed connectors data."""

        sql_create_connectors_table = """
            DROP TABLE IF EXISTS basic.connector CASCADE;
            CREATE TABLE basic.connector (
                index serial NOT NULL UNIQUE,
                id text NOT NULL UNIQUE,
                osm_id int8 NULL,
                geom public.geometry(point, 4326) NOT NULL,
                h3_3 integer NOT NULL,
                h3_6 integer NOT NULL,
                CONSTRAINT connector_pkey PRIMARY KEY (index, h3_3)
            );
            CREATE INDEX idx_connector_id on basic.connector (id);
            CREATE INDEX idx_connector_geom ON basic.connector USING gist (geom);
        """
        self.db.perform(sql_create_connectors_table)


    def initialize_segments_table(self):
        """Create table for storing final processed segments data."""

        sql_create_segments_table = """
            DROP TABLE IF EXISTS basic.segment;
            CREATE TABLE basic.segment (
                id serial NOT NULL,
                length_m float8 NOT NULL,
                length_3857 float8 NOT NULL,
                osm_id int8 NULL,
                bicycle text NULL,
                foot text NULL,
                class_ text NOT NULL,
                impedance_slope float8 NULL,
                impedance_slope_reverse float8 NULL,
                impedance_surface float8 NULL,
                coordinates_3857 json NOT NULL,
                maxspeed_forward integer NULL,
                maxspeed_backward integer NULL,
                "source" integer NOT NULL,
                target integer NOT NULL,
                tags jsonb NULL,
                geom public.geometry(linestring, 4326) NOT NULL,
                h3_3 integer NOT NULL,
                h3_6 integer NOT NULL,
                CONSTRAINT segment_pkey PRIMARY KEY (id, h3_3),
                CONSTRAINT segment_source_fkey FOREIGN KEY ("source") REFERENCES basic.connector(index),
                CONSTRAINT segment_target_fkey FOREIGN KEY (target) REFERENCES basic.connector(index)
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
            FROM ({self.config.collection["region"]}) sub
        """
        region_geom = self.db_rd.select(sql_get_region_geometry)[0][0]

        # TODO Remove this
        with open("src/db/functions/fill_polygon_h3.sql", "r") as f:
            self.db.perform(f.read())
        with open("src/db/functions/to_short_h3_3.sql", "r") as f:
            self.db.perform(f.read())
        with open("src/db/functions/to_short_h3_6.sql", "r") as f:
            self.db.perform(f.read())

        sql_create_region_h3_3_grid = f"""
            DROP TABLE IF EXISTS basic.h3_3_grid;
            CREATE TABLE basic.h3_3_grid AS
                SELECT * FROM
                public.fill_polygon_h3(ST_GeomFromText('{region_geom}', 4326), 3);
            ALTER TABLE basic.h3_3_grid ADD CONSTRAINT h3_3_grid_pkey PRIMARY KEY (h3_index);
            CREATE INDEX ON basic.h3_3_grid USING GIST (h3_boundary);
            CREATE INDEX ON basic.h3_3_grid USING GIST (h3_geom);
        """
        self.db.perform(sql_create_region_h3_3_grid)

        sql_create_region_h3_6_grid = f"""
            DROP TABLE IF EXISTS basic.h3_6_grid;
            CREATE TABLE basic.h3_6_grid AS
                SELECT * FROM
                public.fill_polygon_h3(ST_GeomFromText('{region_geom}', 4326), 6);
            ALTER TABLE basic.h3_6_grid ADD CONSTRAINT h3_6_grid_pkey PRIMARY KEY (h3_index);
            CREATE INDEX ON basic.h3_6_grid USING GIST (h3_boundary);
            CREATE INDEX ON basic.h3_6_grid USING GIST (h3_geom);
        """
        self.db.perform(sql_create_region_h3_6_grid)

        sql_compute_h3_short_index = """
            ALTER TABLE basic.h3_3_grid ADD COLUMN h3_short integer;
            UPDATE basic.h3_3_grid
            SET h3_short = to_short_h3_3(h3_index::bigint);

            ALTER TABLE basic.h3_6_grid ADD COLUMN h3_short integer;
            UPDATE basic.h3_6_grid
            SET h3_short = to_short_h3_6(h3_index::bigint);
        """
        self.db.perform(sql_compute_h3_short_index)

        print_info(f"Computed H3 grid for region: {self.region}.")


    def initiate_segment_processing(self):
        """Utilize multithreading to process segments in parallel."""

        # TODO Remove this
        with open("src/db/functions/classify_segment.sql", "r") as f:
            self.db.perform(f.read())
        with open("src/db/functions/clip_segments.sql", "r") as f:
            self.db.perform(f.read())

        # Load user-configured impedance coefficients for various surface types
        cycling_surfaces = json.dumps(self.config.preparation["cycling_surfaces"])

        # Create separate DB connections for each thread
        db_connections = []
        for _ in range(self.NUM_THREADS):
            connection_string = f"dbname={settings.POSTGRES_DB} user={settings.POSTGRES_USER} \
                                 password={settings.POSTGRES_PASSWORD} host={settings.POSTGRES_HOST} \
                                 port={settings.POSTGRES_PORT}"
            conn = psycopg2.connect(connection_string)
            db_connections.append(conn)

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
                            get_next_h3_index=lambda: h3_3_queue.get() if not h3_3_queue.empty() else None,
                            cycling_surfaces=cycling_surfaces,
                        ).run
                    )
                    for thread_id in range(self.NUM_THREADS)
                ]
                [future.result() for future in as_completed(futures)]

                # Compute segment impedance values
                h3_6_queue = self.get_h3_6_index_queue()
                futures = [
                    executor.submit(
                        ComputeImpedance(
                            thread_id=thread_id,
                            db_connection=db_connections[thread_id],
                            get_next_h3_index=lambda: h3_6_queue.get() if not h3_6_queue.empty() else None,
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
            [conn.close() for conn in db_connections]

        print_info(f"Finished processing segments in {round((time.time() - start_time) / 60)} minutes.")


    def get_h3_3_index_queue(self):
        """Get queue of H3 indexes to be processed by threads."""

        sql_get_h3_indexes = """
            SELECT h3_index
            FROM basic.h3_3_grid
            ORDER BY h3_index;
        """
        h3_indexes = self.db.select(sql_get_h3_indexes)
        h3_index_queue = Queue()
        for h3_index in h3_indexes:
            h3_index_queue.put(h3_index[0])
        return h3_index_queue


    def get_h3_6_index_queue(self):
        """Get queue of H3 indexes to be processed by threads."""

        sql_get_h3_indexes = """
            SELECT h3_short
            FROM basic.h3_6_grid
            ORDER BY h3_index;
        """
        h3_indexes = self.db.select(sql_get_h3_indexes)
        h3_index_queue = Queue()
        for h3_index in h3_indexes:
            h3_index_queue.put(h3_index[0])
        return h3_index_queue


    def clean_up(self):
        """Remove unused temp columns from the connector table."""

        sql_clean_up = """
            ALTER TABLE basic.connector DROP COLUMN id;
            ALTER TABLE basic.connector RENAME COLUMN index TO id;
        """
        self.db.perform(sql_clean_up)


    def run(self):
        """Run Overture network preparation."""

        self.initialize_dem_table()
        self.import_dem_tiles()

        self.initialize_connectors_table()
        self.initialize_segments_table()

        self.compute_region_h3_grid()

        self.initiate_segment_processing()

        self.clean_up()


def prepare_overture_network(region: str):
    print_info(f"Prepare Overture network data for region: {region}.")
    db = Database(settings.LOCAL_DATABASE_URI)
    db_rd = Database(settings.RAW_DATABASE_URI)

    try:
        OvertureNetworkPreparation(
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
