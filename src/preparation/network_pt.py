import os

from src.config.config import Config
from src.core.config import settings
from src.db.db import Database
from src.utils.utils import (
    delete_file,
    make_dir,
    osm_convert,
    osm_crop_to_polygon,
    osm_filter_to_highways,
    osm_generate_polygon,
    print_error,
    print_info,
    timing,
)


class NetworkPTPreparation:
    """Class to process and clip OSM and GTFS data into sub-regions."""

    def __init__(self, db: Database, db_rd: Database, region: str):
        self.region = region
        self.db = db
        self.db_rd = db_rd
        self.config = Config("network_pt", region)

        self.input_dir = os.path.join(settings.INPUT_DATA_DIR, "network_pt", region)
        self.output_dir = os.path.join(settings.OUTPUT_DATA_DIR, "network_pt", region)
        self.local_sub_region_table = self.config.preparation["local_sub_region_table"]

        self.sub_regions = None

    def create_sub_regions(self):
        """Create sub-regions based on the region geometry and grid size specified."""

        sub_region_buffer_dist = self.config.preparation["sub_region_buffer_dist"]
        sub_region_grid_size = self.config.preparation["sub_region_count"]
        sub_region_grid_width = int(sub_region_grid_size / 2)
        sub_region_grid_height = int(sub_region_grid_size / 2)
        if sub_region_grid_size % 2 != 0:
            sub_region_grid_height += 1

        print_info(f"Generating sub-regions for region: {self.region}")

        # Fetch region geometry
        region_geom = self.db_rd.select(f"""
            SELECT ST_AsText(geom) as geom
            FROM ({self.config.preparation["region"]}) AS region;
        """)[0][0]

        # Divide region geometry into sub-regions
        sql_create_table = f"""
            DROP TABLE IF EXISTS {self.local_sub_region_table};
            CREATE TABLE {self.local_sub_region_table} AS
            WITH region AS (
                SELECT ST_GeomFromText('{region_geom}', 4326) AS geom
            )
            SELECT
                ROW_NUMBER() OVER () AS id,
                divided.geom,
                ST_Buffer(divided.geom::geography, {sub_region_buffer_dist})::geometry AS buffer_geom
            FROM region,
            LATERAL basic.divide_polygon(geom, {sub_region_grid_width}, {sub_region_grid_height}) AS divided;
        """
        self.db.perform(sql_create_table)

        # Fetch list of sub-region IDs
        sub_regions = self.db.select(f"""
            SELECT id
            FROM {self.local_sub_region_table};
        """)
        self.sub_regions = [id[0] for id in sub_regions]


    def clip_osm_data(self):
        """Clip OSM data to the buffer geometry of each sub-region."""

        # Initialise output directory
        make_dir(self.output_dir)

        # Generate sub-region polygon filters
        print_info(f"Generating OSM filters for region: {self.region}")
        for id in self.sub_regions:
            osm_generate_polygon(
                db_rd=self.db,
                geom_query=f"SELECT buffer_geom as geom FROM {self.local_sub_region_table} WHERE id = {id}",
                dest_file_path=os.path.join(self.output_dir, f"{id}.poly")
            )

        # Crop region OSM data as per sub-region polygon filters
        for id in self.sub_regions:
            print_info(f"Clipping OSM data for sub-region: {id}")
            osm_crop_to_polygon(
                orig_file_path=os.path.join(self.input_dir, self.config.preparation["local_osm_file"]),
                dest_file_path=os.path.join(self.output_dir, f"{id}.o5m"),
                poly_file_path=os.path.join(self.output_dir, f"{id}.poly")
            )
            delete_file(file_path=os.path.join(self.output_dir, f"{id}.poly"))

    def process_osm_data(self):
        """Further process and optimize OSM data."""

        # Filter OSM files to only include highways
        print_info(f"Filtering clipped OSM datasets: {self.region}")
        for id in self.sub_regions:
            osm_filter_to_highways(
                orig_file_path=os.path.join(self.output_dir, f"{id}.o5m"),
                dest_file_path=os.path.join(self.output_dir, f"{id}_filtered.o5m"),
            )
            delete_file(file_path=os.path.join(self.output_dir, f"{id}.o5m"))

        # Convert OSM files to PBF format
        print_info(f"Processing filtered OSM datasets: {self.region}")
        for id in self.sub_regions:
            osm_convert(
                orig_file_path=os.path.join(self.output_dir, f"{id}_filtered.o5m"),
                dest_file_path=os.path.join(self.output_dir, f"{id}.pbf"),
            )
            delete_file(file_path=os.path.join(self.output_dir, f"{id}_filtered.o5m"))

    def clip_gtfs_data(self):
        """Clip GTFS data to the buffer geometry of each sub-region."""

        # Generate sub-region GTFS datasets
        for id in self.sub_regions:
            print_info(f"Clipping GTFS data for sub-region: {id}")

            region_schema = self.config.preparation["local_gtfs_schema"]
            sub_region_schema = f"network_pt_{self.region}_{id}"

            # Create new schema for sub-region data
            self.db.perform(f"""
                DROP SCHEMA IF EXISTS {sub_region_schema} CASCADE;
                CREATE SCHEMA {sub_region_schema};
            """)

            # Create trips table
            self.db.perform(f"""
                CREATE TABLE {sub_region_schema}.trips AS
                SELECT t.*
                FROM {region_schema}.trips t, (
                    SELECT DISTINCT(trip_id) AS trip_id
                    FROM (
                        SELECT s.stop_id
                        FROM {region_schema}.stops s,
                        {self.local_sub_region_table} r
                        WHERE r.id = {id}
                        AND ST_Intersects(s.geom, r.buffer_geom)
                    ) s,
                    {region_schema}.stop_times st
                    WHERE st.stop_id = s.stop_id
                ) sub
                WHERE t.trip_id = sub.trip_id;
                ALTER TABLE {sub_region_schema}.trips ADD PRIMARY KEY (trip_id);
                CREATE INDEX ON {sub_region_schema}.trips (route_id);
            """)

            # Create routes table
            self.db.perform(f"""
                CREATE TABLE {sub_region_schema}.routes AS (
                    SELECT r.*
                    FROM {region_schema}.routes r, (
                        SELECT DISTINCT(route_id) AS route_id
                        FROM {sub_region_schema}.trips t
                    ) sub
                    WHERE r.route_id = sub.route_id
                );
                ALTER TABLE {sub_region_schema}.routes ADD PRIMARY KEY (route_id);
            """)

            # Create agency table
            self.db.perform(f"""
                CREATE TABLE {sub_region_schema}.agency AS (
                    SELECT a.*
                    FROM {region_schema}.agency a, (
                        SELECT DISTINCT(agency_id) AS agency_id
                        FROM {sub_region_schema}.routes r
                    ) sub
                    WHERE a.agency_id = sub.agency_id
                );
                ALTER TABLE {sub_region_schema}.agency ADD PRIMARY KEY (agency_id);
            """)

            # Create calendar table
            self.db.perform(f"""
                CREATE TABLE {sub_region_schema}.calendar AS (
                    SELECT c.*
                    FROM {region_schema}.calendar c, (
                        SELECT DISTINCT(service_id) AS service_id
                        FROM {sub_region_schema}.trips t
                    ) sub
                    WHERE c.service_id = sub.service_id
                );
                ALTER TABLE {sub_region_schema}.calendar ADD PRIMARY KEY (service_id);
                CREATE INDEX ON {sub_region_schema}.calendar (start_date, end_date);
            """)

            # Create calendar_dates table
            self.db.perform(f"""
                CREATE TABLE {sub_region_schema}.calendar_dates AS (
                    SELECT cd.*
                    FROM {region_schema}.calendar_dates cd, (
                        SELECT DISTINCT(service_id) AS service_id
                        FROM {sub_region_schema}.trips t
                    ) sub
                    WHERE cd.service_id = sub.service_id
                );
                ALTER TABLE {sub_region_schema}.calendar_dates ADD PRIMARY KEY (service_id, date);
            """)

            # Create shapes table
            self.db.perform(f"""
                CREATE TABLE {sub_region_schema}.shapes AS (
                    SELECT s.*
                    FROM {region_schema}.shapes s, (
                        SELECT DISTINCT(shape_id) AS shape_id
                        FROM {sub_region_schema}.trips t
                    ) sub
                    WHERE s.shape_id = sub.shape_id
                );
                ALTER TABLE {sub_region_schema}.shapes ADD PRIMARY KEY (shape_id, shape_pt_sequence, h3_3);
            """)

    def optimize_gtfs_data(self):
        """Optimize the size of GTFS data by removing unnecessary trips."""

        # Check if optimization is desired
        if not self.config.preparation["weekday_tuesday"] or \
            not self.config.preparation["weekday_saturday"] or \
            not self.config.preparation["weekday_sunday"]:
            return

        for id in self.sub_regions:
            print_info(f"Optimizing GTFS data for sub-region: {id}")

            region_schema = self.config.preparation["local_gtfs_schema"]
            sub_region_schema = f"network_pt_{self.region}_{id}"

            # Remove trips that are not scheduled to operate on the specified days
            sql_delete_unused_trips = f"""
                WITH unused_trips AS (
                    WITH inactive_services AS (
                        SELECT service_id
                        FROM {sub_region_schema}.calendar
                        WHERE tuesday = '0'
                        AND saturday = '0'
                        AND sunday = '0'
                    ),
                    active_dates AS (
                        SELECT service_id
                        FROM {sub_region_schema}.calendar_dates
                        WHERE date IN (
                            '{self.config.preparation["weekday_tuesday"]}'::DATE,
                            '{self.config.preparation["weekday_saturday"]}'::DATE,
                            '{self.config.preparation["weekday_sunday"]}'::DATE
                        )
                        AND exception_type = 1
                    )
                    SELECT trip_id
                    FROM (
                        SELECT *
                        FROM inactive_services
                        WHERE service_id NOT IN (SELECT * FROM active_dates)
                    ) inactive,
                    {sub_region_schema}.trips t
                    WHERE t.service_id = inactive.service_id
                )
                DELETE FROM {sub_region_schema}.trips
                WHERE trip_id IN (SELECT trip_id FROM unused_trips);
            """
            self.db.perform(sql_delete_unused_trips)

            # Create stop_times table
            self.db.perform(f"""
                CREATE TABLE {sub_region_schema}.stop_times AS (
                    SELECT st.*
                    FROM {region_schema}.stop_times st,
                    {sub_region_schema}.trips t
                    WHERE st.trip_id = t.trip_id
                );
                ALTER TABLE {sub_region_schema}.stop_times ADD PRIMARY KEY (trip_id, stop_sequence, h3_3);
                CREATE INDEX ON {sub_region_schema}.stop_times (stop_id, arrival_time, departure_time);
            """)

            # Create stops table
            self.db.perform(f"""
                CREATE TABLE {sub_region_schema}.stops AS (
                    SELECT s.*
                    FROM {region_schema}.stops s,
                    (SELECT DISTINCT stop_id, h3_3 FROM {sub_region_schema}.stop_times) st
                    WHERE s.h3_3 = st.h3_3
                    AND s.stop_id = st.stop_id
                );
                ALTER TABLE {sub_region_schema}.stops ADD PRIMARY KEY (stop_id, h3_3);
            """)

            # Import missing parent stations and their child stops
            self.db.perform(f"""
                INSERT INTO {sub_region_schema}.stops
                SELECT s.*
                FROM {region_schema}.stops s, (
                    SELECT DISTINCT(parent_station) AS parent_station
                    FROM {sub_region_schema}.stops
                ) sub
                WHERE s.stop_id = sub.parent_station
                OR s.parent_station = sub.parent_station
                ON CONFLICT DO NOTHING;
            """)

@timing
def prepare_network_pt(region: str):
    print_info(f"Preparing PT network for region: {region}")
    db = Database(settings.LOCAL_DATABASE_URI)
    db_rd = Database(settings.RAW_DATABASE_URI)

    try:
        network_pt_preparation = NetworkPTPreparation(
            db=db,
            db_rd=db_rd,
            region=region
        )
        network_pt_preparation.create_sub_regions()
        network_pt_preparation.clip_osm_data()
        network_pt_preparation.process_osm_data()
        network_pt_preparation.clip_gtfs_data()
        network_pt_preparation.optimize_gtfs_data()
        print_info(f"Finished preparing PT network for region: {region}")
    except Exception as e:
        print_error(f"Failed to prepare PT network for region: {region}")
        raise e
    finally:
        db.close()
        db_rd.close()
