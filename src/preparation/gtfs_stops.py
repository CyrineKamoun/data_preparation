import json
import time

from src.config.config import Config
from src.core.config import settings
from src.db.db import Database
from src.db.tables.poi import POITable
from src.utils.utils import print_info


class GTFSStopsPreparation:
    """Class to prepare & categorize public transport stops from the GTFS dataset."""

    # Route type to mode mapping - must be consistent with the Trip Count Station schema in GOAT Core
    public_transport_types = {
        "bus": {
            3: "Bus",
            11: "Trolleybus",
            700: "Bus Service",
            702: "Express Bus Service",
            704: "Local Bus Service",
            705: "Night Bus Service",
            710: "Sightseeing Bus",
            712: "School Bus",
            715: "Demand and Response Bus Service",
            800: "Trolleybus Service",
        },
        "tram": {
            0: "Tram, Streetcar, Light rail",
            5: "Cable Tram",
            900: "Tram Service",
        },
        "metro": {
            1: "Subway, Metro",
            400: "Metro Service",
            401: "Underground Service",
            402: "Urban Railway Service",
        },
        "rail": {
            2: "Rail",
            100: "Railway Service",
            101: "High Speed Rail Service",
            102: "Long Distance Trains",
            103: "Inter Regional Rail Service",
            105: "Sleeper Rail Service",
            106: "Regional Rail Service",
            107: "Tourist Railway Service",
            109: "Suburban Railway",
            202: "National Coach Service",
            403: "All Urban Railway Services",
        },
        "other": {
            4: "Ferry",
            6: "Aerial lift",
            7: "Funicular",
            1000: "Water Transport Service",
            1300: "Aerial Lift Service",
            1400: "Funicular Service",
            1500: "Taxi Service",
            1700: "Gondola, Suspended cable car",
        },
    }

    wheelchair_boarding_dic = {
            '0': 'no data',
            '1': 'yes',
            '2': 'no',
    }


    def __init__(self, db: Database, region: str):
        self.db = db
        self.region = region
        self.data_config = Config("gtfs_stops", region)
        self.data_config_preparation = self.data_config.preparation

        self.gtfs_schema = self.data_config_preparation['local_gtfs_schema']

    def run(self):
        """Run the public transport stop preparation."""

        # Get the geometires of the study area based on the query defined in the config
        region_geoms = self.db.select(self.data_config_preparation['region'])
        data_set_name=f"public_transport_stop_{self.region}"
        data_set_type='poi'
        schema_name='temporal'

        # Create table for public transport stops
        self.db.perform(POITable(data_set_type=data_set_type, schema_name=schema_name, data_set_name=data_set_name).create_poi_table(table_type='transport'))
        result_table = f"{schema_name}.{data_set_type}_{data_set_name}"
        print_info(f"Created table {result_table}.")

        # Flatten the public transport types dictionary for easy classification
        flat_mode_mapping = {}
        for outer_key, inner_dict in self.public_transport_types.items():
            for inner_key in inner_dict:
                flat_mode_mapping[str(inner_key)] = outer_key

        # Loops through the geometries of the study area and categorizes stops based on their dominant route type
        print_info("Processing GTFS stops...")
        for i, geom in enumerate(region_geoms):
            ts = time.time()

            # Identify wheelchair accessibility based on GTFS classification rules
            classify_gtfs_stop_sql = f"""
                INSERT INTO {result_table} (stop_id, category, name, modes, source, wheelchair, geom)
                WITH clipped_gfts_stops AS (
                    SELECT stop_id, stop_name, geom, h3_3, wheelchair_boarding, parent_station
                    FROM {self.gtfs_schema}.stops
                    WHERE location_type != '1'
                    AND ST_Intersects(geom, ST_SetSRID(ST_GeomFromText(ST_AsText('{geom[0]}')), 4326))
                ),
                categorized_gtfs_stops AS (
                    SELECT c.*, j.route_type::TEXT AS route_type
                    FROM clipped_gfts_stops c
                    CROSS JOIN LATERAL
                    (
                        SELECT DISTINCT o.route_type
                        FROM {self.gtfs_schema}.stop_times_optimized o
                        WHERE o.stop_id = c.stop_id
                        AND o.h3_3 = c.h3_3
                        AND o.route_type IN {tuple(int(key) for key in flat_mode_mapping.keys())}
                    ) j
                ),
                updated_wheelchair_info AS (
                    SELECT child.*, COALESCE(NULLIF(child.wheelchair_boarding, '0'), parent.parent_wheelchair_boarding, '0') AS updated_wheelchair_boarding
                    FROM categorized_gtfs_stops child
                    LEFT JOIN (
                        SELECT stop_id AS parent_stop_id,wheelchair_boarding AS parent_wheelchair_boarding
                        FROM {self.gtfs_schema}.stops
                        WHERE location_type = '1'
                    ) parent
                    ON child.parent_station = parent.parent_stop_id
                )
                SELECT
                stop_id,
                    '{json.dumps(self.data_config_preparation['classification']['station_categories'])}'::jsonb ->> basic.identify_dominant_mode(
                        ARRAY_AGG(DISTINCT route_type),
                        '{json.dumps(flat_mode_mapping)}'::JSONB
                    ) AS category,
                    stop_name AS name,
                    ARRAY_AGG(DISTINCT '{json.dumps(flat_mode_mapping)}'::JSONB ->> route_type) AS modes,
                    'DELFI' AS source,
                    '{json.dumps(self.wheelchair_boarding_dic)}'::JSONB ->> updated_wheelchair_boarding AS wheelchair,
                    geom
                FROM updated_wheelchair_info
                GROUP BY stop_id, stop_name, geom, updated_wheelchair_boarding;
            """

            self.db.perform(classify_gtfs_stop_sql)

            te = time.time()  # End time of the iteration
            iteration_time = te - ts  # Time taken by the iteration
            print_info(f"Processing {i + 1} of {len(region_geoms)}. Iteration time: {round(iteration_time, 3)} seconds.")

        print_info("Preparation of GTFS stops is complete.")

def prepare_gtfs_stops(region: str):
    try:
        db = Database(settings.LOCAL_DATABASE_URI)
        public_transport_stop_preparation = GTFSStopsPreparation(db=db, region=region)
        public_transport_stop_preparation.run()
    finally:
        db.conn.close()
