import csv

from src.config.config import Config
from src.core.config import settings
from src.db.db import Database
from src.utils.utils import (
    print_error,
    print_info,
)


class OverturePlacePreparation:

    def __init__(self, db: Database, db_rd: Database, region: str):
        self.db = db
        self.db_rd = db_rd
        self.region = region
        self.config = Config("overture_place", region)


    def initialize_result_table(self):
        """Create table for storing prepared data."""

        sql_create_result_table = f"""
            DROP TABLE IF EXISTS {self.config.preparation["local_result_table"]} CASCADE;
            CREATE TABLE {self.config.preparation["local_result_table"]} (
                id serial NOT NULL UNIQUE,
                overture_id text NOT NULL UNIQUE,
                name text,
                category text,
                overture_category text,
                overture_category_alternate jsonb,
                confidence numeric,
                source text,
                website text,
                social text,
                email text,
                phone text,
                brand text,
                address_freeform text,
                address_locality text,
                address_postcode text,
                address_region text,
                address_country text,
                geom public.geometry(point, 4326) NOT NULL
            );
            CREATE INDEX ON {self.config.preparation["local_result_table"]} USING gist (geom);
        """
        self.db.perform(sql_create_result_table)


    def parse_category_string(self, category_str):
        """Cleans and splits category strings like '[retail,shopping,supermarket]' into a list."""

        return category_str.strip("[]").split(",")

    def build_category_tree(self, category_list):
        """Builds a nested dictionary from category paths."""

        tree = {}
        for category_str in category_list:
            path = self.parse_category_string(category_str)
            node = tree
            for category in path:
                if category not in node:
                    node[category] = {}
                node = node[category]

        return tree

    def find_deepest_keys(self, d, result):
        """Recursively finds all the deepest keys in a nested dictionary."""

        for k, v in d.items():
            result.append(k)
            if v:
                self.find_deepest_keys(v, result)

        return result  # Returns a set of deepest keys

    def process_place_data(self):
        """Process Overture place data."""

        # Load category mapping from CSV
        csv_path = "/app/src/config/overture_categories.csv"
        category_mapping = {}
        with open(csv_path, mode="r", encoding="utf-8-sig") as csv_file:
            csv_reader = csv.DictReader(csv_file, delimiter=";", skipinitialspace=True)
            for row in csv_reader:
                overture_taxonomy = row["Overture Taxonomy"]
                category_code = row["Category code"]
                category_mapping[overture_taxonomy] = category_code

        # Build category tree
        category_tree = self.build_category_tree(category_mapping.keys())

        # Process data for each category defined in the config
        custom_categories = self.config.preparation["categories"]
        for custom_category in custom_categories:
            print_info(f"Processing category: {custom_category}")

            # Clip category tree to starting depth as defined in the config
            filtered_tree = category_tree.copy()
            starting_depth_key = None
            for filter in custom_categories[custom_category]:
                filtered_tree = filtered_tree[filter]
                starting_depth_key = filter

            # Produce key to top-level category mapping
            valid_sub_categories = {}
            if not filtered_tree:
                valid_sub_categories[starting_depth_key] = starting_depth_key
            else:
                valid_sub_categories[starting_depth_key] = "other"

            for new_cat in filtered_tree.keys():
                valid_sub_categories[new_cat] = new_cat
                if filtered_tree[new_cat]:
                    for deep_cat in self.find_deepest_keys(filtered_tree[new_cat], []):
                        valid_sub_categories[deep_cat] = new_cat

            # Insert all POIs of each valid sub-category into the result table
            total_count = len(valid_sub_categories)
            index = 1
            for sub_category, top_category in valid_sub_categories.items():
                print_info(f"Processing sub-category: {sub_category}, {index} of {total_count}")

                sql_insert = f"""
                    WITH region AS (
                        {self.config.preparation["region"]}
                    )
                    INSERT INTO {self.config.preparation["local_result_table"]} (
                        overture_id, name, category, overture_category, overture_category_alternate,
                        confidence, source, website, social,
                        email, phone, brand, address_freeform,
                        address_locality, address_postcode,
                        address_region, address_country, geom
                    )
                    SELECT id,
                        (names::JSONB)->>'primary', %s, %s, (categories::JSONB)->'alternate',
                        confidence, 'Overture Maps Foundation',
                        websites[1], socials[1], emails[1], phones[1],
                        ((brand::JSONB)->'names')->>'primary',
                        (addresses::JSONB)[0]->>'freeform',
                        (addresses::JSONB)[0]->>'locality',
                        (addresses::JSONB)[0]->>'postcode',
                        (addresses::JSONB)[0]->>'region',
                        (addresses::JSONB)[0]->>'country',
                        geometry
                    FROM {self.config.preparation["local_source_table"]},
                        region
                    WHERE (categories::JSONB)->>'primary' = %s
                    AND ST_Intersects(geometry, geom);
                """
                self.db.perform(sql_insert, (top_category, sub_category, sub_category,))

                index += 1

    def run(self):
        """Run Overture place preparation."""

        self.initialize_result_table()
        self.process_place_data()


def prepare_overture_place(region: str):
    print_info(f"Prepare Overture place data for region: {region}.")
    db = Database(settings.LOCAL_DATABASE_URI)
    db_rd = Database(settings.RAW_DATABASE_URI)

    try:
        OverturePlacePreparation(
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
