import datetime

from src.collection.kart.prepare_kart import PrepareKart
from src.config.config import Config
from src.core.enums import DumpType
from src.db.db import Database
from src.db.tables.poi import POITable
from src.utils.utils import create_table_dump, print_info, restore_table_dump, timing

#TODO: optional
# 0. optional prepare requested data that should be updated by the subscription
# 0.1 OSM collection, preparation
# 0.2 Overture collection, preparation
# 0.3 OSM, Overture fusion
# 0.4 Kart -> self.prepare_kart.prepare_kart()
# 0.5 GTFS PT stops -> preparation

#TODO: add validation

class Subscription:
    """Class to prepare the POIs from OpenStreetMap."""

    def __init__(self, db: Database, db_rd: Database, region: str):
        """Constructor method.

        Args:
            db (Database): Database object
        """
        self.db_rd = db_rd
        self.db = db
        self.db_config = self.db.db_config

        self.poi_categories = [
            "food_drink",
            "health",
            "public_transport",
            "other",
            "mobility_service",
            "public_service",
            "service",
            "shopping",
            "sport",
            "tourism_leisure",
            "childcare",
            "school",
        ]
        self.geonode_schema_name = "goat"

        self.table_name = "poi"
        self.region = region
        self.config_pois = Config(self.table_name, region)
        self.config_pois_preparation = self.config_pois.preparation
        self.repo_url = self.config_pois.subscription["repo_url"]

        self.prepare_kart = PrepareKart(
            self.db,
            repo_url=self.repo_url,
            maintainer=self.db_config.user,
            table_name=self.table_name,
        )
        self.kart_schema = f'kart_{self.table_name}s'
        self.source_to_table = {
            'OSM': f'{self.geonode_schema_name}.poi_osm_{self.region}',
            'Overture': f'{self.geonode_schema_name}.poi_overture_{self.region}',
            'OSM_Overture': f'{self.geonode_schema_name}.poi_osm_overture_{self.region}_fusion_result',
            'GTFS': f'{self.geonode_schema_name}.poi_public_transport_stop_{self.region}'
        }

        # Get the date of the OSM data. In upstream functions it is guaranteeed that all dates are the same
        self.osm_data_date = self.db.select("SELECT date FROM poi_osm_boundary LIMIT 1")[0][0]

    def get_source_table(self, category):
        try:
            source = self.db_rd.select(f"""SELECT source from {self.geonode_schema_name}.data_subscription WHERE category = '{category}' AND rule = 'subscribe'""")[0][0]
            return self.source_to_table[source]
        except IndexError:
            print(f"No source found for category '{category}'")
        except KeyError:
            print(f"No table found for source '{source}'")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

    def get_kart_poi_table_name(self, category):
        try:
            # find correct poi table name within kart
            sql_kart_poi_table_name = f"""
                SELECT table_name
                FROM {self.kart_schema}.poi_categories
                WHERE category = '{category}';
            """
            return self.db.select(sql_kart_poi_table_name)[0][0]
        except IndexError:
            print(f"No table name found for category '{category}'")
            return None
        except Exception as e:
            print(f"An error occurred: {e}")
            return None

    @timing
    def migrate_osm_and_overture(self):
        # OSM data
        create_table_dump(
            db_config=self.db.db_config,
            schema=self.geonode_schema_name,
            table_name='poi_osm_europe'
        )

        self.db_rd.perform(f"DROP TABLE IF EXISTS {self.geonode_schema_name}.poi_osm_europe")

        restore_table_dump(
            db_config=self.db_rd.db_config,
            schema={self.geonode_schema_name},
            table_name='poi_osm_europe'
        )

        # Overture data
        create_table_dump(
            db_config=self.db.db_config,
            schema=self.geonode_schema_name,
            table_name='poi_overture_europe'
        )

        self.db_rd.perform(f"DROP TABLE IF EXISTS {self.geonode_schema_name}.poi_overture_europe")

        restore_table_dump(
            db_config=self.db_rd.db_config,
            schema={self.geonode_schema_name},
            table_name='poi_overture_europe'
        )

        # OSM-Overture-Fusion data
        create_table_dump(
            db_config=self.db.db_config,
            schema=self.geonode_schema_name,
            table_name='poi_osm_overture_europe_fusion_result'
        )

        self.db_rd.perform(f"DROP TABLE IF EXISTS {self.geonode_schema_name}.poi_osm_overture_europe_fusion_result")

        restore_table_dump(
            db_config=self.db_rd.db_config,
            schema={self.geonode_schema_name},
            table_name='poi_osm_overture_europe_fusion_result'
        )

    def prepare_poi_tables(self, poi_category, geonode_poi_table_names):

        poi_table_name = f"poi_{poi_category}"

        # rename outdated tables -> add creation date
        if poi_table_name in geonode_poi_table_names:
            comment_sql = f"""
            SELECT obj_description('{self.geonode_schema_name}.{poi_table_name}'::regclass);
            """
            comment = self.db_rd.select(comment_sql)[0][0]

            # Assuming the comment is in the format 'Created on YYYYMMDD'
            date_str = comment.split(' ')[-1]

            # Now you can use date_str in your ALTER TABLE command
            sql_rename_table = f"""
            ALTER TABLE {self.geonode_schema_name}.{poi_table_name} RENAME TO {poi_table_name}_{date_str};
            """
            self.db_rd.perform(sql_rename_table)

        # create fresh poi table
        if poi_category in ['childcare', 'school']:
            create_table_sql = POITable(data_set_type='poi', schema_name = self.geonode_schema_name, data_set_name = poi_category).create_poi_table(table_type=poi_category, create_index=False, temporary=False)
            self.db_rd.perform(create_table_sql)
        else:
            create_table_sql = POITable(data_set_type='poi', schema_name = self.geonode_schema_name, data_set_name = poi_category).create_poi_table(table_type='standard', create_index=False, temporary=False)
            self.db_rd.perform(create_table_sql)

        #apply constraints to POI tables
        sql_common_poi = f"""
            ALTER TABLE {self.geonode_schema_name}.{poi_table_name}  ALTER COLUMN source SET NOT NULL;
            ALTER TABLE {self.geonode_schema_name}.{poi_table_name}  ALTER COLUMN geom SET NOT NULL;
            ALTER TABLE {self.geonode_schema_name}.{poi_table_name}  DROP CONSTRAINT IF EXISTS name_not_empty_string_check, ADD CONSTRAINT name_not_empty_string_check CHECK (name != '');
            ALTER TABLE {self.geonode_schema_name}.{poi_table_name}  DROP CONSTRAINT IF EXISTS check_name_no_whitespace, ADD CONSTRAINT check_name_no_whitespace CHECK (name = LTRIM(RTRIM(name)) AND name <> '');
            ALTER TABLE {self.geonode_schema_name}.{poi_table_name}  DROP CONSTRAINT IF EXISTS check_street_no_whitespace, ADD CONSTRAINT check_street_no_whitespace CHECK (street = LTRIM(RTRIM(street)) AND street <> '');
            ALTER TABLE {self.geonode_schema_name}.{poi_table_name}  DROP CONSTRAINT IF EXISTS street_not_empty_string_check, ADD CONSTRAINT street_not_empty_string_check CHECK (street != '');
            ALTER TABLE {self.geonode_schema_name}.{poi_table_name}  DROP CONSTRAINT IF EXISTS housenumber_not_empty_string_check, ADD CONSTRAINT housenumber_not_empty_string_check CHECK (housenumber != '');
            ALTER TABLE {self.geonode_schema_name}.{poi_table_name}  DROP CONSTRAINT IF EXISTS zipcode_not_empty_string_check, ADD CONSTRAINT zipcode_not_empty_string_check CHECK (zipcode != '');
            ALTER TABLE {self.geonode_schema_name}.{poi_table_name}  DROP CONSTRAINT IF EXISTS phone_not_empty_string_check, ADD CONSTRAINT phone_not_empty_string_check CHECK (phone != '');
            ALTER TABLE {self.geonode_schema_name}.{poi_table_name}  DROP CONSTRAINT IF EXISTS email_check, ADD CONSTRAINT email_check CHECK (octet_length(email) BETWEEN 6 AND 320 AND email LIKE '_%@_%.__%');
            ALTER TABLE {self.geonode_schema_name}.{poi_table_name}  DROP CONSTRAINT IF EXISTS opening_hours_not_empty_string_check, ADD CONSTRAINT opening_hours_not_empty_string_check CHECK (opening_hours != '');
            ALTER TABLE {self.geonode_schema_name}.{poi_table_name}  DROP CONSTRAINT IF EXISTS website_not_empty_string_check, ADD CONSTRAINT website_not_empty_string_check CHECK (website != '');
            ALTER TABLE {self.geonode_schema_name}.{poi_table_name}  DROP CONSTRAINT IF EXISTS wheelchair_check, ADD CONSTRAINT wheelchair_check CHECK (wheelchair IN ('yes', 'no', 'limited'));
        """
        #TODO: removed for now as data_source table is still (only) in Kart
        # ALTER TABLE {self.geonode_schema_name}.{table_name}  ADD FOREIGN KEY (source) REFERENCES {self.geonode_schema_name}.data_source(name) ON DELETE CASCADE;
        #TODO: not needed anymore?
        # ALTER TABLE {self.geonode_schema_name}.{table_name}  OWNER TO {self.maintainer};
        self.db_rd.perform(sql_common_poi)

        # Add additional constraints all tables besides poi_childcare and poi_school
        if poi_table_name not in ("poi_childcare", "poi_school"):
            sql_addition_constraints = f"""
                ALTER TABLE {self.geonode_schema_name}.{poi_table_name}  ALTER COLUMN category SET NOT NULL;
                ALTER TABLE {self.geonode_schema_name}.{poi_table_name}  DROP CONSTRAINT IF EXISTS other_categories_array_check, ADD CONSTRAINT other_categories_array_check CHECK (other_categories IS NULL OR other_categories::text[] IS NOT NULL);
            """
            #TODO: removed for now as data_source table is still (only) in Kart
            # ALTER TABLE {self.geonode_schema_name}.{table_name}  ADD FOREIGN KEY (category) REFERENCES {self.geonode_schema_name}.poi_categories(category) ON DELETE CASCADE;
            #TODO: not needed anymore?
            # ALTER TABLE {self.geonode_schema_name}.{table_name}  OWNER TO {self.maintainer};
            self.db_rd.perform(sql_addition_constraints)

        # Add additional constraint for operator and tags column to all tables besides poi_childcare
        if poi_table_name not in ("poi_childcare"):
            sql_addition_constraints_operator_tags = f"""
                ALTER TABLE {self.geonode_schema_name}.{poi_table_name}  DROP CONSTRAINT IF EXISTS operator_not_empty_string_check,  ADD CONSTRAINT operator_not_empty_string_check CHECK (operator != '');
                ALTER TABLE {self.geonode_schema_name}.{poi_table_name}  DROP CONSTRAINT IF EXISTS tags_jsonb_check, ADD CONSTRAINT tags_jsonb_check CHECK (jsonb_typeof(tags::jsonb) = 'object');
            """
            #TODO: not needed anymore?
            # ALTER TABLE {self.geonode_schema_name}.{table_name}  OWNER TO {self.maintainer};
            self.db_rd.perform(sql_addition_constraints_operator_tags)

        # Add additional constraints for poi_childcare
        if poi_table_name in ("poi_childcare"):
            sql_addition_constraints_childcare = f"""
                ALTER TABLE {self.geonode_schema_name}.{poi_table_name}  DROP CONSTRAINT IF EXISTS min_age_check, ADD CONSTRAINT min_age_check CHECK (min_age IS NULL OR (min_age >= 0 AND min_age <= 16));
                ALTER TABLE {self.geonode_schema_name}.{poi_table_name}  DROP CONSTRAINT IF EXISTS max_age_check, ADD CONSTRAINT max_age_check CHECK (max_age IS NULL OR (max_age >= 0 AND max_age <= 16));
                ALTER TABLE {self.geonode_schema_name}.{poi_table_name}  DROP CONSTRAINT IF EXISTS carrier_not_empty_string_check, ADD CONSTRAINT carrier_not_empty_string_check CHECK (carrier != '');
                ALTER TABLE {self.geonode_schema_name}.{poi_table_name}  DROP CONSTRAINT IF EXISTS carrier_type_not_empty_string_check, ADD CONSTRAINT carrier_type_not_empty_string_check CHECK (carrier_type != '');
                ALTER TABLE {self.geonode_schema_name}.{poi_table_name}  DROP CONSTRAINT IF EXISTS min_max_check, ADD CONSTRAINT min_max_check CHECK (min_age <= max_age);
            """
            print(f"Executing SQL: {sql_addition_constraints_childcare}")
            #TODO: not needed anymore?
            # ALTER TABLE {self.geonode_schema_name}.{table_name}  OWNER TO {self.maintainer};
            self.db_rd.perform(sql_addition_constraints_childcare)

    def migrate_kart_tables(self, poi_category):
        poi_table_name = f"poi_{poi_category}"

        create_table_dump(
            db_config=self.db.db_config,
            schema=self.kart_schema,
            table_name=poi_table_name,
            dump_type=DumpType.schema
        )

        self.db_rd.perform(f"DROP TABLE IF EXISTS {self.kart_schema}.{poi_table_name}")

        restore_table_dump(
            db_config=self.db_rd.db_config,
            schema={self.kart_schema},
            table_name=poi_table_name,
            dump_type=DumpType.schema
        )

        remove_constraints_sql = f"""
        DO $$
        DECLARE
            constraint_name varchar;
        BEGIN
            FOR constraint_name IN (
                SELECT conname
                FROM pg_constraint
                INNER JOIN pg_namespace ON pg_namespace.oid = pg_constraint.connamespace
                WHERE nspname = '{self.kart_schema}' AND conrelid = '{self.kart_schema}.{poi_table_name}'::regclass
            )
            LOOP
                EXECUTE 'ALTER TABLE ' || '{self.kart_schema}.' || '{poi_table_name}' || ' DROP CONSTRAINT ' || constraint_name;
            END LOOP;
        END $$;
        """
        self.db_rd.perform(remove_constraints_sql)

        create_table_dump(
            db_config=self.db.db_config,
            schema=self.kart_schema,
            table_name=poi_table_name,
            dump_type=DumpType.data
        )

        restore_table_dump(
            db_config=self.db_rd.db_config,
            schema={self.kart_schema},
            table_name=poi_table_name,
            dump_type=DumpType.data
        )

        if poi_category == 'childcare':
            #TODO: in Kart the childcare table does not have a tags column
            insert_restored_data_into_geonode_poi_table_sql = f"""
                INSERT INTO {self.geonode_schema_name}.{poi_table_name}(nursery, kindergarten, after_school, min_age, max_age, carrier, carrier_type, name, street, housenumber, zipcode, phone, email, website, capacity, opening_hours, wheelchair, source, geom)
                SELECT
                    nursery,
                    kindergarten,
                    after_school,
                    min_age, max_age,
                    carrier,
                    carrier_type,
                    name,
                    street,
                    housenumber,
                    zipcode,
                    phone,
                    email,
                    website,
                    capacity,
                    opening_hours,
                    wheelchair,
                    source,
                    geom
                FROM {self.kart_schema}.{poi_table_name}
            """
            self.db_rd.perform(insert_restored_data_into_geonode_poi_table_sql)

        elif poi_category == 'school':
            insert_restored_data_into_geonode_poi_table_sql = f"""
                INSERT INTO {self.geonode_schema_name}.{poi_table_name}(school_isced_level_1, school_isced_level_2, school_isced_level_3, operator, name, street, housenumber, zipcode, phone, email, website, capacity, opening_hours, wheelchair, source, tags, geom)
                SELECT
                    school_isced_level_1,
                    school_isced_level_2,
                    school_isced_level_3,
                    operator,
                    name,
                    street,
                    housenumber,
                    zipcode,
                    phone,
                    email,
                    website,
                    capacity,
                    opening_hours,
                    wheelchair,
                    source,
                    tags::jsonb,
                    geom
                FROM {self.kart_schema}.{poi_table_name}
            """
            self.db_rd.perform(insert_restored_data_into_geonode_poi_table_sql)

        else:
            insert_restored_data_into_geonode_poi_table_sql = f"""
                INSERT INTO {self.geonode_schema_name}.{poi_table_name}(category, other_categories, operator, name, street, housenumber, zipcode, phone, email, website, capacity, opening_hours, wheelchair, source, tags, geom)
                SELECT
                    category,
                    string_to_array(other_categories, ',')::text[] AS other_categories,
                    operator,
                    name,
                    street,
                    housenumber,
                    zipcode,
                    phone,
                    email,
                    website,
                    capacity,
                    opening_hours,
                    wheelchair,
                    source,
                    tags::jsonb,
                    geom
                FROM {self.kart_schema}.{poi_table_name}
            """
            self.db_rd.perform(insert_restored_data_into_geonode_poi_table_sql)

    @timing
    def insert_gtfs_pt_stops(self, category):
        # read data
        #TODO: harmonize filter of insert_gtfs_pt_stops and read_poi
        try:
            # get the area where source is GTFS
            geom_ref_ids_gtfs_sql = f"""
                SELECT geom_ref_id
                FROM {self.geonode_schema_name}.data_subscription
                WHERE source = 'GTFS'
                AND category = '{category}'
            """
            geom_ref_ids_gtfs = self.db_rd.select(geom_ref_ids_gtfs_sql)

            if geom_ref_ids_gtfs == []:
                print_info("No geom for GTFS pt stops.")
            else:
                # Join all returned values into a comma-separated string with quotes around each value
                geom_ref_ids_gtfs = "', '".join([str(row[0]) for row in geom_ref_ids_gtfs])

                geom_filter_sql = f"""
                    DROP TABLE IF EXISTS geom_filter_gtfs;
                    CREATE TEMP TABLE geom_filter_gtfs AS
                    WITH geom_refs_gtfs_areas AS (
                        SELECT geom_ref_id
                        FROM {self.geonode_schema_name}.data_subscription
                        WHERE source = 'GTFS'
                        AND category = '{category}'
                    )
                    SELECT ST_Union(n.geom) AS geom
                    FROM {self.geonode_schema_name}.geom_ref n
                    WHERE n.id IN (SELECT geom_ref_id FROM geom_refs_gtfs_areas)
                """
                self.db_rd.perform(geom_filter_sql)

            insert_gtfs_sql = f"""
                INSERT INTO {self.geonode_schema_name}.{self.get_kart_poi_table_name(category)}(category, other_categories, name, operator, street, housenumber, zipcode, phone, email, website, capacity, opening_hours, wheelchair, source, tags, geom)
                SELECT
                    category,
                    other_categories,
                    CASE WHEN TRIM(name) = '' THEN NULL ELSE TRIM(name) END,
                    CASE WHEN TRIM(operator) = '' THEN NULL ELSE TRIM(operator) END,
                    CASE WHEN TRIM(street) = '' THEN NULL ELSE TRIM(street) END,
                    CASE WHEN TRIM(housenumber) = '' THEN NULL ELSE TRIM(housenumber) END,
                    CASE WHEN TRIM(zipcode) = '' THEN NULL ELSE TRIM(zipcode) END,
                    CASE WHEN TRIM(phone) = '' THEN NULL ELSE TRIM(phone) END,
                    CASE WHEN octet_length(TRIM(email)) BETWEEN 6 AND 320 AND TRIM(email) LIKE '_%@_%.__%' THEN TRIM(email) ELSE NULL END,
                    CASE WHEN TRIM(website) ~* '^[a-z](?:[-a-z0-9\+\.])*:(?:\/\/(?:(?:%[0-9a-f][0-9a-f]|[-a-z0-9\._~!\$&''\(\)\*\+,;=:@])|[\/\?])*)?' :: TEXT THEN TRIM(website) ELSE NULL END,
                    CASE WHEN TRIM(capacity) ~ '^[0-9]+$' THEN CAST(TRIM(capacity) AS INTEGER) ELSE NULL END,
                    CASE WHEN TRIM(opening_hours) = '' THEN NULL ELSE TRIM(opening_hours) END,
                    CASE WHEN TRIM(wheelchair) IN ('yes', 'no', 'limited') THEN TRIM(wheelchair) ELSE NULL END,
                    source,
                    tags,
                    p.geom
                FROM {self.source_to_table['GTFS']} p, geom_filter_gtfs f
                WHERE ST_Intersects(p.geom, f.geom)
                AND p.category = '{category}';
            """
            self.db_rd.perform(insert_gtfs_sql)

        except Exception as e:
            print(f"An error occurred: {e}")


    @timing
    def read_poi(self, category: str):
        """Method to read the relevant POIs from one category into a temporary table based on the subscription criteria.

        Args:
            category (str): Category of POIs to read
        """
        #TODO: harmonize filter of insert_gtfs_pt_stops and read_poi
        try:
            # get the area where the category is subscribed to either OSM, Overture or OSM_Overture
            geom_ref_ids_subscribe_sql = f"""
                SELECT geom_ref_id
                FROM {self.geonode_schema_name}.data_subscription
                WHERE category = '{category}'
                AND rule = 'subscribe'
            """
            geom_ref_ids_subscribe = self.db_rd.select(geom_ref_ids_subscribe_sql)

            if geom_ref_ids_subscribe == []:
                print_info(f"No geom for subscription of category {category}.")
            else:
                # Join all returned values into a comma-separated string with quotes around each value
                geom_ref_ids_subscribe = "', '".join([str(row[0]) for row in geom_ref_ids_subscribe])

                geom_filter_sql = f"""
                    DROP TABLE IF EXISTS geom_filter_subscribe;
                    CREATE TEMP TABLE geom_filter_subscribe AS
                    WITH geom_refs_excluded_areas AS (
                        SELECT geom_ref_id
                        FROM {self.geonode_schema_name}.data_subscription
                        WHERE category = '{category}'
                        AND rule = 'exclude'
                    ),
                    geom_individual_source AS (
                        SELECT ST_Union(n.geom) AS geom
                        FROM {self.geonode_schema_name}.geom_ref n
                        WHERE n.id IN (SELECT geom_ref_id FROM geom_refs_excluded_areas)
                    ),
                    geom_union_subscribe AS (
                        SELECT ST_Union(g.geom) AS geom
                        FROM {self.geonode_schema_name}.geom_ref g
                        WHERE g.id IN ('{geom_ref_ids_subscribe}')
                    )
                    SELECT
                        CASE
                            WHEN NOT EXISTS (SELECT 1 FROM geom_refs_excluded_areas) THEN
                                s.geom
                            ELSE
                                ST_Difference(s.geom, i.geom)
                        END AS geom
                    FROM geom_union_subscribe s, geom_individual_source i;
                """
                self.db_rd.perform(geom_filter_sql)

            # Get the pois to integrate and check if they fit the geometry filter and validation rules
            # TODO: validation here or at the end of poi preparation or both?

            create_table_sql = POITable(data_set_type=f'{self.table_name}', schema_name = '', data_set_name ='to_seed').create_poi_table(table_type='standard', create_index=False, temporary=True)
            self.db_rd.perform(create_table_sql)

            sql_add_loop_id = f"""
                ALTER TABLE {self.table_name}_to_seed ADD COLUMN IF NOT EXISTS loop_id SERIAL;
                CREATE INDEX ON {self.table_name}_to_seed (loop_id);
            """
            self.db_rd.perform(sql_add_loop_id)

            create_poi_to_integrate_sql = f"""
                INSERT INTO {self.table_name}_to_seed(category, other_categories, name, operator, street, housenumber, zipcode, phone, email, website, capacity, opening_hours, wheelchair, source, tags, geom)
                SELECT
                    category,
                    other_categories,
                    CASE WHEN TRIM(name) = '' THEN NULL ELSE TRIM(name) END,
                    CASE WHEN TRIM(operator) = '' THEN NULL ELSE TRIM(operator) END,
                    CASE WHEN TRIM(street) = '' THEN NULL ELSE TRIM(street) END,
                    CASE WHEN TRIM(housenumber) = '' THEN NULL ELSE TRIM(housenumber) END,
                    CASE WHEN TRIM(zipcode) = '' THEN NULL ELSE TRIM(zipcode) END,
                    CASE WHEN TRIM(phone) = '' THEN NULL ELSE TRIM(phone) END,
                    CASE WHEN octet_length(TRIM(email)) BETWEEN 6 AND 320 AND TRIM(email) LIKE '_%@_%.__%' THEN TRIM(email) ELSE NULL END,
                    CASE WHEN TRIM(website) ~* '^[a-z](?:[-a-z0-9\+\.])*:(?:\/\/(?:(?:%[0-9a-f][0-9a-f]|[-a-z0-9\._~!\$&''\(\)\*\+,;=:@])|[\/\?])*)?' :: TEXT THEN TRIM(website) ELSE NULL END,
                    CASE WHEN TRIM(capacity) ~ '^[0-9]+$' THEN CAST(TRIM(capacity) AS INTEGER) ELSE NULL END,
                    CASE WHEN TRIM(opening_hours) = '' THEN NULL ELSE TRIM(opening_hours) END,
                    CASE WHEN TRIM(wheelchair) IN ('yes', 'no', 'limited') THEN TRIM(wheelchair) ELSE NULL END,
                    source,
                    tags,
                    p.geom
                FROM {self.get_source_table(category)} p, geom_filter_subscribe f
                WHERE ST_Intersects(p.geom, f.geom)
                AND p.source in ('OSM', 'Overture', 'OSM_Overture', 'Overture_OSM')
                AND p.category = '{category}';
            """
            self.db_rd.perform(create_poi_to_integrate_sql)
        except IndexError:
            print(f"No data found for category '{category}'")
        except Exception as e:
            print(f"An error occurred: {e}")

    @timing
    def insert_poi(self, category: str):
        """Inserts the POIs into Geonode POI table.

        Args:
            row_cnt (int): Number of rows to process
            category (str): Category of POIs to read
        """

        insert_into_poi_table_sql = f"""
            INSERT INTO {self.geonode_schema_name}.{self.get_kart_poi_table_name(category)}(category, other_categories, operator, name, street, housenumber, zipcode, phone, email, website, capacity, opening_hours, wheelchair, source, tags, geom)
            SELECT
                category,
                other_categories,
                operator,
                name,
                street,
                housenumber,
                zipcode,
                phone,
                email,
                website,
                capacity,
                opening_hours,
                wheelchair,
                source,
                tags,
                geom
            FROM {self.table_name}_to_seed;
        """
        self.db_rd.perform(insert_into_poi_table_sql)

    def update_date_subscription(self, category: str):
        """Updates the date of the data subscription table for the given category. """

        try:
            sources = self.db_rd.select(f"""SELECT DISTINCT source FROM {self.geonode_schema_name}.data_subscription WHERE rule = 'subscribe'""")

            source_to_date = {
                'OSM': self.osm_data_date.replace(tzinfo=None),
                'Overture': self.db.select(f"SELECT update_time FROM temporal.places_{self.region} LIMIT 1")[0][0].replace(tzinfo=None),
                'OSM_Overture': min(self.osm_data_date.replace(tzinfo=None), self.db.select(f"SELECT update_time FROM temporal.places_{self.region} LIMIT 1")[0][0].replace(tzinfo=None)),
                'GTFS': datetime(2024, 6, 9) #TODO: find better solution
            }

            #TODO: add date for GTFS

            for source in sources:

                source_date = source_to_date[source[0]]

                sql_update_date = f"""
                    UPDATE {self.geonode_schema_name}.data_subscription
                    SET source_date = '{source_date}'
                    WHERE rule = 'subscribe'
                    AND category = '{category}'
                    AND source = '{source[0]}'
                """
                self.db_rd.perform(sql_update_date)

        except KeyError:
            print(f"No date found for source '{source[0]}'")
        except Exception as e:
            print(f"An error occurred: {e}")

    @timing
    def subscribe_poi(self):

        # check if postgres user is geonode_p4b_data -> otherwise break
        if self.db_rd.select("SELECT current_user")[0][0] != 'geonode_p4b_data':
            print("User is not 'geonode_p4b_data'. Please change user to 'geonode_p4b_data' in the .env file.")
            return

        self.migrate_osm_and_overture()

        # 2. if poi tables in Geonode exist -> rename tables (add date (probably date of running/ "putting date out of order")) + create fresh poi tables + apply constraints
        #TODO: do i need to worry about index names?
        geonode_poi_table_names_sql = f"""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = '{self.geonode_schema_name}';
        """
        geonode_poi_table_names = self.db_rd.select(geonode_poi_table_names_sql)
        geonode_poi_table_names = [table[0] for table in geonode_poi_table_names]

        #TODO: technicall not the poi_categories, but the poi top level categories
        for poi_category in self.poi_categories:

            self.prepare_poi_tables(poi_category, geonode_poi_table_names)
            #self.migrate_kart_tables(poi_category)
            # Hardcoded new categories to add
            new_categories = {
                "tourism_leisure": ["theater"],
                "shopping": ["clothes", "shoes", "mini_supermarket", "kiosk"],
                "service"  : ["hairdresser"]
            }

            # childcare and school only individual sources
            if poi_category not in ('childcare', 'school'):

                category_update = self.db.select(f"""SELECT category FROM {self.kart_schema}.poi_categories WHERE table_name = 'poi_{poi_category}'""")
                category_update = "', '".join([str(row[0]) for row in category_update])

                # Get categories to update
                sql_get_categories_to_update = f"""
                    SELECT DISTINCT category
                    FROM {self.geonode_schema_name}.poi_{poi_category};
                """
                categories = self.db_rd.select(sql_get_categories_to_update)
                categories = [category[0] for category in categories]
                
                # Add new categories if defined for this poi_category, only if not already present
                if poi_category in new_categories:
                    categories += [cat for cat in new_categories[poi_category] if cat not in categories]

                for category in categories:

                    data_sources = self.db_rd.select(f"""SELECT DISTINCT source FROM {self.geonode_schema_name}.data_subscription WHERE category = '{category}'""")
                    if any('GTFS' in data_source for data_source in data_sources):
                        self.insert_gtfs_pt_stops(category=category)

                    self.read_poi(category)
                    self.insert_poi(category)
                    self.update_date_subscription(category)

                # additional indices: gist on geom and btree on category
                addtional_indices_sql = f"""
                    CREATE INDEX ON {self.geonode_schema_name}.poi_{poi_category} USING GIST (geom);
                    CREATE INDEX ON {self.geonode_schema_name}.poi_{poi_category} (category);
                """
                self.db_rd.perform(addtional_indices_sql)

            # add comment with creation date
            comment_sql = f"""
                COMMENT ON TABLE {self.geonode_schema_name}.poi_{poi_category} IS 'Created on {datetime.datetime.now().strftime("%Y%m%d")}';
            """
            self.db_rd.perform(comment_sql)


