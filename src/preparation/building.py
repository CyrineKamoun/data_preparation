from src.config.config import Config
from src.core.config import settings
from src.core.enums import (
    BuildingClassificationColumnTypes,
    BuildingClassificationTypes,
)
from src.db.db import Database
from src.utils.utils import print_info


#TODO: Add logic to substract one residential level based on POI intersection
class BuildingPreparation:
    def __init__(self, db: Database, region: str):

        self.db = db
        self.region = region
        # Get config for buildings
        self.config = Config("building", region)
        self.schema = self.config.preparation["schema"]

        # Get variables for the classification
        self.config_classification = self.config.preparation["classification"]

        self.bulk_size = 10000

    def define_residential_status(
        self,
        mask_geom: str,
        column_name: BuildingClassificationColumnTypes,
        classification_data: str,
        classification_type: BuildingClassificationTypes,
    ):
        """Classifies buildings by their residential status.

        Args:
            mask_geom (str): Mask geometry as WKT string.
            column_name (BuildingClassificationColumnTypes): Column that needs to be updated.
            classification_data (str): Data that is used for classification.
            classification_type (BuildingClassificationTypes): Type of classification.
        """

        # Insert buildings query
        sql_create = """
            DROP TABLE IF EXISTS temp_output_%s;
            CREATE TEMPORARY TABLE temp_output_%s AS
        """
        sql_insert = """
            INSERT INTO temporal.building_%s
            SELECT * FROM temp_output_%s;
        """

        # Read buildings for the mask. Make sure that only buildings are read that have their centroid in the mask.
        sql_read_buildings = f"""WITH building_to_check AS
        (
            SELECT s.*
            FROM
            (
                SELECT b.*
                FROM {self.schema}.building_{self.region} b
                WHERE ST_Intersects(b.geom, ST_SETSRID(ST_GEOMFROMTEXT('{mask_geom}'), 4326))
                AND ST_Intersects(ST_CENTROID(b.geom), ST_SETSRID(ST_GEOMFROMTEXT('{mask_geom}'), 4326))
                AND ST_IsValid(b.geom)
            ) s
            LEFT JOIN (SELECT * FROM temporal.building_%s WHERE {column_name} IS NOT NULL) c
            ON s.id = c.id
            WHERE c.id IS NULL
        )"""

        # Classify buildings using different logic depending on the classification type
        if column_name == BuildingClassificationColumnTypes.residential_status:

            if classification_type == BuildingClassificationTypes.attribute:
                    for key, value in self.config_classification[column_name][
                        classification_type
                    ][classification_data].items():
                        sql_classify = f"""
                            {sql_create % (column_name, column_name)}
                            {sql_read_buildings % column_name}
                            SELECT id, building_levels, building_levels_residential, '{key}' AS residential_status, '{classification_data}' AS table_name_classified, geom
                            FROM building_to_check
                            WHERE {classification_data} IN ({str(value)[1:-1]});
                            {sql_insert % (column_name, column_name)}
                            """
                    
                        
                    
                    
            
            elif classification_type == BuildingClassificationTypes.point:

                if (
                    self.config_classification[column_name][classification_type][
                        classification_data
                    ]["count"]
                    == 0
                ):
                    operator = "="
                else:
                    operator = "<="

                if 'where' in str.lower(self.config_classification[column_name][classification_type][classification_data]["query"]):
                    where_clause = 'AND'
                else:
                    where_clause = 'WHERE'

                # Check the number of points in the mask that are within each building
                sql_classify = f"""
                    {sql_create % (column_name, column_name)}
                    {sql_read_buildings % column_name}
                    ,classified_buildings AS
                    (
                        SELECT b.id, building_levels, building_levels_residential,
                        CASE WHEN {self.config_classification[column_name][classification_type][classification_data]["count"]} {operator} j.count
                        THEN {self.config_classification[column_name][classification_type][classification_data]["value"]}
                        ELSE NULL END AS {column_name}, geom
                        FROM building_to_check b
                        CROSS JOIN LATERAL
                        (
                            {self.config_classification[column_name][classification_type][classification_data]["query"]}
                            {where_clause} ST_Intersects(b.geom, p.geom)
                        ) j
                    )
                    SELECT id, building_levels, building_levels_residential, {column_name} AS residential_status, '{classification_data}' AS table_name_classified, geom
                    FROM classified_buildings
                    WHERE {column_name} IS NOT NULL;
                    {sql_insert % (column_name, column_name)}
                """
                self.db.perform(sql_classify)

            elif classification_type == BuildingClassificationTypes.polygon:
                # Check share of intersection between each building and the mask data
                sql_classify = f"""
                    {sql_create % (column_name, column_name)}
                    {sql_read_buildings % column_name}
                    , classified_buildings AS
                    (
                        SELECT b.id, building_levels, building_levels_residential,
                        CASE WHEN {self.config_classification[column_name][classification_type][classification_data]["share"]} <= j.share
                        THEN {self.config_classification[column_name][classification_type][classification_data]["value"]}
                        ELSE NULL END AS {column_name}, geom
                        FROM building_to_check b
                        CROSS JOIN LATERAL
                        (
                            SELECT CASE WHEN SUM(share) > 1 THEN 1 ELSE SUM(share) END AS share
                            FROM
                            (
                                SELECT CASE WHEN ST_CONTAINS(p.geom, b.geom) THEN 1
                                ELSE ST_AREA(ST_INTERSECTION(p.geom, b.geom)) / ST_AREA(b.geom) END AS share
                                FROM
                                (
                                    {self.config_classification[column_name][classification_type][classification_data]["query"]}
                                ) p
                                WHERE ST_Intersects(b.geom, p.geom)
                            ) s
                        )  j
                    )
                    SELECT id, building_levels, building_levels_residential, {column_name} AS residential_status, '{classification_data}' AS table_name_classified, geom
                    FROM classified_buildings
                    WHERE {column_name} IS NOT NULL;
                    {sql_insert % (column_name, column_name)}
                """
                self.db.perform(sql_classify)

        elif (
            column_name == BuildingClassificationColumnTypes.building_levels_residential
        ):

            if classification_type == BuildingClassificationTypes.point:

                sql_classify = f"""
                    {sql_create % (column_name, column_name)}
                    {sql_read_buildings % column_name}
                    , classified_buildings AS
                       (
                        SELECT b.id, building_levels, building_levels - j.substract AS building_levels_residential,
                        CASE WHEN building_levels - j.substract = 0 THEN 'no_residents' ELSE NULL END AS residential_status, geom
                        FROM building_to_check b
                        CROSS JOIN LATERAL
                        (
                            {self.config_classification[column_name][classification_type][classification_data]["query"]}
                        ) j
                    )
                    SELECT id, building_levels, {column_name} AS building_levels_residential, residential_status, '{classification_data}' AS table_name_classified, geom
                    FROM classified_buildings
                    WHERE {column_name} IS NOT NULL;
                    {sql_insert % (column_name, column_name)}
                """
                self.db.perform(sql_classify)

    def get_processing_units(self, study_area_ids: list[int]) -> list[str]:
        """Get the processing units for the study area.

        Args:
            study_area_ids (list[int]): _description_

        Returns:
            list[str]: Processing units as WKT strings.
        """

        # Get processing units from study area by creating rectangular grid and make sure they intersect the study area
        processing_units = self.db.select(
            f"""
            WITH grids AS
            (
                SELECT DISTINCT ST_TRANSFORM(s.geom, 4326) AS geom
                FROM study_area, ST_SquareGrid(5000, ST_TRANSFORM(geom, 3857)) s
                WHERE id IN ({str(study_area_ids)[1:-1]})
            )
            SELECT ST_AsText(j.geom)
            FROM grids g
            CROSS JOIN LATERAL
            (
                SELECT g.geom
                FROM public.study_area s
                WHERE ST_Intersects(g.geom, s.geom)
            ) j
        """
        )
        processing_units = [p[0] for p in processing_units]

        return processing_units

    def run(self):
        """Run the building classification.
        """

        # Create temporary table for classified buildings
        for column_name in self.config_classification:
            # Create one table per classified column
            sql_building_classified = f"""
            DROP TABLE IF EXISTS temporal.building_{column_name};
            CREATE TABLE temporal.building_{column_name}
            (
                id integer,
                building_levels integer,
                building_levels_residential integer,
                residential_status text,
                table_name_classified text,
                geom geometry
            );
            CREATE INDEX ON temporal.building_{column_name} (id);
            """
            self.db.perform(sql_building_classified)

        # Get processing units
        processing_units = self.get_processing_units(study_area_ids=self.config.preparation["study_area_ids"])

        # Classify buildings using the config file
        for i in range(len(processing_units)):
            processing_unit = processing_units[i]
            print_info(f"Calculating for processing unit: {i} of {len(processing_units)}")
            for column_name in self.config_classification:
                for classification_type in self.config_classification[column_name]:

                    for classification_data in self.config_classification[column_name][
                        classification_type
                    ]:
                        self.define_residential_status(
                            mask_geom=processing_unit,
                            column_name=column_name,
                            classification_data=classification_data,
                            classification_type=classification_type,
                        )

        for column_name in self.config_classification:
            # Create primary key and GIST index
            self.db.perform(
                f"""
                ALTER TABLE temporal.building_{column_name}  ADD PRIMARY KEY (id);
                CREATE INDEX ON temporal.building_{column_name}  USING GIST (geom);
                ALTER TABLE temporal.building_{column_name}  ADD id_loop serial;
                CREATE INDEX ON temporal.building_{column_name}  (id_loop);"""
            )

            get_max_id_classified = self.db.select(
                f"SELECT last_value FROM temporal.building_{column_name}_id_loop_seq;"
            )

            # Update building table in bulks of self.bulk_size
            for i in range(0, get_max_id_classified[0][0], self.bulk_size):
                print_info(
                    f"Updating building table {i} to {i+self.bulk_size} that are classified"
                )
                sql_update_building_table = f"""
                    UPDATE {self.schema}.building_{self.region}  b
                    SET {column_name} = t.{column_name}
                    FROM temporal.building_{column_name} t
                    WHERE b.id = t.id
                    AND t.id_loop BETWEEN {i} AND {i+self.bulk_size};
                """
                self.db.perform(sql_update_building_table)


        get_max_id_building = self.db.select(
            f"SELECT MAX(id) FROM {self.schema}.building_{self.region} ;"
        )
        for i in range(0, get_max_id_building[0][0], self.bulk_size):
            print_info(
                f"Updating building table {i} to {i+self.bulk_size} that are not classified"
            )
            # Update remaining buildings as with residents
            sql_update_remaining_buildings = f"""
                UPDATE {self.schema}.building_{self.region}  b
                SET residential_status = 'with_residents'
                WHERE b.residential_status IS NULL
                AND b.id BETWEEN {i} AND {i+self.bulk_size};
            """
            self.db.perform(sql_update_remaining_buildings)

        # Update building_levels_residential in bulks of self.bulk_size
        for i in range(0, get_max_id_building[0][0], self.bulk_size):
            print_info(
                f"Updating building_levels_residential {i} to {i+self.bulk_size}"
            )
            sql_update_building_levels_residential = f"""
                UPDATE {self.schema}.building_{self.region}  b
                SET building_levels_residential = b.building_levels,
                area = ST_Area(b.geom::geography),
                gross_floor_area_residential = ST_Area(b.geom::geography) * b.building_levels
                WHERE b.id BETWEEN {i} AND {i+self.bulk_size}
            """
            self.db.perform(sql_update_building_levels_residential)

        self.db.conn.close()


def prepare_building(region: str):

    db_rd = Database(settings.LOCAL_DATABASE_URI)
    building_preparation = BuildingPreparation(db=db_rd, region=region)
    building_preparation.run()


if __name__ == "__main__":
    prepare_building("de")
