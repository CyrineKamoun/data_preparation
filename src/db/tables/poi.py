class POITable:
    def __init__(self, data_set_type: str, schema_name: str, data_set_name: str):
        self.data_set_type = data_set_type
        self.data_set_name = data_set_name
        self.schema_name = schema_name

    def create_table(self, table_name: str, category_columns: list) -> str:
        # Common columns for all POI tables
        common_columns = [
            "id SERIAL PRIMARY KEY",
            "name text NULL",
            "operator text NULL",
            "street text NULL",
            "housenumber text NULL",
            "zipcode text NULL",
            "phone text NULL",
            "email text NULL",
            "website text NULL",
            "capacity text NULL",
            "opening_hours text NULL",
            "wheelchair text NULL",
            "source text NULL",
            "tags jsonb DEFAULT '{"'"extended_source"'": {}}'::jsonb",
            "geom geometry NOT NULL"
        ]

        # Combine category columns with common columns
        all_columns = category_columns + common_columns

        # Create SQL query for table creation
        all_columns_str = ",\n".join(all_columns)
        sql_create_table = f"""
            DROP TABLE IF EXISTS {self.schema_name}.{table_name};
            CREATE TABLE {self.schema_name}.{table_name} (
                {all_columns_str}
            );
            CREATE INDEX ON {self.schema_name}.{table_name} USING gist (geom);
            """
        return sql_create_table

    def create_poi_table(self, table_type: str = 'standard') -> str:
        if table_type == "standard":
            table_name = f"{self.data_set_type}_{self.data_set_name}"
            category_columns = [
                "category text NULL",
                "other_categories text[] NULL"
            ]
        elif table_type == "school":
            table_name = f"{self.data_set_type}_school_{self.data_set_name}"
            category_columns = [
                "school_isced_level_1 bool NULL",
                "school_isced_level_2 bool NULL",
                "school_isced_level_3 bool NULL"
            ]
        elif table_type == "childcare":
            table_name = f"{self.data_set_type}_childcare_{self.data_set_name}"
            category_columns = [
                "nursery bool NULL",
                "kindergarten bool NULL",
                "after_school bool NULL"
            ]
        else:
            raise ValueError("Invalid table_type. Supported values are 'standard', 'school', or 'childcare'.")

        return self.create_table(table_name, category_columns)