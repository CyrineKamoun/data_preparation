class POITable:
    def __init__(self, data_set_type: str, schema_name: str, data_set_name: str):
        self.data_set_type = data_set_type
        self.data_set_name = data_set_name
        self.schema_name = schema_name

    def create_table(self, table_name: str, category_columns: list, temporary: bool = False, create_index: bool = True) -> str:
        # Common columns for all POI tables
        common_columns = [
            "id SERIAL PRIMARY KEY",
            "name text NULL",
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
        table_type = "TEMPORARY" if temporary else ""
        table_name_with_schema = f"{self.schema_name}.{table_name}" if not temporary else table_name
        drop_table_name = table_name_with_schema if not temporary else table_name
        index_statement = f"CREATE INDEX ON {table_name_with_schema} USING gist (geom);" if create_index else ""
        sql_create_table = f"""
            DROP TABLE IF EXISTS {drop_table_name};
            CREATE {table_type} TABLE {table_name_with_schema} (
                {all_columns_str}
            );
            {index_statement}
            """
        return sql_create_table

    def create_poi_table(self, table_type: str = 'standard', temporary: bool = False, create_index: bool = True) -> str:
        if table_type == "standard":
            table_name = f"{self.data_set_type}_{self.data_set_name}"
            category_columns = [
                "category text NULL",
                "other_categories text[] NULL",
                "operator text NULL"
            ]
        elif table_type == "school":
            table_name = f"{self.data_set_type}_{self.data_set_name}"
            category_columns = [
                "school_isced_level_1 bool NULL",
                "school_isced_level_2 bool NULL",
                "school_isced_level_3 bool NULL",
                "operator text NULL"
            ]
        elif table_type == "childcare":
            table_name = f"{self.data_set_type}_{self.data_set_name}"
            category_columns = [
                "nursery bool NULL",
                "kindergarten bool NULL",
                "after_school bool NULL",
                "min_age int NULL",
                "max_age int NULL",
                "carrier text NULL",
                "carrier_type text NULL"
            ]
        elif table_type == "transport":
            table_name = f"{self.data_set_type}_{self.data_set_name}"
            category_columns = [
                "stop_id text NOT NULL",
                "category text NOT NULL",
                "bus text NOT NULL",
                "tram text NOT NULL",
                "metro text NOT NULL",
                "rail text NOT NULL",
                "other text NOT NULL",
            ]
        else:
            raise ValueError("Invalid table_type. Supported values are 'standard', 'school', 'childcare', or 'transport.")

        return self.create_table(table_name, category_columns, temporary, create_index)
