import os
import subprocess

from src.config.config import Config
from src.core.config import settings
from src.db.db import Database
from src.db.tables.gtfs import GtfsTables
from src.utils.utils import print_info, replace_dir


class GTFSCollection:
    def __init__(self, db: Database, region: str):
        self.db = db
        self.region = region
        self.config = Config("gtfs", region)
        self.network_dir = self.config.preparation["network_dir"]
        self.schema = self.config.preparation["target_schema"]
        self.chunk_size = 1000000
        # Create tables
        gtfs_tables = GtfsTables(self.schema)
        self.create_queries = gtfs_tables.sql_create_table()

    def create_table_schema(self):
        """Create the schema for the gtfs data."""

        print_info("Create schema for gtfs data.")
        # Check if schema exists
        schema_exists = self.db.select(
            f"SELECT EXISTS(SELECT schema_name FROM information_schema.schemata WHERE schema_name = '{self.schema}');"
        )[0][0]
        if not schema_exists:
            print_info(f"Create schema {self.schema}.")
            self.db.perform(f"CREATE SCHEMA {self.schema};")
        else:
            print_info(
                f"Schema {self.schema} already exists. It will be dropped and recreated."
            )
            self.db.perform(f"DROP SCHEMA {self.schema} CASCADE;")
            self.db.perform(f"CREATE SCHEMA {self.schema};")

        print_info("Create tables for gtfs data.")


        for table in self.create_queries:
            self.db.perform(self.create_queries[table])

        # Make stop, stop_times and shapes distributed
        distributed_tables = ["stop_times", "shapes", "stops"]
        print_info(f"Distribute tables using CITUS: {distributed_tables}")
        for table in distributed_tables:
            sql_make_table_distributed = f"SELECT create_distributed_table('{self.schema}.{table}', 'h3_3');"
            self.db.perform(sql_make_table_distributed)


    def split_file(self, table: str, output_dir: str):
        """Split file into chunks and removes header."""

        input_file = os.path.join(settings.INPUT_DATA_DIR, "gtfs", self.network_dir, table + ".txt")
        print_info(
            f"Split file {input_file} into chunks of max. {self.chunk_size} rows."
        )

        # Read header to define column order and remove header afterwards from file
        with open(input_file, "r") as f:
            header = f.readline().strip().replace('"', '').split(",")

        # Check if header is same as column of table
        columns = self.db.select(
            f"SELECT column_name FROM information_schema.columns WHERE table_schema = '{self.schema}' AND table_name = '{table}';"
        )
        # Get all columns besides the loop_id and id column
        columns = [
            column[0]
            for column in columns
            if column[0] not in ["loop_id", "id", "h3_3"]
        ]

        # Compare columns of table with columns of gtfs file. Identify missing columns in gtfs file and in table.
        if set(columns) - set(header):
            print_info(f"Columns of table {table} are missing in the gtfs file.")

        # Columns are missing in table return them to drop them later.
        excess_columns = set(header) - set(columns)
        if excess_columns:
            print_info(f"Columns {excess_columns} are missing in table {table} but are in the gtfs file. Continuing with table columns only.")
            """raise ValueError(
                f"Columns {excess_columns} are missing in table {table} that are in the gtfs file. Import stopped."
            )"""

        # Split file into chunks and use file name as prefix
        subprocess.run(
            [
                "split",
                "-l",
                str(self.chunk_size),
                input_file,
                os.path.join(output_dir, table + "_"),
            ]
        )

        # Get first file
        first_file = os.path.join(output_dir, table + "_aa")

        # Remove header from the first file
        with open(first_file, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
        with open(first_file, "w", encoding="utf-8") as f:
            f.writelines(lines[1:])

        # Clean all split files to remove invalid UTF-8 characters
        for temp_file in os.listdir(output_dir):
            temp_file_path = os.path.join(output_dir, temp_file)
            with open(temp_file_path, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
            with open(temp_file_path, "w", encoding="utf-8") as f:
                f.writelines(lines)
            print_info(f"Cleaned file {temp_file_path}.")

        return header, columns

    def import_file(self, input_dir: str, table: str, header: list, table_columns: list):
        """Import file into database using copy."""

        files = os.listdir(input_dir)
        print_info(f"Importing {len(files)} files for table {table}.")
        cnt = 0
        # Loop over all files and import them one by one
        for file in files:
            # Create temp table
            sql_create_temp_table = f"""
                DROP TABLE IF EXISTS {self.schema}.{table}_temp;
                CREATE UNLOGGED TABLE {self.schema}.{table}_temp
                ({",".join([value + " text" for value in header])})
            """
            self.db.perform(sql_create_temp_table)

            sql_get_column_datatypes = f"""
                SELECT column_name, data_type
                FROM information_schema.columns
                WHERE table_schema = '{self.schema}'
                AND table_name = '{table}';
            """
            columns = self.db.select(sql_get_column_datatypes)
            for column_name, column_type in columns:
                if column_name not in header:
                    continue

                sql_update_column_type = f"""
                    ALTER TABLE {self.schema}.{table}_temp
                    ALTER COLUMN {column_name} TYPE {column_type};
                """ if column_type == "text" else f"""
                    ALTER TABLE {self.schema}.{table}_temp
                    ALTER COLUMN {column_name} TYPE {column_type}
                    USING {column_name}::{column_type};
                """
                self.db.perform(sql_update_column_type)

            # Copy data to temp table
            file_path_postgres = os.path.join("/tmp/gtfs", self.network_dir, "temp", file)
            sql_copy = f"""
                COPY {self.schema}.{table}_temp ({",".join(header)}) FROM '{file_path_postgres}'
                CSV DELIMITER ',' QUOTE '"' ESCAPE '"' ENCODING 'UTF8';
            """
            self.db.perform(sql_copy)

            # Get list of only the data columns that we use
            output_cols = set(table_columns) - (set(table_columns) - set(header))
            output_cols_formatted = ",".join(output_cols)

            # Check if table not shapes or stops
            if table not in ["shapes", "stops", "stop_times"]:
                # Copy data directly from temp table to table
                sql_copy = f"""
                    INSERT INTO {self.schema}.{table} ({output_cols_formatted})
                    SELECT {output_cols_formatted}
                    FROM {self.schema}.{table}_temp;
                """
            elif table == "shapes":
                # Copy data and create geometry
                sql_copy = f"""
                    INSERT INTO {self.schema}.{table} ({output_cols_formatted}, geom, h3_3)
                    SELECT {output_cols_formatted}, ST_SetSRID(ST_MakePoint(shape_pt_lon, shape_pt_lat), 4326) AS geom,
                    basic.to_short_h3_3(h3_lat_lng_to_cell(ST_SetSRID(ST_MakePoint(shape_pt_lon, shape_pt_lat), 4326)::point, 3)::bigint) AS h3_3
                    FROM {self.schema}.{table}_temp;
                """
            elif table == "stops":
                sql_copy = f"""
                    ALTER TABLE {self.schema}.{table}_temp ADD COLUMN h3_3 int;

                    UPDATE {self.schema}.{table}_temp
                    SET h3_3 = grid.h3_3
                    FROM basic.h3_3_grid grid
                    WHERE ST_Intersects(
                        ST_SetSRID(ST_MakePoint(stop_lon, stop_lat), 4326),
                        grid.geom
                    );

                    UPDATE {self.schema}.{table}_temp
                    SET h3_3 = basic.to_short_h3_3(
                        h3_lat_lng_to_cell(ST_SetSRID(ST_MakePoint(stop_lon, stop_lat), 4326)::point, 3)::bigint
                    )
                    WHERE h3_3 IS NULL;

                    INSERT INTO {self.schema}.{table} ({output_cols_formatted}, geom, h3_3)
                    SELECT {output_cols_formatted}, ST_SetSRID(ST_MakePoint(stop_lon::float4, stop_lat::float4), 4326) AS geom, h3_3
                    FROM {self.schema}.{table}_temp;
                """
            elif table == "stop_times":
                # Make temp table logged and create index on stop_id
                self.db.perform(f"ALTER TABLE {self.schema}.{table}_temp SET LOGGED;")
                self.db.perform(f"CREATE INDEX ON {self.schema}.{table}_temp (stop_id);")

                # Get h3_3 from stops table
                columns = ", t.".join(output_cols)
                columns = "t." + columns
                sql_copy = f"""
                    INSERT INTO {self.schema}.{table} ({output_cols_formatted}, h3_3)
                    SELECT {columns}, s.h3_3
                    FROM {self.schema}.{table}_temp t
                    LEFT JOIN {self.schema}.stops s ON t.stop_id = s.stop_id;
                """
            self.db.perform(sql_copy)

            sql_drop_temp_table = f"DROP TABLE IF EXISTS {self.schema}.{table}_temp;"
            self.db.perform(sql_drop_temp_table)

            cnt += 1
            print_info(f"Imported {cnt} of {len(files)} files for table {table}.")

        # Make table logged
        self.db.perform(f"ALTER TABLE {self.schema}.{table} SET LOGGED;")

    def create_indices(self, table: str):
        """Make tables distributed using CITUS and create indices."""

        # Add Constraints
        print_info("Add constraints to tables.")

        if table == "stops":
            sql_command = f"""
                ALTER TABLE {self.schema}.stops ADD PRIMARY KEY (h3_3, stop_id);
                CREATE INDEX ON {self.schema}.stops USING GIST (h3_3, geom);
                CREATE INDEX ON {self.schema}.stops (h3_3, parent_station);
            """
        elif table == "stop_times":
            sql_command = f"""
                ALTER TABLE {self.schema}.stop_times ADD FOREIGN KEY (h3_3, stop_id)
                REFERENCES {self.schema}.stops(h3_3, stop_id);
                CREATE INDEX ON {self.schema}.stop_times (h3_3, stop_id);
                CREATE INDEX ON {self.schema}.stop_times (h3_3, trip_id);
                """
        elif table == "trips":
            sql_command = f"""
                ALTER TABLE {self.schema}.trips ADD PRIMARY KEY (trip_id);
                ALTER TABLE {self.schema}.trips ADD FOREIGN KEY (route_id) REFERENCES {self.schema}.routes(route_id);
                CREATE INDEX ON {self.schema}.trips (shape_id);
                CREATE INDEX ON {self.schema}.trips (route_id);
                CREATE INDEX ON {self.schema}.trips (service_id);
            """
        elif table == "routes":
            sql_command = f"""
                ALTER TABLE {self.schema}.routes ADD PRIMARY KEY (route_id);
            """
        elif table == "shapes":
            sql_command = f"""
                CREATE INDEX ON {self.schema}.shapes (h3_3, shape_id);
                CREATE INDEX ON {self.schema}.shapes USING GIST(h3_3, geom);
            """
        elif table == "calendar":
            sql_command = f"""
                ALTER TABLE {self.schema}.calendar ADD PRIMARY KEY (service_id);
            """
        elif table == "calendar_dates":
            sql_command = f"""
                ALTER TABLE {self.schema}.calendar_dates ADD PRIMARY KEY (service_id, date);
            """
        else:
            return

        # Add indices
        self.db.perform(sql_command)


    def run(self):
        """Run the gtfs collection."""
        self.create_table_schema()

        # Check if for all table there is a gtfs file

        for table in self.create_queries:
            file_dir = os.path.join(settings.INPUT_DATA_DIR, "gtfs", self.network_dir, table + ".txt")
            if not os.path.exists(file_dir):
                raise Exception(f"File {file_dir} not found.")

            # Create temp dir
            temp_dir = os.path.join(settings.INPUT_DATA_DIR, "gtfs", self.network_dir, "temp")
            replace_dir(temp_dir)
            # Split file into chunks
            header, table_columns = self.split_file(table, temp_dir)
            # Import file into database
            self.import_file(temp_dir, table, header, table_columns)
            self.create_indices(table)




def collect_gtfs(region: str):
    print_info(f"Prepare GTFS data for the region {region}.")
    db = Database(settings.LOCAL_DATABASE_URI)

    try:
        GTFSCollection(db=db, region=region).run()
        db.close()
        print_info("Finished GTFS collection.")
    except Exception as e:
        print(e)
        raise e
    finally:
        db.close()
