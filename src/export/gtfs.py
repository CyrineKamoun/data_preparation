import csv
import os
import shutil

from src.config.config import Config
from src.core.config import settings
from src.db.db import Database
from src.db.tables.gtfs import GtfsTables
from src.utils.utils import delete_file, make_dir, print_info


class GTFSExport:
    def __init__(self, db: Database, region: str):
        self.db = db
        self.region = region
        self.config = Config("gtfs", region)
        self.schema = self.config.export["local_gtfs_schema"]

        # Create tables
        gtfs_tables = GtfsTables(self.schema)
        self.create_queries = gtfs_tables.sql_create_table()
        self.select_queries = gtfs_tables.sql_select_table()

        # Output directory
        self.output_dir = os.path.join(settings.OUTPUT_DATA_DIR, "gtfs", region)

    def initialize_output_file(self, table_name: str):
        """Initialize an output file for a GTFS table."""

        file_path = os.path.join(self.output_dir, f"{table_name}.txt")

        # Delete any existing output file
        delete_file(file_path)

        # Create the output directory if it does not exist
        make_dir(self.output_dir)

        # Create the output file
        with open(file_path, 'w', newline='') as f:
            pass

        print_info(f"Initialized output file for table {table_name}.")

    def write_to_output_file(self, table_name: str, h3_3: int=None, overwrite: bool=True):
        """Write a GTFS table to an initialized output file."""

        db_cursor = self.db.cursor()

        # Fetch table data
        sql_query = self.select_queries[table_name]
        if h3_3:
            sql_query = f"{sql_query} WHERE h3_3 = {h3_3}"
        db_cursor.execute(sql_query)
        table_data = db_cursor.fetchall()

        # Fetch column names
        columns = [desc[0] for desc in db_cursor.description]

        # Write the table data to a .txt file
        with open(os.path.join(self.output_dir, f"{table_name}.txt"), 'w' if overwrite else 'a', newline='') as f:
            writer = csv.writer(f)
            if overwrite:
                writer.writerow(columns)
            writer.writerows(table_data)

        db_cursor.close()

    def compress_output_files(self):
        """Compress the output files into a single .zip file."""

        print_info("Compressing output files.")
        os.execvp('zip', ['zip', '-r', '-j', os.path.join(self.output_dir, f"gtfs_{self.region}.zip"), self.output_dir])

    def run(self):
        """Run the gtfs export."""

        # Warn the user if the output directory already contains a .zip file
        if os.path.exists(os.path.join(self.output_dir, f"gtfs_{self.region}.zip")):
            print_info("Warning: The output directory already contains a .zip file. This file will be deleted.")
            delete_file(os.path.join(self.output_dir, f"gtfs_{self.region}.zip"))

        for table in self.create_queries:
            # Initialize the output file
            self.initialize_output_file(table)

            # Write table data to the output file
            print_info(f"Exporting table {table}.")

            if table not in ["shapes", "stop_times"]:
                self.write_to_output_file(table)
            else:
                # Fetch a list of h3_3 cells
                h3_3_list = self.db.select(f"""
                    SELECT DISTINCT h3_3
                    FROM {self.schema}.stop_times;
                """)
                h3_3_list = [val[0] for val in h3_3_list]

                # Write table data for each h3_3 cell
                self.write_to_output_file(table, h3_3=h3_3_list[0])
                for h3_3 in h3_3_list[1:]:
                    self.write_to_output_file(table, h3_3=h3_3, overwrite=False)

        # Compress the output files
        self.compress_output_files()

def export_gtfs(region: str):
    print_info(f"Export GTFS data for the region {region}.")
    db = Database(settings.LOCAL_DATABASE_URI)

    try:
        GTFSExport(db=db, region=region).run()
        db.close()
        print_info("Finished GTFS export.")
    except Exception as e:
        print(e)
        raise e
    finally:
        db.close()
