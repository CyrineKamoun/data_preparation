import csv
import os
import random
import shutil
import string
import subprocess
import time
from functools import wraps
from io import StringIO
from pathlib import Path
from typing import Any
from urllib.request import urlopen

import numpy as np
import polars as pl
from cdifflib import CSequenceMatcher
from rich import print as print
from shapely.geometry import MultiPolygon
from sqlalchemy import text

from src.core.config import settings
from src.core.enums import DumpType, IfExistsType, TableDumpFormat
from src.db.db import Database


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time.time()
        result = f(*args, **kw)
        te = time.time()
        total_time = te - ts
        if total_time > 1:
            total_time = round(total_time, 2)
            total_time_string = f"{total_time} seconds"
        elif total_time > 0.001:
            time_miliseconds = int((total_time) * 1000)
            total_time_string = f"{time_miliseconds} miliseconds"
        else:
            time_microseconds = int((total_time) * 1000000)
            total_time_string = f"{time_microseconds} microseconds"
        print_info(f"func: {f.__name__} took: {total_time_string}")

        return result

    return wrap


def make_dir(dir_path: str):
    """Creates a new directory if it doesn't already exist"""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def delete_file(file_path: str) -> None:
    """Delete file from disk."""
    try:
        os.remove(file_path)
    except OSError as e:
        pass


def delete_dir(dir_path: str) -> None:
    """Delete file from disk."""
    try:
        shutil.rmtree(dir_path)
    except OSError as e:
        pass


def replace_dir(dir_path: str) -> None:
    """Delete folder from disk and recreate empty one with same path."""
    delete_dir(dir_path)
    os.mkdir(dir_path)


def print_hashtags():
    print(
        "#################################################################################################################"
    )


def print_separator_message(message: str):
    print_hashtags()
    print_info(message)
    print_hashtags()


def print_info(message: str):
    print(f"[bold green]INFO[/bold green]: {message}")


def print_error(message: str):
    print(f"[bold red]ERROR[/bold red]: {message}")


def print_warning(message: str):
    print(f"[red magenta]WARNING[/red magenta]: {message}")


def download_link(directory: str, link: str, new_filename: str = None):
    if new_filename is not None:
        filename = new_filename
    else:
        filename = os.path.basename(link)

    download_path = Path(directory) / filename
    with urlopen(link) as image, download_path.open("wb") as f:
        f.write(image.read())

    print_info(f"Download ended for {link}")


def check_string_similarity(
    input_value: str, match_values: list[str], target_ratio: float
) -> bool:
    """Check if a string is similar to a list of strings.

    Args:
        input_value (str): Input value to check.
        match_values (list[str]): List of strings to check against.
        target_ratio (float): Target ratio to match.

    Returns:
        bool: True if the input value is similar to one of the match values.
    """

    for match_value in match_values:
        if input_value in match_value or match_value in input_value:
            return True
        elif CSequenceMatcher(None, input_value, match_value).ratio() >= target_ratio:
            return True
        else:
            pass
    return False


def check_string_similarity_bulk(
    input_value: str, match_dict: dict, target_ratio: float
) -> bool:
    """Check if a string is similar to a dictionary with lists of strings.

    Args:
        input_value (str): Input value to check.
        match_dict (dict): Dictionary with lists of strings to check against.
        target_ratio (float): Target ratio to match.

    Returns:
        bool: True if the input value is similar to one of the match values.
    """
    if input_value is None:
        return False

    for key, match_values in match_dict.items():
        if check_string_similarity(
            match_values=match_values,
            input_value=input_value.lower(),
            target_ratio=target_ratio,
        ):
            return True
    return False


vector_check_string_similarity_bulk = np.vectorize(check_string_similarity_bulk)


def create_pgpass(db_config):
    """Creates pgpass file for specified DB config

    Args:
        db_config: Database configuration.
    """
    db_name = db_config.path[1:]
    delete_file(f"""~/.pgpass_{db_name}""")
    os.system(
        "echo "
        + ":".join(
            [
                db_config.host,
                str(db_config.port),
                db_name,
                db_config.user,
                db_config.password,
            ]
        )
        + f""" > ~/.pgpass_{db_name}"""
    )
    os.system(f"""chmod 0600  ~/.pgpass_{db_name}""")


def check_table_exists(db, table_name: str, schema: str) -> bool:
    """_summary_

    Args:
        db (_type_): _description_
        table_name (str): _description_
        schema (str): _description_

    Returns:
        bool: _description_
    """    

    check_if_exists = db.select(
        f"""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE  table_schema = '{schema}'
                AND    table_name   = '{table_name}'
            );"""
    )
    return check_if_exists[0][0]


def create_table_dump(
    db_config: dict, schema: str, table_name: str, dump_type: DumpType = DumpType.all
):
    """Create a dump from a table

    Args:
        db_config (str): Database configuration dictionary.
        table_name (str): Specify the table name including the schema.
        schema (str): Specify the schema.
        data_only (bool, optional): Is it a data only dump. Defaults to False.
    """

    try:
        dir_output = os.path.join(settings.OUTPUT_DATA_DIR, table_name + ".dump")

        # Delete the file if it already exists
        delete_file(dir_output)

        # Set the password to the environment variable
        os.environ["PGPASSWORD"] = db_config.password
        # Construct the pg_dump command
        command = [
            "pg_dump",
            "-h",
            db_config.host,
            "-p",
            db_config.port,
            "-U",
            db_config.user,
            "-d",
            db_config.path[1:],
            "-t",
            f"{schema}.{table_name}",
            "-F",
            "c",
            "-f",
            dir_output,
            "--no-owner",
        ]
        # Append to the end of the command if it is a data only dump
        if dump_type.value == DumpType.data.value:
            command.append("--data-only")
        elif dump_type.value == DumpType.schema.value:
            command.append("--schema-only")
        elif dump_type.value == DumpType.all.value:
            pass
        else:
            raise ValueError(f"Dump type {dump_type} not supported")
        
        # Run the pg_dump command and capture the output
        output = subprocess.check_output(command, stderr=subprocess.STDOUT)
        print_info(f"Successfully dumped {schema}.{table_name} to {dir_output}")
    except Exception as e:
        print_warning(f"The following exeption happened when dumping {table_name}: {e}")


def restore_table_dump(
    db_config: dict, schema: str, table_name: str, dump_type: DumpType = DumpType.all
):
    """Restores the dump from a table

    Args:
        db_config (dict): Database configuration dictionary.
        table_name (str): Specify the table name including the schema.
        data_only (bool, optional): Is it a data only dump. Defaults to False.

    Raises:
        ValueError: If the file is not found.
    """

    # Define the output directory
    dir_output = os.path.join(settings.OUTPUT_DATA_DIR, table_name + ".dump")
    # Check if the file exists
    if not os.path.isfile(dir_output):
        raise ValueError(f"File {dir_output} does not exist")
    try:
        # Set the password to the environment variable
        os.environ["PGPASSWORD"] = db_config.password

        # Construct the pg_dump command
        command = [
            "pg_restore",
            "-h",
            db_config.host,
            "-p",
            db_config.port,
            "-U",
            db_config.user,
            "-d",
            db_config.path[1:],
            "--no-owner",
            "--no-privileges",
            dir_output,
        ]
        # Append to -2 position of the command if it is a data only dump
        if dump_type == DumpType.data.value:
            command.insert(-2, "--data-only")
        elif dump_type == DumpType.schema.value:
            command.insert(-2,"--schema-only")
        elif dump_type == DumpType.all.value:
            pass
        else:
            raise ValueError(f"Dump type {dump_type} not supported")
        
        # Run the command
        output = subprocess.check_output(command, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        print_warning(
            f"The following exception happened when restoring {table_name}: {e.output.decode()}"
        )
    else:
        print_info(f"Successfully restored {table_name}.dump from {dir_output}")


def create_table_schema(db: Database, table_full_name: str):
    """Function that creates a table schema from a database dump.

    Args:
        db (Database): Database connection class.
        table_full_name (str): Name with the schema of the table (e.g. basic.poi).
    """
    db_config = db.db_config
    db.perform(query="CREATE SCHEMA IF NOT EXISTS basic;")
    db.perform(query="CREATE SCHEMA IF NOT EXISTS extra;")
    db.perform(query="DROP TABLE IF EXISTS %s" % table_full_name)
    table_name = table_full_name.split(".")[1]
    # Set the password to the environment variable
    os.environ["PGPASSWORD"] = db_config.password
    subprocess.run(
        f'pg_restore -U {db_config.user} --schema-only -h {db_config.host} --no-owner -n basic -d {db_config.path[1:]} -t {table_name} {"/app/src/data/input/dump.tar"}',
        shell=True,
        check=True,
    )
    # # TODO: Temp fix here only to convert poi.id a serial instead of integer
    db.perform(
        f"""
        ALTER TABLE {table_full_name} DROP COLUMN IF EXISTS id;
        ALTER TABLE {table_full_name} ADD COLUMN id SERIAL;
        """
    )


def create_standard_indices(db: Database, table_full_name: str):
    """Create standard indices for the database on the id and geometry column.

    Args:
        db (Database): Database connection class.
    """

    db.perform(
        f"""
        ALTER TABLE {table_full_name} ADD PRIMARY KEY (id);
        CREATE INDEX IF NOT EXISTS {table_full_name.replace('.', '_')}_geom_idx ON {table_full_name} USING GIST (geom);
        """
    )


def download_dir(self, prefix, local, bucket, client):
    """Downloads data directory from AWS S3
    Args:
        prefix (str): Path to the directory in S3
        local (str): Path to the local directory
        bucket (str): Name of the S3 bucket
        client (obj): S3 client object
    """
    keys = []
    dirs = []
    next_token = ""
    base_kwargs = {
        "Bucket": bucket,
        "Prefix": prefix,
    }
    while next_token is not None:
        kwargs = base_kwargs.copy()
        if next_token != "":
            kwargs.update({"ContinuationToken": next_token})
        results = client.list_objects_v2(**kwargs)
        contents = results.get("Contents")
        for i in contents:
            k = i.get("Key")
            if k[-1] != "/":
                keys.append(k)
            else:
                dirs.append(k)
        next_token = results.get("NextContinuationToken")
    for d in dirs:
        dest_pathname = os.path.join(local, d)
        if not os.path.exists(os.path.dirname(dest_pathname)):
            os.makedirs(os.path.dirname(dest_pathname))
    for k in keys:
        dest_pathname = os.path.join(local, k)
        if not os.path.exists(os.path.dirname(dest_pathname)):
            os.makedirs(os.path.dirname(dest_pathname))
        client.download_file(bucket, k, dest_pathname)


def upload_dir(self, prefix, local, bucket, client):
    """Uploads data directory to AWS S3
    Args:
        prefix (str): Path to the directory in S3
        local (str): Path to the local directory
        bucket (str): Name of the S3 bucket
        client (obj): S3 client object
    """
    for root, dirs, files in os.walk(local):
        for filename in files:
            # construct the full local path
            local_path = os.path.join(root, filename)


def parse_poly(dir):
    """Parse an Osmosis polygon filter file.
    Based on: https://wiki.openstreetmap.org/wiki/Osmosis/Polygon_Filter_File_Python_Parsing
    Args:
        dir (str): Path to the polygon filter file.

    Returns:
        (shapely.geometry.multipolygon): Returns the polygon in the poly foramat as a shapely multipolygon.
    """
    in_ring = False
    coords = []
    with open(dir, "r") as polyfile:
        for index, line in enumerate(polyfile):
            if index == 0:
                # first line is junk.
                continue

            elif index == 1:
                # second line is the first polygon ring.
                coords.append([[], []])
                ring = coords[-1][0]
                in_ring = True

            elif in_ring and line.strip() == "END":
                # we are at the end of a ring, perhaps with more to come.
                in_ring = False

            elif in_ring:
                # we are in a ring and picking up new coordinates.
                ring.append(list(map(float, line.split())))

            elif not in_ring and line.strip() == "END":
                # we are at the end of the whole polygon.
                break

            elif not in_ring and line.startswith("!"):
                # we are at the start of a polygon part hole.
                coords[-1][1].append([])
                ring = coords[-1][1][-1]
                in_ring = True

            elif not in_ring:
                # we are at the start of a polygon part.
                coords.append([[], []])
                ring = coords[-1][0]
                in_ring = True

        return MultiPolygon(coords)


# Copied from https://pynative.com/python-generate-random-string/
def get_random_string(length):
    # choose from all lowercase letter
    letters = string.ascii_lowercase
    result_str = "".join(random.choice(letters) for i in range(length))
    return result_str


def psql_insert_copy(table, conn, keys, data_iter):
    """
    Execute SQL statement inserting data

    Parameters
    ----------
    table : pandas.io.sql.SQLTable
    conn : sqlalchemy.engine.Engine or sqlalchemy.engine.Connection
    keys : list of str
        Column names
    data_iter : Iterable that iterates the values to be inserted
    """
    # gets a DBAPI connection that can provide a cursor
    dbapi_conn = conn.connection
    with dbapi_conn.cursor() as cur:
        s_buf = StringIO()
        writer = csv.writer(s_buf)
        writer.writerows(data_iter)
        s_buf.seek(0)

        columns = ", ".join(['"{}"'.format(k) for k in keys])

        if table.schema:
            table_name = "{}.{}".format(table.schema, table.name)
        else:
            table_name = table.name

        if "this_is_the_geom_column" in keys:
            columns.replace(
                "this_is_the_geom_column", "ST_GEOMFROMTEXT(this_is_the_geom_column)"
            )
        if "this_is_the_jsonb_column" in keys:
            columns.replace(
                "this_is_the_jsonb_column", "this_is_the_jsonb_column::jsonb"
            )

        sql = "COPY {} ({}) FROM STDIN WITH CSV".format(table_name, columns)
        cur.copy_expert(sql=sql, file=s_buf)


# TODO: Finish docstring and add comments. Check error handling
def polars_df_to_postgis(
    engine,
    df: pl.DataFrame,
    table_name: str,
    schema: str = "public",
    if_exists: IfExistsType = "replace",
    geom_column: str = "geom",
    srid: int = 4326,
    create_geom_index: bool = True,
    jsonb_column: str = False,
):
    """Blazing fast method to import a polars DataFrame into a PostGIS database with geometry and JSONB column.
    Avoid using 'this_is_the_geom_column' and 'this_is_the_jsonb_column' as column names in the dataframe as they are reserved for the geometry and JSONB columns during the import.

    Args:
        engine (SQLAlchemy): SQLAlchemy engine
        df (pl.DataFrame): Polars DataFrame
        table_name (str): Name of the table to be created
        schema (str, optional): Schema name. Defaults to "public".
        if_exists (IfExistsType, optional): What should happen if table exist. There are the options: 'fail', 'append', 'replace'. Defaults to "replace".
        geom_column (str, optional): What is the name of the geometry column in the dataframe. The geometry column should be a WKT string. The same name will also be used in the PostGIS table. Defaults to "geom".
        srid (int, optional): What is the SRID of the geom. Defaults to 4326.
        create_geom_index (bool, optional): Should a GIST index be created on the geometry. Defaults to True.
        jsonb_column (str, optional): Add the name of column that should added as JSONB. Defaults to False.

    Raises:
        ValueError: Name of the geometry column is not in the dataframe
        ValueError: Name of the JSONB column is not in the dataframe
        ValueError: If the if_exists parameter is 'fail'
    """

    # make a connection
    df_pd = df.to_pandas()
    db = engine.connect()
    # Check if table should be created or appended
    if if_exists == IfExistsType.replace.value:
        df_pd.head(0).to_sql(
            table_name,
            engine,
            method=psql_insert_copy,
            index=False,
            if_exists=IfExistsType.replace.value,
            chunksize=1,
            schema=schema,
        )
        print_info("Table {} will be created in schema {}.".format(table_name, schema))

        columns_to_rename = {}
        # Check if geom column exists and if it should be converted to geometry
        if geom_column in df_pd.columns and geom_column is not None:
            # Get a uuid column
            random_column_name_geom = "this_is_the_geom_column"

            db.execute(
                text(
                    "ALTER TABLE {}.{} RENAME COLUMN {} TO {};".format(
                        schema, table_name, geom_column, random_column_name_geom
                    )
                )
            )

            db.execute(
                text(
                    "ALTER TABLE {}.{} ALTER COLUMN {} TYPE geometry;".format(
                        schema, table_name, random_column_name_geom
                    )
                )
            )
            db.execute(
                text(
                    "SELECT UpdateGeometrySRID('{}','{}','{}', {})".format(
                        schema, table_name, random_column_name_geom, srid
                    )
                )
            )
            columns_to_rename[geom_column] = random_column_name_geom

        elif geom_column not in df_pd.columns and geom_column is not None:
            raise ValueError("Spefified column for Geometry not found in DataFrame")

        if jsonb_column in df_pd.columns and jsonb_column is not None:
            random_column_name_jsonb = "this_is_the_jsonb_column"
            db.execute(
                text(
                    "ALTER TABLE {}.{} RENAME COLUMN {} TO {};".format(
                        schema, table_name, jsonb_column, random_column_name_jsonb
                    )
                )
            )

            db.execute(
                text(
                    "ALTER TABLE {}.{} ALTER COLUMN {} TYPE JSONB USING {}::jsonb".format(
                        schema,
                        table_name,
                        random_column_name_jsonb,
                        random_column_name_jsonb,
                    )
                )
            )
            columns_to_rename[jsonb_column] = random_column_name_jsonb

        elif jsonb_column not in df_pd.columns and jsonb_column is not None:
            raise ValueError("Spefified column for JSONB not found in DataFrame")

    elif if_exists.value == IfExistsType.append.value:
        print_info("Table {} in schema {} already exists".format(table_name, schema))
    elif if_exists.value == IfExistsType.fail.value:
        raise ValueError(
            "Table {} in schema {} already exists".format(table_name, schema)
        )

    df_pd = df_pd.rename(columns=columns_to_rename)
    # Insert data into table
    df_pd.to_sql(
        table_name,
        engine,
        method=psql_insert_copy,
        index=False,
        if_exists="append",
        chunksize=10000,
        schema=schema,
    )

    # Rename columns back to original names
    if "this_is_the_geom_column" in df_pd.columns:
        db.execute(
            text(
                "ALTER TABLE {}.{} RENAME COLUMN this_is_the_geom_column TO {};".format(
                    schema, table_name, geom_column
                )
            )
        )
    if "this_is_the_jsonb_column" in df_pd.columns:
        db.execute(
            text(
                "ALTER TABLE {}.{} RENAME COLUMN this_is_the_jsonb_column TO {};".format(
                    schema, table_name, jsonb_column
                )
            )
        )

    # Create index on geom column if it does not exist and is desired
    if create_geom_index == True:
        idx = db.execute(
            text(
                "SELECT indexdef FROM pg_indexes WHERE tablename = '{}';".format(
                    table_name
                )
            )
        )

        if "gist" not in idx and "(geom)" not in idx:
            print_info("Creating index on geom column")
            db.execute(
                text(
                    "CREATE INDEX ON {}.{} USING GIST (geom);".format(
                        schema, table_name
                    )
                )
            )
        else:
            print_info("GIST-Index on geom column already exists")

    # Close connection
    db.close()


def osm_crop_to_polygon(orig_file_path: str, dest_file_path: str, poly_file_path: str):
    """
    Crops OSM data as per polygon file

    Args:
            orig_file_path (str): Path to the input OSM data file
            dest_file_path (str): Path to the output OSM data file (incl. filename with extension) where OSM data is to be written
            poly_file_path (str): Path to a polygon filter file (as per the format described here: https://wiki.openstreetmap.org/wiki/Osmosis/Polygon_Filter_File_Format)
    """

    subprocess.run(
        f"osmconvert {orig_file_path} -B={poly_file_path} --complete-ways --drop-relations --drop-author --drop-version -o={dest_file_path}",
        shell=True,
        check=True,
    )


def osm_filter_to_highways(orig_file_path: str, dest_file_path: str):
    """
    Crops OSM data as per polygon file

    Args:
            orig_file_path (str): Path to the input OSM data file
            dest_file_path (str): Path to the output OSM data file (incl. filename with extension) where OSM data is to be written
    """

    subprocess.run(
        f'osmfilter {orig_file_path} --keep="highway=" -o={dest_file_path}',
        shell=True,
        check=True,
    )


def osm_convert(orig_file_path: str, dest_file_path: str):
    """
    Crops OSM data as per polygon file

    Args:
            orig_file_path (str): Path to the input OSM data file
            dest_file_path (str): Path to the output OSM data file (incl. filename with extension) where OSM data is to be written
    """

    subprocess.run(
        f'osmconvert {orig_file_path} -o={dest_file_path}',
        shell=True,
        check=True,
    )


def osm_generate_polygon(db_rd, geom_query: str, dest_file_path: str):
    """
    Generates a polygon filter file for cropping OSM data

    Args:
        db_rd (Database): A database connection object
        geom_query (str): The query to be run for retrieving geometry data for a region (returned column must be named "geom")
        dest_file_path (str): Path to the output file (incl. filename with extension ".poly") where polygon data is to be written
    """
    
    coordinates = db_rd.select(f"""SELECT ST_x(coord.geom), ST_y(coord.geom)
                                        FROM (
                                            SELECT (ST_dumppoints(geom_data.geom)).geom
                                            FROM (
                                                {geom_query}
                                            ) geom_data
                                        ) coord;"""
                                )
    with open(dest_file_path, "w") as file:
        file.write("1\n")
        file.write("polygon\n")
        file.write("\n".join([f" {i[0]} {i[1]}" for i in coordinates]))
        file.write("\nEND\nEND")


def get_region_bbox_coords(db: Database, geom_query: str):
    """Get the bounding coordinates of a specified geometry."""

    if not geom_query:
        return None

    sql_get_region_bbox_coords = f"""
        SELECT
            ST_XMin(ST_Envelope(geom)) AS xmin,
            ST_YMin(ST_Envelope(geom)) AS ymin,
            ST_XMax(ST_Envelope(geom)) AS xmax,
            ST_YMax(ST_Envelope(geom)) AS ymax
        FROM
            ({geom_query}) sub;
    """
    bbox_coords = db.select(sql_get_region_bbox_coords)[0]
    return {
        "xmin": bbox_coords[0],
        "ymin": bbox_coords[1],
        "xmax": bbox_coords[2],
        "ymax": bbox_coords[3]
    }
