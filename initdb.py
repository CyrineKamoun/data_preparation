import os
import subprocess

from src.core.config import settings
from src.db.db import Database
from src.utils.utils import print_error, print_info


def create_db():
    """Create DataPreparation database."""
    # Connect to default database
    db_name = settings.POSTGRES_DB
    subprocess.run(["psql", "-U", "rds", "-d", "postgres", "-c", f"CREATE DATABASE {db_name};"])


def init_db(db):
    # Create extension
    db.perform("CREATE EXTENSION IF NOT EXISTS postgis;")
    db.perform("CREATE EXTENSION IF NOT EXISTS postgis_raster;")
    db.perform("CREATE EXTENSION IF NOT EXISTS hstore;")
    db.perform("CREATE EXTENSION IF NOT EXISTS h3;")
    db.perform("CREATE EXTENSION IF NOT EXISTS citus;")
    db.perform("CREATE EXTENSION IF NOT EXISTS btree_gist;")
    db.perform("CREATE EXTENSION IF NOT EXISTS pg_trgm;")

    # Create schema
    db.perform("CREATE SCHEMA IF NOT EXISTS basic;")
    db.perform("CREATE SCHEMA IF NOT EXISTS temporal;")

    # Create database functions
    for file in os.listdir("src/db/functions"):
        if file.endswith(".sql"):
            with open(f"src/db/functions/{file}", "r") as f:
                db.perform(f.read())

if __name__ == "__main__":
    create_db()

    db = Database(settings.LOCAL_DATABASE_URI)
    try:
        init_db(db)
        print_info("Database initialized.")
    except Exception as e:
        print_error(e)
        print_error("Database initialization failed.")
    finally:
        db.close()
