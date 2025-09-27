
from src.collection.osm_collection_base import OSMCollection
from src.core.config import settings
from src.db.db import Database


class OSMBuildingCollection(OSMCollection):
    """Collects all POIs from OSM."""
    def __init__(self, db_config, region):
        self.db_config = db_config
        self.dbname = db_config.path[1:]
        self.user = db_config.user
        self.host = db_config.host
        self.port = db_config.port
        self.password = db_config.password
        self.cache = 100000
        super().__init__(self.db_config, dataset_type="building", region=region)

    def building_collection(self, db: Database):
        """Collects all building from OSM"""

        osm_filter = '--keep="building=" --drop-nodes --drop-relations'
        self.download_bulk_osm()
        self.prepare_bulk_osm(osm_filter=osm_filter)
        self.merge_osm_and_import()
        db.perform(f"DROP TABLE IF EXISTS building_osm_{self.region};")
        db.perform(f"ALTER TABLE osm_building_{self.region}_polygon RENAME TO building_osm_{self.region};")

def collect_building(region: str):
    db = Database(settings.LOCAL_DATABASE_URI)
    OSMBuildingCollection(db_config=db.db_config, region=region).building_collection(db=db)
    db.conn.close()


if __name__ == "__main__":
    collect_building()