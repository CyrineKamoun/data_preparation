from src.collection.osm_collection_base import OSMCollection
from src.config.config import Config
from src.core.config import settings
from src.db.db import Database
from src.utils.utils import timing


class OSMPOICollection(OSMCollection):
    """Collects all POIs from OSM."""
    def __init__(self, db_config, region):
        self.db_config = db_config
        super().__init__(self.db_config, dataset_type="poi", region=region)

    def poi_collection(self):
        """Collects all POIs from OSM."""
        # Create OSM filter for POIs
        osm_filter = ""
        if self.data_config.collection["osm_tags"]:
            for tag in self.data_config.collection["osm_tags"]:
                if self.data_config.collection["osm_tags"][tag]:
                    for tag_value in self.data_config.collection["osm_tags"][tag]:
                        osm_filter += tag + "=" + tag_value + " "
                else:
                    osm_filter += tag + " "

        if osm_filter:
            osm_filter = '--keep="' + osm_filter + '"'

        # Remove not needed osm feature categories
        if self.data_config.collection["nodes"] == False:
            osm_filter += "--drop-nodes "
        if self.data_config.collection["ways"] == False:
            osm_filter += "--drop-ways "
        if self.data_config.collection["relations"] == False:
            osm_filter += "--drop-relations "

        self.download_bulk_osm()
        self.prepare_bulk_osm(osm_filter=osm_filter)
        self.merge_osm_and_import()

@timing
def collect_poi(region: str):
    """Main function to collect and process POI data."""
    db = Database(settings.LOCAL_DATABASE_URI)

    if region == 'europe':
        for loop_region in Config("poi", region).regions:
            process_poi_collection(db, loop_region)
        process_poi_collection(db, 'europe')
    else:
        process_poi_collection(db, region)

    db.conn.close()

def process_poi_collection(db: Database, region: str):
    """Process POI collection for a given region."""
    osm_poi_collection = OSMPOICollection(db_config=db.db_config, region=region)
    osm_poi_collection.poi_collection()
    osm_poi_collection.export_osm_boundaries_db(db=db)
    #osm_poi_collection.upload_raw_osm_data(boto_client=settings.S3_CLIENT)


if __name__ == "__main__":
    collect_poi()
