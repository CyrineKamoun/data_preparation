import pyspark.sql.types as pyspark_types
from pyspark.sql import SparkSession
from pyspark.sql.functions import expr, to_json
from sedona.spark import SedonaContext

from src.config.config import Config
from src.core.config import settings
from src.db.db import Database
from src.utils.utils import (
    get_region_bbox_coords,
    print_error,
    print_info,
    timing,
)


class OvertureCollection:

    def __init__(self, db: Database, db_rd: Database, region: str):
        self.region = region
        self.db = db
        self.db_rd = db_rd
        self.data_config = Config("overture", region)
        self.data_config_collection = self.data_config.collection

    def validate_config(self):
        """Validate Overture data collection configuration."""

        overture_schema = {
            "addresses": ["address"],
            "base": ["infrastructure", "land", "land_cover", "land_use", "water"],
            "buildings": ["building", "building_part"],
            "divisions": ["division", "division_area", "division_boundary"],
            "places": ["place"],
            "transportation": ["connector", "segment"],
        }

        if not self.data_config_collection.get("version"):
            raise ValueError("Overture release version not specified.")

        if not self.data_config_collection.get("theme") or \
            self.data_config_collection["theme"] not in overture_schema.keys():
            raise ValueError(f"Overture theme not specified. Must be one of: {list(overture_schema.keys())}")

        if not self.data_config_collection.get("type") or \
            self.data_config_collection["type"] not in overture_schema[self.data_config_collection["theme"]]:
            raise ValueError(f"Overture type not specified. Must be one of: {overture_schema[self.data_config_collection['theme']]}")

        if not self.data_config_collection.get("local_result_table"):
            raise ValueError("Local result table not specified.")

    def initialize_spark_session(self) -> SparkSession:
        """Initialze SparkSession with required dependencies, AWS credentials provider and resource allocations."""

        config = SedonaContext.builder() \
            .config('spark.jars.packages',
                'org.apache.sedona:sedona-spark-shaded-3.4_2.12:1.5.0,'
                    'org.datasyslab:geotools-wrapper:1.5.0-28.2,'
                    'org.apache.hadoop:hadoop-aws:3.3.4,'
                    'com.amazonaws:aws-java-sdk-bundle:1.12.583,'
                    'org.postgresql:postgresql:42.6.0'
            ) \
            .config(
                "fs.s3a.aws.credentials.provider",
                "org.apache.hadoop.fs.s3a.AnonymousAWSCredentialsProvider"
            ) \
            .config("spark.driver.host", "localhost") \
            .config("spark.executor.memory", "4g") \
            .config("spark.driver.memory", "4g") \
            .config("spark.logConf", "true") \
            .getOrCreate()
        return SedonaContext.create(config)

    def initialize_jdbc_properties(self):
        """Initialize PostgreSQL JDBC connection properties."""

        self.jdbc_url = f"jdbc:postgresql://{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/{settings.POSTGRES_DB}"
        self.jdbc_conn_properties = {
            "user": settings.POSTGRES_USER,
            "password": settings.POSTGRES_PASSWORD,
            "driver": "org.postgresql.Driver",
            "batchsize": "10000"
        }

    def build_overture_s3_uri(self, version, theme, type):
        """Build S3 URI for Overture data source."""

        return f"s3a://overturemaps-us-west-2/release/{version}/theme={theme}/type={type}"

    def initialize_data_source(self, spark: SparkSession):
        """Initialize Overture geoparquet file source and Spark data frames."""

        # Load Overture geoparquet data into Spark DataFrames
        self.data_frame = spark.read.format("geoparquet").load(
            path=self.build_overture_s3_uri(
                version=self.data_config_collection["version"],
                theme=self.data_config_collection["theme"],
                type=self.data_config_collection["type"],
            )
        )

        print_info("Initialized data source.")

    def is_complex_type(self, data_type):
        """Recursively check if the data type or any subfield is a complex type."""

        if isinstance(data_type, pyspark_types.StructType):
            return True
        elif isinstance(data_type, pyspark_types.ArrayType):
            return self.is_complex_type(data_type.elementType)
        elif isinstance(data_type, pyspark_types.MapType):
            return True
        return False

    def process_data_frame(self, bbox_coords: dict):
        """Filter data according to bounds and handle complex data types."""

        # Filter data frame values according to region bounds
        if bbox_coords is not None:
            self.data_frame = self.data_frame.filter(
                (self.data_frame.bbox.xmin > bbox_coords["xmin"]) &
                (self.data_frame.bbox.ymin > bbox_coords["ymin"]) &
                (self.data_frame.bbox.xmax < bbox_coords["xmax"]) &
                (self.data_frame.bbox.ymax < bbox_coords["ymax"])
            )
        self.data_frame = self.data_frame.drop(self.data_frame.bbox)

        # Update data type of complex fields in data frame
        for field in self.data_frame.schema.fields:
            if field.name == "geometry":
                # Convert WKB to WKT
                self.data_frame = self.data_frame.withColumn(
                    field.name,
                    expr("ST_AsText(ST_GeomFromWKB(geometry))"),
                )

            # Convert complex data to JSON
            if self.is_complex_type(field.dataType):
                self.data_frame = self.data_frame.withColumn(
                    field.name,
                    to_json(field.name),
                )

        print_info("Processed data frame.")

    @timing
    def fetch_data(self):
        """Fetch data from Spark dataframe and write to local PostgreSQL database."""

        print_info(f"Downloading data to: {self.data_config_collection['local_result_table']}")
        self.data_frame.write.jdbc(
            url=self.jdbc_url,
            table=self.data_config_collection["local_result_table"],
            mode="overwrite",
            properties=self.jdbc_conn_properties
        )

    @timing
    def alter_result_table(self):
        """Alter local result table to update column data types, create constraints and indexes."""

        print_info(f"Altering table: {self.data_config_collection['local_result_table']}")
        sql_alter_table = f"""
            ALTER TABLE {self.data_config_collection['local_result_table']}
            ALTER COLUMN geometry SET DATA TYPE GEOMETRY(GEOMETRY, 4326);
            ALTER TABLE {self.data_config_collection['local_result_table']} ADD PRIMARY KEY (id);
            CREATE INDEX ON {self.data_config_collection['local_result_table']} USING GIST (geometry);
        """
        self.db.perform(sql_alter_table)

    def run(self):
        """Run Overture data collection process."""

        # Validate configuration
        self.validate_config()

        # Initialize Overture data source
        spark = self.initialize_spark_session()
        self.initialize_jdbc_properties()
        self.initialize_data_source(spark)

        # Process data frame and filter by region bounds
        bbox_coords = get_region_bbox_coords(
            geom_query=self.data_config_collection["region"],
            db=self.db_rd
        )
        self.process_data_frame(bbox_coords)

        # Fetch data to local PostgreSQL database
        self.fetch_data()

        # Update column data types, create constraints and indexes
        self.alter_result_table()


@timing
def collect_overture(region: str):
    print_info(f"Collecting Overture data for region: {region}")
    db = Database(settings.LOCAL_DATABASE_URI)
    db_rd = Database(settings.RAW_DATABASE_URI)

    try:
        OvertureCollection(
            db=db,
            db_rd=db_rd,
            region=region,
        ).run()
        print_info(f"Finished collecting Overture data for region: {region}")
    except Exception as e:
        print_error(f"Failed to collect Overture data for region: {region}")
        raise e
    finally:
        db.close()
        db_rd.close()
