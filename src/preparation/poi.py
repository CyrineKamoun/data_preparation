import os

import geopandas as gpd
import numpy as np
import polars as pl

from src.config.config import Config
from src.core.config import settings
from src.db.db import Database
from src.db.tables.poi import POITable
from src.preparation.subscription import Subscription
from src.utils.utils import (
    create_table_dump,
    polars_df_to_postgis,
    print_info,
    restore_table_dump,
    timing,
    vector_check_string_similarity_bulk,
)


class PoiPreparation:
    """Class to prepare the POIs from OpenStreetMap."""

    def __init__(self, db: Database, region: str):
        """Constructor method.

        Args:
            db (Database): Database object
        """
        self.db = db
        self.db_config = self.db.db_config
        self.db_uri = f"postgresql://{self.db_config.user}:{self.db_config.password}@{self.db_config.host}:{self.db_config.port}{self.db_config.path}"
        self.engine = self.db.return_sqlalchemy_engine()
        self.region = region
        self.config_pois = Config("poi", region)

        self.config_pois_preparation = self.config_pois.preparation
        # Extend config with values that are not in the preparation config but in the collection config
        self.extended_pois_preparation = self.extend_config()
        self.config_pois_preparation.update(self.extended_pois_preparation)

    def extend_config(self):
        """Build an extended config with all values that are not in the preparation config but in the collection config."""

        if self.config_pois_preparation is None:
            self.config_pois_preparation = {}

        config_collection = self.config_pois.collection
        # Check if config not in preparation but in collection
        new_config_collection = {}

        osm_tags = config_collection["osm_tags"]

        if not osm_tags: # osm_tags is an empty list which means all osm_tags are collected
            print_info("Extracting tags")
            osm_tags = Config.get_unique_tags(os.path.join(self.config_pois.dataset_dir, 'merged.osm'))

        for osm_tag in osm_tags:
            values = osm_tags[osm_tag]
            for value in values:
                if value not in self.config_pois_preparation:
                    new_config_collection[value] = {
                        "classify_by_tag": {value: {osm_tag: [value]}}
                    }
        print_info("For key-value pairs that are not in the preparation config but in the collection config the POIs are added as preperation by tag.")
        return new_config_collection

    @timing
    def read_poi(self) -> pl.DataFrame:
        """Reads the POIs from the database from the OSM point and OSM polygon table.

        Returns:
            pl.DataFrame: Polars DataFrame with the POIs. The geometry is stored as WKT inside a column called geom.
        """

        # Relevant column names
        column_names = """
        osm_id::bigint, name, brand, "addr:street" AS street, "addr:housenumber" AS housenumber,
        "addr:postcode" AS zipcode, phone, email, website, capacity, opening_hours, wheelchair, operator, origin, organic,
        subway, amenity, shop, tourism, railway, leisure, sport, highway, public_transport, historic, tags::jsonb AS tags
        """

        # Read POIs from database
        sql_query = [
            f"""SELECT {column_names}, 'n' AS osm_type, ST_ASTEXT(way) AS geom FROM public.osm_poi_{self.region}_point""",
            f"""SELECT {column_names}, 'w' AS osm_type, ST_ASTEXT(ST_CENTROID(way)) AS geom FROM public.osm_poi_{self.region}_polygon""",
        ]
        df = pl.read_database_uri(sql_query, self.db_uri)
        return df

    @timing
    def check_by_tag(
        self, df: pl.DataFrame, poi_config: dict, key: str, new_column_names: list[str]
    ) -> list[pl.DataFrame, list[str]]:
        """Checks if a POI has a specific value for a tag.

        Args:
            df (pl.DataFrame): POIs to classify
            poi_config (dict): Configuration for poi category
            key (str): POI category to assign if check is True
            new_column_names (list[str]): List with column names of already classified POIs

        Raises:
            ValueError: Raise error if tag does not exist

        Returns:
            list[pl.DataFrame, list[str]]: Classified POIs and the list with new column names.
        """

        # Get tag and values
        tag = list(poi_config.keys())[0]
        values = poi_config[tag]

        # Check if values are True then only check if tag has value
        if new_column_names == []:
            when_condition = pl.when((pl.col(tag).is_not_null()))
        elif new_column_names != [] and values == [True]:
            conditions = [
                (pl.col(tag).is_not_null()) & ~(pl.col(name) == True)
                for name in new_column_names
            ]

            condition = conditions[0]
            for cond in conditions[1:]:
                condition = condition & cond

            when_condition = pl.when(condition)
        # Check if values are in list of conditions
        elif new_column_names == [] and values != [True]:
            values = [x.lower() for x in values]
            when_condition = pl.when((pl.col(tag).str.to_lowercase().is_in(values)))
        elif new_column_names != [] and values != [True]:
            values = [x.lower() for x in values]
            conditions = [
                (pl.col(tag).str.to_lowercase().is_in(values)) & ~(pl.col(name) == True)
                for name in new_column_names
            ]

            condition = conditions[0]
            for cond in conditions[1:]:
                condition = condition & cond

            when_condition = pl.when(condition)
        else:
            raise ValueError("Conditions for tags cannot be set.")

        # Apply condition
        df = df.with_columns(
            when_condition.then(True).otherwise(False).alias(key + "___tag")
        )

        # Add new column name to list
        new_column_names.append(key + "___tag")
        return df, new_column_names

    @timing
    def check_by_name_in(
        self,
        df: pl.DataFrame,
        poi_config: list[str],
        key: str,
        new_column_names: list[str],
    ) -> list[pl.DataFrame, list[str]]:
        """Checks if a POI has a specific value in the name, brand or website.

        Args:
            df (pl.DataFrame): POIs to classify
            poi_config (list[str]): Configuration for poi category
            key (str): POI category to assign if check is True
            new_column_names (list[str]): List with column names of already classified POIs

        Returns:
            list[pl.DataFrame, list[str]]: Classified POIs and the list with new column names.
        """

        # Make sure that all values are lower case
        values = [x.lower() for x in poi_config]

        # Merge array to text with | as seperator for regex
        values_joined = "|".join(values)

        # Check if values are in list of conditions
        if new_column_names:
            conditions = [pl.col(name) == True for name in new_column_names]
            condition = conditions[0]
            for cond in conditions[1:]:
                condition = condition | cond
        else:
            condition = pl.lit(False)

        when_condition = pl.when(
            ~condition
            & (
                (pl.col("name").str.to_lowercase().is_in(values))
                | (pl.col("brand").str.to_lowercase().is_in(values))
                | (pl.col("website").str.to_lowercase().is_in(values))
                | (pl.col("name").str.to_lowercase().str.contains(values_joined))
                | (pl.col("brand").str.to_lowercase().str.contains(values_joined))
                | (pl.col("website").str.to_lowercase().str.contains(values_joined))
            )
        )

        df = df.with_columns(
            when_condition.then(True).otherwise(False).alias(key + "___name_in")
        )

        # Add new column name to list
        new_column_names.append(key + "___name_in")
        return df, new_column_names

    @timing
    def check_name_similarity(
        self,
        df: pl.DataFrame,
        poi_config: dict,
        threshold: float,
        key: str,
        new_column_names: list[str],
    ) -> list[pl.DataFrame, list[str]]:
        """Checks if a POI has a similar name or brand to a list of POIs.

        Args:
            df (pl.DataFrame): POIs to classify
            poi_config (dict): Configuration for poi category
            threshold (float): Threshold for similarity. Higher is stricter.
            key (str): POI category to assign if check is True
            new_column_names (list[str]): List with column names of already classified POIs

        Returns:
            list[pl.DataFrame, list[str]]: Classified POIs and the list with new column names.
        """

        conditions = [pl.col(name) == True for name in new_column_names]
        condition = conditions[0]
        for cond in conditions[1:]:
            condition = condition | cond

        df_unclassified = df.filter(~condition)
        df_classified = df.filter(condition)

        arr_names = df_unclassified["name"].to_numpy()
        arr_brands = df_unclassified["brand"].to_numpy()

        # Check if name or brand is similar
        check_brands = vector_check_string_similarity_bulk(
            arr_names, poi_config, threshold
        )
        check_names = vector_check_string_similarity_bulk(
            arr_brands, poi_config, threshold
        )

        # Check if name or brand is true
        check_both = np.logical_or(check_brands, check_names)
        df_unclassified = df_unclassified.with_columns(
            pl.Series(name=key + "___name_similarity", values=check_both)
        )

        # Add new column to list
        new_column_names.append(key + "___name_similarity")

        df = pl.concat([df_unclassified, df_classified], how="diagonal")
        return df, new_column_names

    def classify_by_config(self, df: pl.DataFrame, category: str) -> pl.DataFrame:
        """Classifies POIs by config file.

        Args:
            df (pl.DataFrame): POIs to classify.
            category (str): POI category to assign.

        Returns:
            pl.DataFrame: Classified POIs.
        """

        # New columns for classification
        new_column_names = []

        # Classify by tag
        config_by_tag = self.config_pois_preparation[category].get("classify_by_tag")
        if config_by_tag != None:
            for key in config_by_tag:
                df, new_column_names = self.check_by_tag(
                    df=df,
                    poi_config=config_by_tag[key],
                    key=key,
                    new_column_names=new_column_names,
                )

        # Classify by name in list
        config_by_name = self.config_pois_preparation[category].get("classify_by_name")
        if config_by_name != None:
            for key in config_by_name:
                # Merge all values of children
                poi_config = [
                    [k] + v for k, v in config_by_name[key]["children"].items()
                ]
                # Make one dimensional list
                poi_config = list({item for sublist in poi_config for item in sublist})

                # Check if name inside list or wise versa
                df, new_column_names = self.check_by_name_in(
                    df=df,
                    poi_config=poi_config,
                    key=key,
                    new_column_names=new_column_names,
                )

        # Classify by name and brand similarity
        if config_by_name != None:
            for key in config_by_name:
                df, new_column_names = self.check_name_similarity(
                    df=df,
                    poi_config=config_by_name[key]["children"],
                    threshold=config_by_name[key]["threshold"],
                    key=key,
                    new_column_names=new_column_names,
                )

        # Classify categories
        for column_name in new_column_names:
            # Assign category
            value = column_name.split("__")[0]
            df = df.with_columns(
                pl.when(pl.col(column_name) == True)
                .then(pl.lit(value))
                .otherwise(pl.col("category"))
                .alias("category")
            )
            # Drop not need columns
            df = df.drop(column_name)

        unmatched_category = self.config_pois_preparation[category].get("unmatched")
        df = df.with_columns(
            pl.when(pl.col("category") == "str")
            .then(pl.lit(unmatched_category))
            .otherwise(pl.col("category"))
            .alias("category")
        )
        return df

    def classify_public_transport(self, df: pl.DataFrame) -> pl.DataFrame:
        """Classifies public transport POIs.

        Args:
            df (pl.DataFrame): POIs to classify.

        Returns:
            pl.DataFrame: Classified POIs.
        """
        ####################################################################
        # Classify bus stops
        ####################################################################
        df = df.with_columns(
            pl.when(
                (pl.col("highway") == "bus_stop")
                & (pl.col("name").is_not_null())
                & (pl.col("category") == "str")
            )
            .then(pl.lit("bus_stop"))
            .otherwise(pl.col("category"))
            .alias("category")
        )

        df = df.with_columns(
            pl.when(
                (pl.col("public_transport") == "platform")
                & (pl.col("name").is_not_null())
                & (pl.col("tags").str.json_path_match(r"$.bus") == "yes")
                & (pl.col("category") == "str")
            )
            .then(pl.lit("bus_stop"))
            .otherwise(pl.col("category"))
            .alias("category")
        )

        ####################################################################
        # Classify tram stops
        ####################################################################
        df_tram_stops = df.filter(
            (pl.col("public_transport") == "stop_position")
            & (pl.col("name").is_not_null())
            & (pl.col("tags").str.json_path_match(r"$.tram") == "yes")
        )
        pdf_tram_stops = df_tram_stops.to_pandas()
        # Get all tram platforms
        pdf_tram_platforms = df.filter(
            (pl.col("public_transport") == "platform")
            & (pl.col("name").is_not_null())
            & (pl.col("railway") == "tram_stop")
        ).to_pandas()

        # Convert tram stops to GeoDataFrame
        gdf_tram_stops = gpd.GeoDataFrame(
            pdf_tram_stops,
            geometry=gpd.GeoSeries.from_wkt(pdf_tram_stops["geom"], crs="EPSG:4326"),
            crs="EPSG:4326",
        ).to_crs(3857)

        # Convert tram platform to GeoDataFrame
        gdf_tram_platforms = gpd.GeoDataFrame(
            pdf_tram_platforms,
            geometry=gpd.GeoSeries.from_wkt(
                pdf_tram_platforms["geom"], crs="EPSG:4326"
            ),
            crs="EPSG:4326",
        ).to_crs(3857)
        engine = self.db.return_sqlalchemy_engine()
        gdf_tram_stops.to_postgis(name="tram_stops", con=engine, if_exists="replace")

        # Get all tram platforms that are not having a tram stop in 100 meters distance
        gdf_additional_tram_stops = gdf_tram_platforms[
            gdf_tram_platforms.sjoin_nearest(
                gdf_tram_stops[["geometry"]],
                how="left",
                max_distance=100,
                distance_col="distance",
            )["distance"].isna()
        ]
        # Drop geometry column and convert to pandas DataFrame
        pdf_additional_tram_stops = gdf_additional_tram_stops.drop(columns=["geometry"])

        # Convert back to Polars DataFrame, concat with original tram stops and rest of the data
        df_tram_stops = pl.concat(
            [pl.from_pandas(pdf_additional_tram_stops), df_tram_stops], how="diagonal"
        ).with_columns(
            pl.when(pl.col("category") == "str")
            .then(pl.lit("tram_stop"))
            .otherwise(pl.col("category"))
            .alias("category")
        )
        df = pl.concat([df, df_tram_stops], how="diagonal")

        ####################################################################
        # Classify railway stations
        ####################################################################
        df = df.with_columns(
            pl.when(
                (pl.col(["railway"]) == "station")
                & (pl.col("name").is_not_null())
                & (pl.col("tags").str.json_path_match(r"$.disused").is_null())
                & (pl.col("tags").str.json_path_match(r"$.railway:disused").is_null())
                & (
                    (pl.col("tags").str.json_path_match(r"$.usage") != "tourism")
                    | (pl.col("tags").str.json_path_match(r"$.usage").is_null())
                )
            )
            .then(pl.lit("rail_station"))
            .otherwise(pl.col("category"))
            .alias("category")
        )

        df.filter(
            (pl.col(["railway"]) == "station")
            & (pl.col("name").is_not_null())
            & (pl.col("tags").str.json_path_match(r"$.disused").is_null())
            & (pl.col("tags").str.json_path_match(r"$.railway:disused").is_null())
            & (
                (pl.col("tags").str.json_path_match(r"$.usage") != "tourism")
                | (pl.col("tags").str.json_path_match(r"$.usage").is_null())
            )
        )

        df = df.with_columns(
            pl.when(
                (
                    (pl.col("railway").is_in(["stop", "station", "stop_position"]))
                    | (pl.col("public_transport").is_in(["stop", "station", "stop_position"]))
                )
                & ~(pl.col("railway").is_in(["disused_station", "proposed"]))
                & (pl.col("name").is_not_null())
                & (
                    (pl.col("tags").str.json_path_match(r"$.train") == "yes")
                    | (pl.col("tags").str.json_path_match(r"$.rail") == "yes")
                )
                & (pl.col("tags").str.json_path_match(r"$.disused").is_null())
                & (
                    (pl.col("tags").str.json_path_match(r"$.usage") != "tourism")
                    | (pl.col("tags").str.json_path_match(r"$.usage").is_null())
                )
            )
            .then(pl.lit("rail_station"))
            .otherwise(pl.col("category"))
            .alias("category")
        )

        # Classify subway entrances with names
        df = df.with_columns(
            pl.when(
                (pl.col("railway") == "subway_entrance")
                & (pl.col("name").is_not_null())
            )
            .then(pl.lit("subway_entrance"))
            .otherwise(pl.col("category"))
            .alias("category")
        )

        # Classify subway stations without names by assigning the name of the nearest subway station
        # Get subway stations
        pdf_subway_stations = df.filter(
            (pl.col("railway") == "station")
            & (pl.col("subway") == "yes")
            & (pl.col("railway") != "proposed")
        ).to_pandas()

        # Filter subway entrances
        df_subway_entrances = df.filter(
            (pl.col("railway") == "subway_entrance") & (pl.col("category") == "str")
        )

        # Get dataframe without subway entrances
        df = df.join(df_subway_entrances, on="osm_id", how="anti")

        # Convert pandas dataframe to geopandas dataframe
        pdf_subway_entrances = df_subway_entrances.to_pandas()
        gdf_subway_stations = gpd.GeoDataFrame(
            pdf_subway_stations,
            geometry=gpd.GeoSeries.from_wkt(
                pdf_subway_stations["geom"], crs="EPSG:4326"
            ),
            crs="EPSG:4326",
        ).to_crs(3857)

        # Convert pandas dataframe to geopandas dataframe
        gdf_subway_entrances = gpd.GeoDataFrame(
            pdf_subway_entrances,
            geometry=gpd.GeoSeries.from_wkt(
                pdf_subway_entrances["geom"], crs="EPSG:4326"
            ),
            crs="EPSG:4326",
        ).to_crs(3857)

        # Join subway stations to subway entrances and assign the nearest subway station name to the subway entrance
        # Converting to numpy array and then to normal list due to problem with type conversion in polars
        name_to_assign = list(
            gdf_subway_entrances.sjoin_nearest(
                gdf_subway_stations[["geometry", "name"]],
                how="left",
                max_distance=350,
                distance_col="distance",
            )["name_right"].to_numpy()
        )

        # Assigning the name to the subway entrance
        df_subway_entrances = (
            df_subway_entrances.with_columns(
                pl.Series(name="new_name", values=name_to_assign, dtype=pl.Utf8)
            )
            .with_columns(
                pl.when(pl.col("new_name") != "nan")
                .then(pl.col("new_name"))
                .otherwise(pl.col("name"))
                .alias("name")
            )
            .with_columns(
                pl.when((pl.col("name").is_not_null()))
                .then(pl.lit("subway_entrance"))
                .otherwise(pl.col("category"))
                .alias("category")
            )
        )
        df_subway_entrances = df_subway_entrances.drop("new_name")

        # Concatenate the dataframes
        df = pl.concat(
            [df, df_subway_entrances.filter(pl.col("name").is_not_null())],
            how="diagonal",
        )
        return df

    @timing
    def classify_poi(self, df):
        # Create dictionary with empty lists to track classified tags
        classified_tags = {
            k: [] for k in self.config_pois.collection["osm_tags"].keys()
        }

        # Adding category column
        df = df.with_columns(pl.lit("str").alias("category"))

        # Classify Playgrounds
        df = df.with_columns(
            pl.when(
                (pl.col("leisure") == "playground")
                | (pl.col("amenity") == "playground")
            )
            .then(pl.lit("playground"))
            .otherwise(pl.col("category"))
            .alias("category")
        )
        classified_tags["leisure"].append("playground")
        classified_tags["amenity"].append("playground")

        # Classify bikesharing stations
        df = df.with_columns(
            pl.when(
                (pl.col("amenity") == "bicycle_rental")
                & (pl.col("shop").is_null())
                & (pl.col("osm_type") == "n")
            )
            .then(pl.lit("bike_sharing"))
            .otherwise(pl.col("category"))
            .alias("category")
        )
        classified_tags["amenity"].append("bicycle_rental")

        # Classify gyms
        df = df.with_columns(
            pl.when(
                (pl.col("leisure") == "fitness_centre")
                | (
                    (pl.col("leisure") == "sports_centre")
                    & (pl.col("sport") == "fitness")
                )
                & (
                    pl.col("sport").is_in(["multi", "fitness"])
                    | (pl.col("sport").is_null())
                )
                & (pl.col("name").str.to_lowercase().str.contains("yoga") == False)
            )
            .then(pl.lit("gym"))
            .otherwise(pl.col("category"))
            .alias("category")
        )
        classified_tags["leisure"].extend(["fitness_centre", "sports_centre"])
        classified_tags["sport"].extend(["fitness", "multi"])

        # Classify yoga studios
        df = df.with_columns(
            pl.when(
                (
                    (pl.col("sport") == "yoga")
                    | (pl.col("name").str.to_lowercase().str.contains("yoga"))
                )
                & (pl.col("shop").is_null())
            )
            .then(pl.lit("yoga"))
            .otherwise(pl.col("category"))
            .alias("category")
        )
        classified_tags["sport"].append("yoga")

        # Classify public transport
        df = self.classify_public_transport(df=df)
        classified_tags["public_transport"].extend(["stop_position", "station"])
        classified_tags["railway"].extend(
            ["station", "platform", "stop", "tram_stop", "subway_entrance"]
        )
        classified_tags["highway"].append("bus_stop")

        # leisure = water_park need to be categorized before sport = swimming to make sure that water_park is not overwritten by swimming
        df = df.with_columns(
            pl.when(
                (pl.col("leisure") == "water_park")
            )
            .then(pl.lit("water_park"))
            .otherwise(pl.col("category"))
            .alias("category")
        )
        classified_tags["leisure"].append("water_park")

        # exclude swimming pools as we do not want the swimming pools itself, but the whole facility
        df_swimming_pools = df.filter(
            (pl.col("leisure") == "swimming_pool") & (pl.col("category") == "str")
        )

        # Get dataframe without swimming pools
        df = df.join(df_swimming_pools, on="osm_id", how="anti")

        df = df.with_columns(
            pl.when(
                (pl.col("sport") == "swimming")
            )
            .then(pl.lit("swimming"))
            .otherwise(pl.col("category"))
            .alias("category")
        )
        classified_tags["sport"].append("swimming")

        # classify farm shops
        df = df.with_columns(
            pl.when(
                (pl.col("shop") == "farm") | (pl.col("shop") == "honey")
                )
            .then(pl.lit("farm_shop"))
            .otherwise(pl.col("category"))
            .alias("category")
        )
        classified_tags["shop"].extend(["farm", "honey"])

        # classifies ebike charging stations and charging stations
        df = df.with_columns(
            pl.when(
                (pl.col("amenity") == "charging_station")
            )
            .then(
                pl.when(
                    pl.col("tags").str.contains('bicycle":"yes"')
                )
                .then(pl.lit("ebike_charging_station"))
                .otherwise(pl.lit("charging_station"))
            )
            .otherwise(pl.col("category"))
            .alias("category")
        )
        classified_tags["amenity"].append("charging_station")

        # classifies food vending machines
        df = df.with_columns(
            pl.when(
                (pl.col("amenity") == "vending_machine")
                & (pl.col("tags").map_elements(
                    lambda tags: any(tag in tags for tag in ['food', 'bread', 'milk', 'eggs', 'meat', 'potato', 'honey', 'cheese']),
                    return_dtype=pl.Boolean
                ))
            )
            .then(pl.lit("food_vending_machine"))
            .otherwise(pl.col("category"))
            .alias("category")
        )
        classified_tags["amenity"].append("vending_machine")

        # classifies religious sites
        df = df.with_columns(
            pl.when(
                ((pl.col("amenity") == "monastery") | (pl.col("amenity") == "place_of_worship"))
                & (pl.col("tags").str.contains('wikidata'))
            )
            .then(pl.lit("religious_site"))
            .otherwise(pl.col("category"))
            .alias("category")
        )
        classified_tags["amenity"].extend(["monastery", "place_of_worship"])

        # Loop through config
        for key in self.config_pois_preparation:
            if key not in [item for sublist in classified_tags.values() for item in sublist]:
                df_classified_config = pl.DataFrame()
                # Get OSM tag
                for tag, values in self.config_pois.collection["osm_tags"].items():
                    if key in values:
                        osm_tag = tag
                        break

                # Check if config should be inherited
                if list(self.config_pois_preparation[key].keys())[0] == "inherit":
                    category = self.config_pois_preparation[key]["inherit"]
                else:
                    category = key

                # If filter returns nothing continue
                if df.filter(pl.col(osm_tag) == key).height == 0:
                    continue
                else:
                    df_classified_config = self.classify_by_config(
                        df=df.filter(pl.col(osm_tag) == key), category=category
                    )

                # Remove rows classified by config
                df = df.filter(~(pl.col(osm_tag) == key) | (pl.col(osm_tag).is_null()))
                df = pl.concat([df_classified_config, df], how="diagonal")
                classified_tags[osm_tag].append(key)

        # Remove all categories from config that were already classified
        cleaned_config_poi = self.config_pois.collection["osm_tags"]
        for key in cleaned_config_poi:
            cleaned_config_poi[key] = list(
                set(cleaned_config_poi[key]) - set(classified_tags[key])
            )

        # # Assign remaining categories
        # for key in cleaned_config_poi:
        #     for value in cleaned_config_poi[key]:
        #         df = df.with_columns(
        #             pl.when((pl.col(key) == value))
        #             .then(value)
        #             .otherwise(pl.col("category"))
        #             .alias("category")
        #         )

        return df


def prepare_poi(region: str):
    """Prepare POI data for the region.

    Args:
        region (str): Region to prepare POI data for.
    """

    db = Database(settings.LOCAL_DATABASE_URI)

    if region == 'europe':

        create_table_sql = POITable(data_set_type='poi', schema_name = 'poi', data_set_name =f'osm_{region}').create_poi_table(table_type='standard')
        db.perform(create_table_sql)

        for loop_region in Config("poi", region).regions:
            process_poi_preparation(db, loop_region)

            # Insert data from regional table into 'europe' table
            insert_poi_osm_sql = f"""
                INSERT INTO poi.poi_osm_europe(category, name, operator, street, housenumber, zipcode, phone, email, website, capacity, opening_hours, wheelchair, source, tags, geom)
                SELECT
                    category,
                    name,
                    operator,
                    street,
                    housenumber,
                    zipcode,
                    phone,
                    email,
                    website,
                    capacity,
                    opening_hours,
                    wheelchair,
                    source,
                    tags,
                    geom
                FROM public.poi_osm_{loop_region}
            """
            db.perform(insert_poi_osm_sql)
    else:
        process_poi_preparation(db, region)

    print_info(f'Preparation of region {region} is finished.')

    db.conn.close()

def process_poi_preparation(db: Database, region: str):
    """Process POI preparation for a given region."""
    poi_preparation = PoiPreparation(db=db, region=region)

    # Read and classify POI data
    df = poi_preparation.read_poi()
    df = poi_preparation.classify_poi(df)

    # Export raw data to local PostGIS
    engine = db.return_sqlalchemy_engine()
    polars_df_to_postgis(
        engine=engine,
        df=df.filter(pl.col("category") != "str"),
        table_name=f"poi_osm_{region}_raw",
        schema="public",
        if_exists="replace",
        geom_column="geom",
        srid=4326,
        create_geom_index=False,
        jsonb_column="tags",
    )

    # create poi schema
    db.perform("""CREATE SCHEMA IF NOT EXISTS poi;""")

    # insert into our POI schema
    create_table_sql = POITable(data_set_type='poi', schema_name = 'public', data_set_name =f'osm_{region}').create_poi_table(table_type='standard')
    db.perform(create_table_sql)

    insert_poi_osm_sql = f"""
        INSERT INTO public.poi_osm_{region}(category, name, operator, street, housenumber, zipcode, phone, email, website, capacity, opening_hours, wheelchair, source, tags, geom)
        SELECT
            category,
            TRIM(name),
            operator,
            street,
            housenumber,
            zipcode,
            phone,
            CASE WHEN octet_length(email) BETWEEN 6 AND 320 AND email LIKE '_%@_%.__%' THEN email ELSE NULL END AS email,
            CASE WHEN website ~* '^[a-z](?:[-a-z0-9\+\.])*:(?:\/\/(?:(?:%[0-9a-f][0-9a-f]|[-a-z0-9\._~!\$&''\(\)\*\+,;=:@])|[\/\?])*)?' :: TEXT THEN website ELSE NULL END AS website,
            CASE WHEN capacity ~ '^[0-9\.]+$' THEN try_cast_to_int(capacity) ELSE NULL END AS capacity,
            opening_hours,
            CASE WHEN wheelchair IN ('yes', 'no', 'limited') THEN wheelchair ELSE NULL END AS wheelchair,
            'OSM' AS source,
            (jsonb_strip_nulls(
                (jsonb_build_object(
                    'origin', origin, 'organic', organic, 'subway', subway, 'amenity', amenity,
                    'shop', shop, 'tourism', tourism, 'railway', railway, 'leisure', leisure, 'sport', sport, 'highway',
                    highway, 'public_transport', public_transport, 'historic', historic, 'brand', brand
                ) || tags) || jsonb_build_object('extended_source', jsonb_build_object('osm_id', osm_id, 'osm_type', osm_type))
            )) AS tags,
            geom
        FROM public.poi_osm_{region}_raw
    """
    db.perform(insert_poi_osm_sql)

def export_poi(region: str):
    """Export POI data to remote database

    Args:
        region (str): Region to export
    """
    db = Database(settings.LOCAL_DATABASE_URI)
    db_rd = Database(settings.RAW_DATABASE_URI)

    subscription = Subscription(db=db, db_rd = db_rd, region=region)
    subscription.subscribe_poi()

    db.conn.close()
    db_rd.conn.close()


if __name__ == "__main__":
    export_poi()
