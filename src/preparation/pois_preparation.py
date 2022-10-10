import time
import json
from difflib import SequenceMatcher
import numpy as np
import pandas as pd
import geopandas as gp
from src.db.config import DATABASE, DATABASE_RD
from src.db.db import Database

# from pandas.core.accessor import PandasDelegate
from src.config.config import Config
from src.other.utility_functions import gdf_conversion, table_dump
from src.other.utils import return_tables_as_gdf
from src.collection.osm_collection import OsmCollection
from sqlalchemy.sql import text
from sqlalchemy.dialects.postgresql import JSONB

from src.other.utils import create_table_schema, create_table_dump


# ================================== POIs preparation =============================================#
#!!!!!!!!!!!!!!! This codebase needs to be rewritten !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#
def osm_poi_classification(df: gp.GeoDataFrame, config: dict):
    def similar(a, b):
        return SequenceMatcher(None, a, b).ratio()

    # Function search in config
    def poi_return_search_condition(name, var_dict):
        # Func asses probability of string similarity

        for key, value in var_dict.items():
            for v in value:
                if (similar(name, v) > 0.8 or (name in v or v in name)) and name != "":
                    return key
                else:
                    pass

    # Timer start
    print("Preparation started...")
    start_time = time.time()

    df["id"] = df["osm_id"]
    df = df.rename(columns={"way": "geom"})
    df = df.rename(
        columns={"addr:housenumber": "housenumber"}
    )  # , "osm_type" : "origin_geometry"
    df = df.assign(source="osm")

    # Replace None values with empty strings in "name" column and dict in "tags" column
    # To be able to search within the values
    df["name"] = df["name"].fillna(value="")
    df["amenity"] = df["amenity"].fillna(value="")

    # variables for preparation
    # !!! Some columns could be not in the list
    # REVISE it (probabaly check columns - if value from config is not there - create column)

    i_amenity = df.columns.get_loc("amenity")
    i_tourism = df.columns.get_loc("tourism")
    i_shop = df.columns.get_loc("shop")
    i_name = df.columns.get_loc("name")
    i_leisure = df.columns.get_loc("leisure")
    i_sport = df.columns.get_loc("sport")
    i_organic = df.columns.get_loc("organic")
    i_operator = df.columns.get_loc("operator")
    i_highway = df.columns.get_loc("highway")
    i_public_transport = df.columns.get_loc("public_transport")
    i_railway = df.columns.get_loc("railway")
    i_tags = df.columns.get_loc("tags")

    # Depending on zone "origin" can be not presented
    try:
        i_origin = df.columns.get_loc("origin")
    except:
        df = df.assign(origin=None)
        i_origin = df.columns.get_loc("origin")

    # Try to get location of subway column if it exists
    try:
        i_subway = df.columns.get_loc("subway")
    except:
        df = df.assign(subway=None)
        i_subway = df.columns.get_loc("subway")

    # This section getting var from conf class (variables container)
    var = config.preparation
    # Related to sport facilities
    sport_var_disc = var["sport"]["sport_var_disc"]
    leisure_var_add = var["sport"]["leisure_var_add"]
    leisure_var_disc = var["sport"]["leisure_var_disc"]
    # Related to Supermarkets
    health_food_var = var["health_food"]
    hypermarket_var = var["hypermarket"]
    no_end_consumer_store_var = var["no_end_consumer_store"]
    discount_supermarket_var = var["discount_supermarket"]
    supermarket_var = var["supermarket"]
    chemist_var = var["chemist"]
    organic_var = var["organic"]
    # Banks
    bank_var = var["bank"]
    # Related to Discount Gyms
    discount_gym_var = var["discount_gym"]

    # Convert polygons to points and set origin geometry for all elements
    df.loc[df["geom"].geom_type == "MultiPolygon", "geom"] = df["geom"].centroid
    df.loc[df["geom"].geom_type == "Polygon", "geom"] = df["geom"].centroid
    df.loc[df["geom"].geom_type == "Point", "origin_geometry"] = "point"
    df.loc[df["geom"].geom_type == "MultiPolygon", "origin_geometry"] = "polygon"
    df.loc[df["geom"].geom_type == "Polygon", "origin_geometry"] = "polygon"

    df = df.reset_index(drop=True)

    # # Playgrounds
    df["amenity"] = np.where(
        (df["leisure"] == "playground")
        & (df["leisure"] != df["amenity"])
        & (df["amenity"]),
        df["leisure"],
        df["amenity"],
    )
    df["amenity"] = np.where(
        (df["leisure"] == "playground") & (df["amenity"] == ""),
        df["leisure"],
        df["amenity"],
    )

    # drop operator value for supermarkets
    df.loc[df["shop"] == "supermarket", "operator"] = ""

    # bycicle rental & doctors renaming
    df.loc[df.amenity == "bicycle_rental", "amenity"] = "bike_sharing"
    df.loc[df.amenity == "doctors", "amenity"] = "general_practitioner"

    # Iterate through the rows
    for i in df.index:
        df_row = df.iloc[i]

        if (
            df_row[i_tourism]
            and df_row[i_amenity] != ""
            and df_row[i_tourism] != df_row[i_amenity]
            and df_row[i_tourism] != "yes"
        ):
            df_row["amenity"] = df_row["tourism"]
            df = df.append(df_row)
        elif (
            df_row[i_tourism] and df_row[i_amenity] == "" and df_row[i_tourism] != "yes"
        ):
            df.iat[i, i_amenity] = df.iat[i, i_tourism]

        # Sport pois from leisure and sport features
        if (
            df_row[i_sport]
            or df_row[i_leisure] in leisure_var_add
            and df_row[i_leisure] not in leisure_var_disc
            and df_row[i_sport] not in sport_var_disc
        ):
            df.iat[i, i_amenity] = "sport"
            if df_row[i_sport]:
                df.iat[i, i_tags]["sport"] = df_row[i_sport]
            elif df_row[i_leisure]:
                df.iat[i, i_tags]["leisure"] = df_row[i_leisure]
        elif df_row[i_leisure] not in leisure_var_add and df_row[i_leisure]:
            df.iat[i, i_tags]["leisure"] = df_row[i_leisure]

        # Gyms and discount gyms -> Fitness centers
        if (
            (
                df_row[i_leisure] == "fitness_centre"
                or (
                    df_row[i_leisure] == "sport_centre" and df_row[i_sport] == "fitness"
                )
            )
            and (df_row[i_sport] in ["multi", "fitness"] or not df_row[i_sport])
            and "yoga" not in df_row[i_name].lower()
        ):
            operator = poi_return_search_condition(
                df_row[i_name].lower(), discount_gym_var
            )
            if operator:
                df.iat[i, i_operator] = operator
                df.iat[i, i_amenity] = "discount_gym"
            else:
                df.iat[i, i_amenity] = "gym"
            continue

        # Yoga centers check None change here
        if (
            df_row[i_sport] == "yoga"
            or "yoga" in df_row[i_name]
            or "Yoga" in df_row[i_name]
        ) and not df_row[i_shop]:
            df.iat[i, i_amenity] = "yoga"
            continue

        # Recclasify shops. Define convenience and clothes, others assign to amenity. If not rewrite amenity with shop value
        if df_row[i_shop] == "grocery" and df_row[i_amenity] == "":
            if df_row[i_organic] == "only":
                df.iat[i, i_amenity] = "organic_supermarket"
                df.iat[i, i_tags]["organic"] = df_row[i_organic]
                operator = poi_return_search_condition(
                    df_row[i_name].lower(), organic_var
                )
                if operator:
                    df.iat[i, i_operator] = operator
                continue
            elif df_row[i_origin]:
                df.iat[i, i_amenity] = "international_hypermarket"
                df.iat[i, i_tags]["origin"] = df_row[i_origin]
                continue
            else:
                df.iat[i, i_amenity] = "convenience"
                df.iat[i, i_shop] = None
                continue

        elif df_row[i_shop] == "fashion" and df_row[i_amenity] == "":
            df.iat[i, i_amenity] = "clothes"
            df.iat[i, i_shop] = None
            continue

        # Supermarkets recclassification
        elif df_row[i_shop] == "supermarket" and df_row[i_amenity] == "":
            operator = [
                poi_return_search_condition(df_row[i_name].lower(), health_food_var),
                poi_return_search_condition(df_row[i_name].lower(), hypermarket_var),
                poi_return_search_condition(
                    df_row[i_name].lower(), no_end_consumer_store_var
                ),
                poi_return_search_condition(
                    df_row[i_name].lower(), discount_supermarket_var
                ),
                poi_return_search_condition(df_row[i_name].lower(), supermarket_var),
            ]
            if any(operator):
                for op in operator:
                    if op:
                        df.iat[i, i_operator] = op
                        o_ind = operator.index(op)
                        df.iat[i, i_amenity] = [
                            cat
                            for i, cat in enumerate(
                                [
                                    "health_food",
                                    "hypermarket",
                                    "no_end_consumer_store",
                                    "discount_supermarket",
                                    "supermarket",
                                ]
                            )
                            if i == o_ind
                        ][0]
                        continue
                    else:
                        pass
            else:
                if df_row[i_organic] == "only":
                    df.iat[i, i_amenity] = "organic_supermarket"
                    df.iat[i, i_tags]["organic"] = df_row[i_organic]
                    operator = poi_return_search_condition(
                        df_row[i_name].lower(), organic_var
                    )
                    if operator:
                        df.iat[i, i_operator] = operator
                    continue
                elif df_row[i_origin]:
                    df.iat[i, i_amenity] = "international_hypermarket"
                    df.iat[i, i_tags]["origin"] = df_row[i_origin]
                    continue
                # rewrite next block - move search condition to config, write function for search
                elif (
                    "müller " in df_row[i_name].lower()
                    or df_row[i_name].lower() == "müller"
                ):
                    df.iat[i, i_amenity] = "chemist"
                    df.iat[i, i_operator] = "müller"
                    continue
                elif "dm " in df_row[i_name].lower() or "dm-" in df_row[i_name].lower():
                    df.iat[i, i_amenity] = "chemist"
                    df.iat[i, i_operator] = "dm"
                    continue
                else:
                    df.iat[i, i_amenity] = "supermarket"
                    continue
        elif df_row[i_shop] == "chemist" and df_row[i_amenity] == "":
            operator = poi_return_search_condition(df_row[i_name].lower(), chemist_var)
            if operator:
                df.iat[i, i_operator] = operator
                df.iat[i, i_amenity] = "chemist"
                continue
            else:
                df.iat[i, i_amenity] = "chemist"
                continue
        elif df_row[i_shop] == "health_food" and df_row[i_amenity] == "":
            operator = poi_return_search_condition(
                df_row[i_name].lower(), health_food_var
            )
            if operator:
                df.iat[i, i_operator] = operator
                df.iat[i, i_amenity] = "health_food"
                continue
            else:
                df.iat[i, i_amenity] = "health_food"
                continue
        elif df_row[i_shop] and df_row[i_shop] != "yes" and df_row[i_amenity] == "":
            df.iat[i, i_amenity] = df.iat[i, i_shop]
            df.iat[i, i_tags]["shop"] = df_row[i_shop]
            continue

        # Banks
        if df_row[i_amenity] == "bank":
            operator = poi_return_search_condition(df_row[i_name].lower(), bank_var)
            if operator:
                df.iat[i, i_operator] = operator
                continue

        # Transport stops
        if df_row[i_highway] == "bus_stop" and df_row[i_name] != "":
            df.iat[i, i_amenity] = "bus_stop"
            continue
        elif (
            df_row[i_public_transport] == "platform"
            and df_row[i_tags]
            and df_row[i_highway] != "bus_stop"
            and df_row[i_name] != ""
            and ("bus", "yes") in df_row[i_tags].items()
        ):
            df.iat[i, i_amenity] = "bus_stop"
            df.iat[i, i_tags]["public_transport"] = df_row[i_public_transport]
            continue
        elif (
            df_row[i_public_transport] == "stop_position"
            and isinstance(df_row[i_tags], dict)
            and ("tram", "yes") in df_row[i_tags].items()
            and df_row[i_name] != ""
        ):
            df.iat[i, i_amenity] = "tram_stop"
            df.iat[i, i_tags]["public_transport"] = df_row[i_public_transport]
            continue
        elif df_row[i_railway] == "subway_entrance":
            df.iat[i, i_amenity] = "subway_entrance"
            df.iat[i, i_tags]["railway"] = df_row[i_railway]
            continue
        elif (
            df_row[i_railway] == "stop"
            and df_row[i_tags]
            and ("train", "yes") in df_row[i_tags].items()
        ):
            df.iat[i, i_amenity] = "rail_station"
            df.iat[i, i_tags]["railway"] = df_row[i_railway]
            continue
        elif df_row[i_highway]:
            df.iat[i, i_tags]["highway"] = df_row[i_highway]
        elif df_row[i_public_transport]:
            df.iat[i, i_tags]["public_transport"] = df_row[i_public_transport]
        elif df_row[i_railway]:
            df.iat[i, i_tags]["railway"] = df_row[i_railway]
        elif df_row[i_subway]:
            df.iat[i, i_tags]["subway"] = df_row[i_subway]

    df = df.reset_index(drop=True)

    # # # Convert DataFrame back to GeoDataFrame (important for saving geojson)
    df = gp.GeoDataFrame(df, geometry="geom")
    df.crs = "EPSG:4326"

    # Filter subway entrances
    try:
        df_sub_stations = df[
            (df["public_transport"] == "station")
            & (df["subway"] == "yes")
            & (df["railway"] != "proposed")
        ]
        df_sub_stations = df_sub_stations[["name", "geom", "id"]]
        df_sub_stations = df_sub_stations.to_crs(31468)
        df_sub_stations["geom"] = df_sub_stations["geom"].buffer(250)
        df_sub_stations = df_sub_stations.to_crs(4326)

        df_sub_entrance = df[(df["amenity"] == "subway_entrance")]
        df_sub_entrance = df_sub_entrance[["name", "geom", "id"]]

        df_snames = gp.overlay(df_sub_entrance, df_sub_stations, how="intersection")
        df_snames = df_snames[["name_2", "id_1"]]
        df = (
            df_snames.set_index("id_1")
            .rename(columns={"name_2": "name"})
            .combine_first(df.set_index("id"))
        )
    except:
        print("No subway stations for given area.")
        df = df.drop(columns={"id"})

    # Remove irrelevant columns an rows with not defined amenity
    df = df.drop(
        columns={
            "shop",
            "tourism",
            "leisure",
            "sport",
            "highway",
            "origin",
            "organic",
            "public_transport",
            "railway",
            "subway",
        }
    )
    df = df.drop_duplicates(subset=["osm_id", "amenity", "name"], keep="first")
    df = df.drop(df[df.amenity == ""].index)

    # Timer finish
    print("Preparation took %s seconds ---" % (time.time() - start_time))

    return gp.GeoDataFrame(df, geometry="geom")


# Preparation 'jedeschule' table ->> conversion to fusable format
def school_categorization(df, config, result_name, return_type):
    var = config.preparation
    var_schools = var["schools"]

    schule = var_schools["schule"]
    grundschule = var_schools["grundschule"]
    hauptschule_mittelschule = var_schools["hauptschule_mittelschule"]
    exclude = var_schools["exclude"]

    df["name_1"] = df["name"].str.lower()
    df["name_1"] = df["name"].replace({np.nan: ""})

    for ex in exclude:
        temp_df = df.loc[df["name_1"].str.contains(ex, case=False)]
        m = ~df.id.isin(temp_df.id)
        df = df[m]
    df = df.drop(columns={"name_1"})
    df = df.reset_index()

    df.loc[df["school_t_1"].isin(schule), "amenity"] = df["school_t_1"].str.lower()
    df_base = df[~df["amenity"].isnull()]

    df.loc[df["school_t_1"].isin(grundschule), "amenity"] = "grundschule"
    df_grund = df[df["school_t_1"].isin(grundschule)]

    df.loc[
        df["school_t_1"].isin(hauptschule_mittelschule), "amenity"
    ] = "hauptschule_mittelschule"
    df_hauptmittel = df[df["school_t_1"].isin(hauptschule_mittelschule)]

    df_result = pd.concat([df_base, df_grund, df_hauptmittel], sort=False)

    # Should return 2 dataframes grundschule and mittel_hauptschule
    return gdf_conversion(df_result, result_name, return_type)


# function deaggregates childacare amenities to four groups according to value in "age_group" column
def kindergarten_deaggrgation(df, result_name, return_type):
    df.loc[df["age_group"] == "0-3", "amenity"] = "nursery"
    df.loc[
        (df["age_group"] == "3-6") | (df["age_group"] == "2-6"), "amenity"
    ] = "kindergarten"
    df.loc[df["age_group"] == "6+", "amenity"] = "kinderhort"

    df_temp = df[(df["age_group"] == "0-6") | (df["age_group"] == "1-6")]
    df_temp["amenity"] = "nursery"

    df.loc[
        (df["age_group"] == "0-6") | (df["age_group"] == "1-6"), "amenity"
    ] = "kindergarten"

    df_result = pd.concat([df, df_temp], sort=False).reset_index(drop=True)

    return gdf_conversion(df_result, result_name, return_type)


# ================================ Landuse preparation ============================================#


def landuse_preparation(dataframe, config=None, filename=None, return_type=None):
    """introduces the landuse_simplified column and classifies it according to the config input"""

    df = dataframe

    if not config:
        config = Config("landuse")

    # Timer start
    print("Preparation started...")
    start_time = time.time()

    df = df.rename(columns={"id": "osm_id"})

    # Preprocessing: removing, renaming and reordering of columns
    # df = df.drop(columns={"timestamp", "version", "changeset"})
    if "geometry" in df.columns:
        df = df.rename(columns={"geometry": "geom"})
    if "way" in df.columns:
        df = df.rename(columns={"way": "geom"})

    # classify by geometry
    df.at[df["geom"].geom_type == "Point", "origin_geometry"] = "point"
    df.at[df["geom"].geom_type == "MultiPolygon", "origin_geometry"] = "polygon"
    df.at[df["geom"].geom_type == "Polygon", "origin_geometry"] = "polygon"
    df.at[df["geom"].geom_type == "LineString", "origin_geometry"] = "line"

    # remove lines and points from dataset
    df = df[df.origin_geometry != "line"]
    df = df.reset_index(drop=True)
    df = df[df.origin_geometry != "point"]
    df = df.reset_index(drop=True)

    df["landuse_simplified"] = None
    df = df[
        [
            "landuse_simplified",
            "landuse",
            "tourism",
            "amenity",
            "leisure",
            "natural",
            "name",
            "tags",
            "osm_id",
            "origin_geometry",
            "geom",
        ]
    ]

    df = df.assign(source="osm")

    # Fill landuse_simplified coulmn with values from the other columns
    custom_filter = config.collection["osm_tags"]

    if custom_filter is None:
        print(
            "landuse_simplified can only be generated if the custom_filter of collection\
               is passed"
        )
    else:
        for i in custom_filter.keys():
            df["landuse_simplified"] = df["landuse_simplified"].fillna(
                df[i].loc[df[i].isin(custom_filter[i])]
            )

        # import landuse_simplified dict from config
        landuse_simplified_dict = config.preparation["landuse_simplified"]

        # Rename landuse_simplified by grouping
        # e.g. ["basin","reservoir","salt_pond","waters"] -> "water"
        for i in landuse_simplified_dict.keys():
            df["landuse_simplified"] = df["landuse_simplified"].replace(
                landuse_simplified_dict[i], i
            )

    if df.loc[
        ~df["landuse_simplified"].isin(list(landuse_simplified_dict.keys()))
    ].empty:
        print("All entries were classified in landuse_simplified")
    else:
        print(
            "The following tags in the landuse_simplified column need to be added to the\
               landuse_simplified dict in config.yaml:"
        )
        print(
            df.loc[~df["landuse_simplified"].isin(list(landuse_simplified_dict.keys()))]
        )

    # remove lines from dataset
    df = df[df.origin_geometry != "line"]
    df = df.reset_index(drop=True)

    # Convert DataFrame back to GeoDataFrame (important for saving geojson)
    df = gp.GeoDataFrame(df, geometry="geom")
    df.crs = "EPSG:4326"
    df = df.reset_index(drop=True)

    # Timer finish
    print(f"Preparation took {time.time() - start_time} seconds ---")

    return gdf_conversion(df, filename, return_type)


# ================================ Buildings preparation ======================================#


def buildings_preparation(dataframe, config=None, filename=None, return_type=None):
    """introduces the landuse_simplified column and classifies it according to the config input"""
    if not config:
        config = Config("buildings")

    df = dataframe

    config_pop = Config("population")

    # Timer start
    print("Preparation started...")
    start_time = time.time()
    # Preprocessing: removing, renaming, reordering and data type adjustments of columns

    if "geometry" in df.columns:
        df = df.rename(columns={"geometry": "geom"})
    if "way" in df.columns:
        df = df.rename(columns={"way": "geom"})

    df = df.rename(
        columns={
            "addr:street": "street",
            "addr:housenumber": "housenumber",
            "building:levels": "building_levels",
            "roof:levels": "roof_levels",
        }
    )
    df["residential_status"] = None
    df["area"] = None

    # classify by geometry
    df.at[df["geom"].geom_type == "Point", "origin_geometry"] = "point"
    df.at[df["geom"].geom_type == "MultiPolygon", "origin_geometry"] = "polygon"
    df.at[df["geom"].geom_type == "Polygon", "origin_geometry"] = "polygon"
    df.at[df["geom"].geom_type == "LineString", "origin_geometry"] = "line"

    # remove lines and points from dataset
    df = df[df.origin_geometry != "line"]
    df = df.reset_index(drop=True)
    df = df[df.origin_geometry != "point"]
    df = df.reset_index(drop=True)

    df = df[
        [
            "osm_id",
            "building",
            "amenity",
            "leisure",
            "residential_status",
            "street",
            "housenumber",
            "area",
            "building_levels",
            "roof_levels",
            "origin_geometry",
            "geom",
        ]
    ]
    df["building_levels"] = pd.to_numeric(
        df["building_levels"], errors="coerce", downcast="float"
    )
    df["roof_levels"] = pd.to_numeric(
        df["roof_levels"], errors="coerce", downcast="float"
    )
    df = df.assign(source="osm")

    # classifying residential_status in 'with_residents', 'potential_residents', 'no_residents'
    df.loc[
        (
            (df.building.str.contains("yes"))
            & (df.amenity.isnull())
            & (df.amenity.isnull())
        ),
        "residential_status",
    ] = "potential_residents"
    df.loc[
        df.building.isin(config_pop.preparation["building_types_residential"]),
        "residential_status",
    ] = "with_residents"
    df.residential_status.fillna("no_residents", inplace=True)

    # Convert DataFrame back to GeoDataFrame (important for saving geojson)
    df = gp.GeoDataFrame(df, geometry="geom")
    df.crs = "EPSG:4326"
    df = df.reset_index(drop=True)

    # calculating the areas of the building outlines in m^2
    df = df.to_crs({"init": "epsg:3857"})
    df["area"] = df["geom"].area.round(2)
    df = df[df.area != 0]
    df = df.to_crs({"init": "epsg:4326"})

    # Timer finish
    print(f"Preparation took {time.time() - start_time} seconds ---")

    return gdf_conversion(df, filename, return_type)


class PoisPreparation:
    """Class to preprare the POIs."""

    def __init__(self, db_config: dict):
        self.root_dir = "/app"
        self.dbname, self.host, self.username, self.port = (
            db_config["dbname"],
            db_config["host"],
            db_config["user"],
            db_config["port"],
        )
        self.db_config = db_config
        self.db = Database(self.db_config)
        self.sqlalchemy_engine = self.db.return_sqlalchemy_engine()
        self.config_pois = Config("pois")
        self.config_pois_preparation = self.config_pois.preparation

    def perform_pois_preparation(self, db_reading):
        """_summary_

        Args:
            db_reading (Database): Database object to read the custom POI data.

        Returns:
            GeoDataFrame: the prepared POIs
        """

        osm_collection = OsmCollection(self.db_config)
        osm_collection.pois_collection()

        created_tables = ["osm_pois_point", "osm_pois_polygon"]
        for table in created_tables:
            self.db.perform(
                f"ALTER TABLE {table} ALTER COLUMN tags TYPE jsonb USING tags::jsonb;"
            )

        poi_gdf = return_tables_as_gdf(self.sqlalchemy_engine, created_tables)
        poi_gdf = osm_poi_classification(poi_gdf, self.config_pois)
        create_table_schema(self.db, self.db_config, "basic.poi")

        poi_gdf = poi_gdf.reset_index(drop=True)
        copy_gdf = poi_gdf.copy()
        keys_for_tags = [
            "phone",
            "website",
            "operator",
            "source",
            "brand",
            "addr:city",
            "addr:country",
            "origin_geometry",
            "osm_id",
        ]

        poi_gdf.rename(
            columns={
                "amenity": "category",
                "addr:street": "street",
                "addr:postcode": "zipcode",
            },
            inplace=True,
        )
        poi_gdf.drop(
            columns=keys_for_tags,
            inplace=True,
        )
        # Replace category with the custom data
        custom_table = self.config_pois.config["pois"]["replace"]["table_name"]
        if custom_table is not None:

            categories = db_reading.select(
                f"""
                SELECT DISTINCT category 
                FROM {custom_table}
                """
            )
            categories = [i[0] for i in categories]
            poi_gdf = self.replace_osm_pois_by_custom(
                poi_gdf, db_reading, custom_table, categories
            )

        # Create preliminary GOAT UID
        poi_gdf["uid"] = (
            (poi_gdf.centroid.x * 1000).apply(np.floor).astype(int).astype(str)
            + "-"
            + (poi_gdf.centroid.y * 1000).apply(np.floor).astype(int).astype(str)
            + "-"
            + poi_gdf.category
        )
        poi_gdf["uid"] = (
            poi_gdf["uid"]
            + "-"
            + (
                ((poi_gdf.groupby("uid").cumcount() + 1) / 1000).astype(str)
            ).str.replace(".", "", regex=False)
        )
        
        # Match schema and put remaining attributes in tags
        loc_tags = poi_gdf.columns.get_loc("tags")
        for i in range(len(poi_gdf.index)):
            row = copy_gdf.iloc[i]

            new_tags = row["tags"]
            for key in keys_for_tags:
                if row[key] is not None:
                    new_tags[key] = str(row[key])

            poi_gdf.iat[i, loc_tags] = json.dumps(new_tags)

        
        # Upload to PostGIS
        poi_gdf.to_postgis(
            name="poi",
            con=self.sqlalchemy_engine,
            schema="basic",
            if_exists="append",
            index=False,
            dtype={"tags": JSONB},
        )
        return poi_gdf

    def replace_osm_pois_by_custom(
        self, poi_gdf, db_reading, custom_table: str, categories: list[str]
    ):
        """Replaces the OSM POIs by the custom POIs.

        Args:
            gdf (GeoDataFrame): The prepared OSM POIs
            db_reading (Database): The database to read the custom POIs from
            custom_table (str): The name of the custom POIs table
            categories (list): The categories of the custom POIs to be replaced

        Returns:
            gdf: _description_
        """ 

        sql_select_table = f"""
            SELECT * FROM {custom_table} 
            WHERE category IN (SELECT UNNEST(ARRAY{categories}));
        """

        custom_poi_gdf = gp.GeoDataFrame.from_postgis(
            sql_select_table, db_reading.return_sqlalchemy_engine()
        )

        if bool(set(list(poi_gdf.category.unique())) & set(categories)):
            poi_gdf = poi_gdf[~poi_gdf.category.isin(categories)]
            columns_to_drop = list(set(custom_poi_gdf.columns) - set(poi_gdf.columns))
            custom_poi_gdf.drop(columns_to_drop, axis=1, inplace=True)
            poi_gdf = pd.concat([poi_gdf, custom_poi_gdf], ignore_index=True)

        return poi_gdf

    # Indexing data in dataframe with goat indexes
    def dataframe_goat_index(df):
        db = Database("reading")
        con = db.connect_rd()
        cur = con.cursor()
        df = df[df["amenity"].notna()]
        df["id_x"] = df.centroid.x * 1000
        df["id_y"] = df.centroid.y * 1000
        df["id_x"] = df["id_x"].apply(np.floor)
        df["id_y"] = df["id_y"].apply(np.floor)
        df = df.astype({"id_x": int, "id_y": int})
        df["poi_goat_id"] = (
            df["id_x"].map(str) + "-" + df["id_y"].map(str) + "-" + df["amenity"]
        )
        df = df.drop(columns=["id_x", "id_y"])
        df["osm_id"] = df["osm_id"].fillna(value=0)
        df_poi_goat_id = df[["poi_goat_id", "osm_id", "name", "origin_geometry"]]

        cols = ",".join(list(df_poi_goat_id.columns))
        tuples = [tuple(x) for x in df_poi_goat_id.to_numpy()]

        cnt = 0

        cur.execute(
            "DROP TABLE IF EXISTS poi_goat_id_temp; \
                    CREATE TABLE poi_goat_id_temp AS TABLE poi_goat_id;"
        )

        for tup in tuples:
            tup_l = list(tup)
            id_number = tup_l[0]
            query_select = f"SELECT max(index) FROM poi_goat_id_temp WHERE poi_goat_id = '{id_number}';"
            last_number = db.select_rd(query_select)
            if (list(last_number[0])[0]) is None:
                tup_new = tup_l
                tup_new.append(0)
                tup_new = tuple(tup_new)
                cur.execute(
                    """INSERT INTO poi_goat_id_temp(poi_goat_id, osm_id, name, origin_geometry, index) VALUES (%s,%s,%s,%s,%s)""",
                    tup_new,
                )
                con.commit()
                df.iloc[cnt, df.columns.get_loc("poi_goat_id")] = f"{id_number}-0000"
            else:
                new_ind = list(last_number[0])[0] + 1
                tup_new = tup_l
                tup_l.append(new_ind)
                tup_new = tuple(tup_new)
                cur.execute(
                    """INSERT INTO poi_goat_id_temp(poi_goat_id, osm_id, name, origin_geometry, index) VALUES (%s,%s,%s,%s,%s)""",
                    tup_new,
                )
                con.commit()
                df.iloc[
                    cnt, df.columns.get_loc("poi_goat_id")
                ] = f"{id_number}-{new_ind:04}"
            cnt += 1

        cur.execute("DROP TABLE poi_goat_id_temp;")
        con.close()
        df = df.astype({"osm_id": int})
        return df


# pois_preparation = PoisPreparation(DATABASE)
# db_reading = Database(DATABASE_RD)
# pois_preparation.perform_pois_preparation(db_reading)

# db = Database(DATABASE)
#create_table_schema(db, DATABASE, "basic.aoi")


# create_table_schema(db, DATABASE, "basic.building")
# create_table_schema(db, DATABASE, "basic.population")
# create_table_schema(db, DATABASE, "basic.study_area")
# create_table_schema(db, DATABASE, "basic.sub_study_area")


# create_table_schema(db, DATABASE, "basic.grid_calculation")
# create_table_schema(db, DATABASE, "basic.grid_visualization")
# create_table_schema(db, DATABASE, "basic.study_area_grid_visualization")
 # create_table_schema(db, DATABASE, "basic.poi")

# create_table_dump(db_config=DATABASE, table_name="basic.building", data_only=True)
# create_table_dump(db_config=DATABASE, table_name="basic.population", data_only=True)
# create_table_dump(db_config=DATABASE, table_name="basic.study_area", data_only=True)
# create_table_dump(db_config=DATABASE, table_name="basic.sub_study_area", data_only=True)
# create_table_dump(db_config=DATABASE, table_name="basic.grid_calculation", data_only=True)
# create_table_dump(db_config=DATABASE, table_name="basic.grid_visualization", data_only=True)
# create_table_dump(db_config=DATABASE, table_name="basic.study_area_grid_visualization", data_only=True)
# create_table_dump(db_config=DATABASE, table_name="basic.edge", data_only=True)
# create_table_dump(db_config=DATABASE, table_name="basic.node", data_only=True)
# create_table_dump(db_config=DATABASE, table_name="basic.poi", data_only=True)

# create_table_schema(db, DATABASE, "basic.aoi")