import os
import sys
import geopandas as gpd
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree
from shapely import geometry
from collection import gdf_conversion, Config
from preparation import file2df
from db.db import Database
from db.config import DATABASE

# Create connection to geonode db
def geonode_connection():
    db = Database()
    con = db.connect()
    return con

# Publish dataframe in geonode datastore 
# df - dataframe, name - table name to store, if_exists="replace" to overwrite datatable
def df2geonode(df,name,if_exists='replace'):
    db = Database()
    con = db.connect_sqlalchemy()
    df.to_postgis(con=con, name=name, if_exists=if_exists)

# Returns geonode table as a dataframe 
def geonode_table2df(con, table_name, geometry_column='geometry'):
    query = "SELECT * FROM %s" % table_name
    df = gpd.read_postgis(con=con,sql=query, geom_col=geometry_column)
    return df

# Returns study area as df from geonode db (germany_municipalities) according to rs code 
def study_area2df(con,rs):
    query = "SELECT * FROM germany_municipalities WHERE rs = '%s'" % rs
    df_area = gpd.read_postgis(con=con,sql=query, geom_col='geom')
    df_area = df_area.filter(['geom'], axis=1)
    return df_area

# Creates DataFrame with buffer (default 8300 meters) of geometries, provided as a list of dataframes
def area_n_buffer2df(con, rs_set, buffer=8300):
    list_areas = []
    for rs in rs_set:
        df_area = study_area2df(con,rs)
        list_areas.append(df_area)
    df_area_union = pd.concat(list_areas,sort=False).reset_index(drop=True)
    df_area_union["dis_field"] = 1
    df_area_union = df_area_union.dissolve(by="dis_field")
    area_union_buffer = df_area_union
    area_union_buffer = area_union_buffer.to_crs(31468)
    area_union_buffer["geom"] = area_union_buffer["geom"].buffer(buffer)
    area_union_buffer = area_union_buffer.to_crs(4326)
    buffer_serie = area_union_buffer.difference(df_area_union)
    df_buffer_area = gpd.GeoDataFrame(geometry=buffer_serie)
    df_buffer_area = df_buffer_area.set_crs('epsg:4326')
    df_buffer_area = df_buffer_area.reset_index(drop=True)
    df_buffer_area = df_buffer_area.rename(columns={"geometry":"geom"})    
    df = pd.concat([df_area_union,df_buffer_area], sort=False).reset_index(drop=True)
    df["dis_field"] = 1
    df = df.dissolve(by="dis_field").reset_index(drop=True)
    return df

    # Fuction deaggregates address (street+housenumber) to separate strings returns tuple (street,number)
def addr_deaggregate(addr_street):
    street_l = []
    number_l = []
    addr_split = addr_street.split()
    for a in addr_split:
        if len(a) < 2 or any(map(str.isdigit, a)):
            number_l.append(a)
        else:
            street_l.append(a)

    street = ' '.join(street_l)
    number = ' '.join(number_l)
       
    return street, number

def df2area(df,df_area):
    df2area = gpd.overlay(df, df_area, how='intersection')
    return df2area

def find_nearest(gdA, gdB, max_dist):
    gdA.crs = "epsg:4326"
    gdB.crs = "epsg:4326"
    gdA = gdA.to_crs(31468)
    gdB = gdB.to_crs(31468)
    nA = np.array(list(gdA.geometry.apply(lambda x: (x.x, x.y))))
    nB = np.array(list(gdB.geometry.apply(lambda x: (x.x, x.y))))
    btree = cKDTree(nB)
    dist, idx = btree.query(nA, k=1)
    # gdB_nearest = gdB.iloc[idx].drop(columns={"geometry"}).reset_index(drop=True)
    gdB_nearest = gdB.iloc[idx].reset_index(drop=True)
    gdA = gdA.rename(columns={"addr:street": "street"})
    gdf = pd.concat(
        [
            gdA.reset_index(drop=True),
            gdB_nearest["osm_id"],
            pd.Series(dist, name='dist')
        ], 
        axis=1)

    gdf = gdf.rename(columns={"street": "addr:street", "city": "addr:city", "postcode" : "addr:postcode", "country": "addr:country"})
    gdf_fus = gdf[gdf.dist < max_dist]
    m = ~gdf.id.isin(gdf_fus.id)
    gdf_not_fus = gdf[m]
    gdf_fus = gdf_fus.to_crs(4326)
    gdf_not_fus = gdf_not_fus.to_crs(4326)
    gdf_fus = gdf_fus.drop(columns={"dist"})
    gdf_not_fus = gdf_not_fus.drop(columns={"dist"})
    gdf_not_fus["osm_id"] = None

    return gdf_fus, gdf_not_fus

# POIs fusion function for fusion custom pois data with osm data, referenced to table "germany_municipalities" with regions 
# RETURNS tuple(df,name) and saves data as file if is specified
# - df_base2area - df with base data  (osm) 
# - df_area - df with polygon of study area
# - df_input - df with data to fuse
# - amenity_replace - str name of amenity which objects will be removed from base osm df
# - amenity_update - str name of amenity to update (works together with brand_replace)
# - brand_replace - str brand to replace (works)
# - columns2drop_input - list(str) with column names which are not merging to base os df 
# - return_name - str name of data, can be used as filename
# - return_type - str ("GeoJSON", "GPKG") file type for storage return
def replace_data_area(df_base2area, df_area, df_input, amenity_replace=None, amenity_set=False, amenity_brand_replace=None, 
                      columns2rename=None, columns2drop=None, column_set_value=None, return_name = None, return_type=None):
    # Cut data to given area 
    df_input2area = gpd.overlay(df_input, df_area, how='intersection')

    # Remove amenity class from base dataframe
    if amenity_replace:
        df_base2area = df_base2area[df_base2area.amenity != amenity_replace]
    elif amenity_brand_replace:
        amenity_brand_replace = eval(amenity_brand_replace)
        df_base2area = df_base2area[(df_base2area.operator.str.lower() != amenity_brand_replace[1].lower()) & (df_base2area.amenity.str.lower() != amenity_brand_replace[0].lower())]
    else:
        print("Amenity (and brand) were not specified.. ")

    # Prepare input data for concatination
    if columns2rename:
        df_input2area = df_input2area.rename(columns=columns2rename)
    if columns2drop:
        df_input2area = df_input2area.drop(columns={*columns2drop})

    if amenity_brand_replace:
        df_input2area['amenity'] = amenity_brand_replace[0]
        df_input2area['brand'] = amenity_brand_replace[1]
        df_input2area['operator'] = amenity_brand_replace[1].lower()
    elif amenity_set:
        df_input2area['amenity'] = amenity_replace

    df_input2area['amenity'] = df_input2area['amenity'].str.lower()

    # Set tags and geom from osm data to input

    # Deaggregate street and number
    if "addr:street" in df_input2area.columns.tolist():
        if 'housenumber' not in df_input2area.columns.tolist():
            for i in df_input2area.index:
                df_row = df_input2area.iloc[i]
                address = addr_deaggregate(df_row["addr:street"])
                df_input2area.at[i,"addr:street"] = address[0]
                df_input2area.at[i,"housenumber"] = address[1]
    
    if column_set_value:
        for col in column_set_value.keys():
            df_input2area[col] = column_set_value[col]


    # Concatination of dataframes
    df_result = pd.concat([df_base2area,df_input2area],sort=False).reset_index(drop=True)
    df_result = df_result.replace({np.nan: None})


    # Writedown result of fusion
    return gdf_conversion(df_result, return_name, return_type=return_type)


    # function to replace old fusion

def fuse_data_area(df_base2area, df_area, df_input, amenity_fuse=None, amenity_set=False, amenity_brand_fuse=None, 
                   columns2rename=None, columns2drop=None, column_set_value=None, columns2fuse=None, return_name = None, return_type=None):
    # Cut input data to given area 
    df_input2area = gpd.overlay(df_input, df_area, how='intersection')

    # Sort df by config settings from base dataframe
    if amenity_fuse:
        df_base_amenity = df_base2area[df_base2area.amenity == amenity_fuse]
        df_base2area_rest = df_base2area[df_base2area.amenity != amenity_fuse]
    elif amenity_brand_fuse:
        amenity_brand_fuse = eval(amenity_brand_fuse)
        df_base_amenity = df_base2area[((df_base2area.operator.str.lower() == amenity_brand_fuse[1].lower()) |
                                        (df_base2area.brand.str.lower() == amenity_brand_fuse[1].lower())) & 
                                        (df_base2area.amenity.str.lower() == amenity_brand_fuse[0].lower())]
        df_base2area_rest = df_base2area[~df_base2area.apply(tuple,1).isin(df_base_amenity.apply(tuple,1))]
    else:
        print("Amenity (and brand) were not specified.. ")

    if "addr:street" in df_input2area.columns.tolist():
        if 'housenumber' not in df_input2area.columns.tolist():
            for i in df_input2area.index:
                df_row = df_input2area.iloc[i]
                address = addr_deaggregate(df_row["addr:street"])
                df_input2area.at[i,"addr:street"] = address[0]
                df_input2area.at[i,"housenumber"] = address[1]

    if column_set_value:
        for col in column_set_value.keys():
            df_input2area[col] = column_set_value[col]

    # Create temp sorted base dataframe 
    df_base_temp = df_base_amenity[["geometry", "osm_id"]]      
    # find closest points to fuse, default max distance for fusion 150 meters
    # return 2 df - find closests and not
    df_fus, df_not_fus = find_nearest(df_input2area, df_base_temp, 250)

    fus_col_fus = columns2fuse
    fus_col_fus.append("osm_id")
    df_fus = df_fus[fus_col_fus]

    # Prepare input data for concatination
    if columns2rename:
        df_not_fus = df_not_fus.rename(columns=columns2rename)
    if columns2drop:
        df_not_fus = df_not_fus.drop(columns={*columns2drop})

    if amenity_brand_fuse:
        df_not_fus['amenity'] = amenity_brand_fuse[0]
        df_not_fus['operator'] = amenity_brand_fuse[1].lower()
        columns2fuse.extend(('amenity', 'operator', 'source', 'geometry'))
        df_not_fus = df_not_fus[columns2fuse]
    elif amenity_set:
        df_not_fus['amenity'] = amenity_fuse.lower()
        columns2fuse.extend(('amenity', 'source', 'geometry'))
        df_not_fus = df_not_fus[columns2fuse]
    # df_not_fus['amenity'] = df_not_fus['amenity'].str.lower()

    # Concatination of dataframes
    df_result = (df_fus.set_index('osm_id').combine_first(df_base_amenity.set_index('osm_id')))
    df_result = df_result.reset_index()
    df_result = gpd.GeoDataFrame(df_result, geometry="geometry")
    df_result = pd.concat([df_result,df_not_fus],sort=False)
    df_result = df_result.replace({np.nan: None})
    df_result = pd.concat([df_result,df_base2area_rest],sort=False)
    df_result.crs = "epsg:4326"

    return gdf_conversion(df_result, return_name, return_type=return_type)

def fusion_set(config, result_name=None, return_type=None):
    con = geonode_connection()

    table_base = config.fusion["table_base"]
    rs_set = config.fusion["rs_set"]
    typen = ["geonode","geojson"]

    df_base = geonode_table2df(con, table_base, geometry_column="geom")
    df_area = area_n_buffer2df(con, rs_set, buffer=8300)
    df_base2area = df2area(df_base, df_area)

    for typ in typen:
        fusion_key_set = config.fusion_key_set(typ)

        for key in fusion_key_set:
            print(key)
            fusion_set = config.fusion_set(typ,key)
            if typ == 'geojson':
                filename = key + '.' + typ
                df_input = file2df(filename)
            elif typ == 'geonode':
                df_input = geonode_table2df(con, key)
            df_input2area = df2area(df_input, df_area)
            
            if config.fusion_type(typ, key) == "fuse":
                df_base2area = fuse_data_area(df_base2area, df_area, df_input, *fusion_set, return_name = None, return_type=None)[0]
            elif config.fusion_type(typ, key) == "replace":
                df_base2area = replace_data_area(df_base2area, df_area, df_input, *fusion_set[:-1], return_name = None, return_type=None)[0]
            else:
                print("Fusion type for %s was not defined. Fusion was not done." % key)
                pass

    return gdf_conversion(df_base2area, result_name, return_type=return_type)