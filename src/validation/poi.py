import os
import numpy as np
import subprocess
import pandas as pd
import geopandas as gpd
from shapely import wkb, wkt
import difflib
from typing import Optional, Tuple
from math import radians, sin, cos, sqrt, atan2

from scipy.spatial import KDTree
from src.config.config import Config
from src.core.config import settings
from src.db.db import Database
from src.utils.utils import timing, print_info, print_hashtags, print_warning, print_error


class PoiValidation:
    """ 
    Validates the POIs by doing a comparison between existing and new points.

    Args:
        df (pl.DataFrame): POIs to classify.

    Returns:
        pl.DataFrame: Classified POIs.
    """
    
    # Definition of the class attributes and config variables from the poi.yaml file
    def __init__(self, db_config, region):
        self.db_config = db_config
        self.region = region
        self.data_dir = settings.INPUT_DATA_DIR
        self.config = Config("poi", region)
        self.old_poi_tables = self.config.validation["old_poi_table"]
        self.new_poi_table = self.config.validation["new_poi_table"]
        
        self.all_metrics = list(self.config.validation["metrics"].keys())
        
        self.lcs_config = self.config.validation['lcs']
        self.poi_columns = self.lcs_config['poi_columns']
        self.excluded_categories = self.lcs_config['excluded_categories']
        self.proximity_radius_m = self.lcs_config['search_radius_m']
        self.name_weight = self.lcs_config['weights']['name']
        self.category_weight = self.lcs_config['weights']['category']
        self.threshold_lcs = self.lcs_config['threshold_lcs']
 
    
    def get_geom_reference_table_and_query(self, metric_name):
        """
        Get the geometry reference query for a specific metric.
        
        Args:
            metric_name (str): The name of the metric for which to get the geometry reference query.
        
        Returns:
            str: The geometry reference query for the specified metric.
        """
        if metric_name == "poi_count":
            geom_reference_query = self.config.validation["metrics"][metric_name]["geom_reference_query"]
            temp_ref_clone_name = geom_reference_query.split("FROM", 1)[1].strip().split()[0]
            return geom_reference_query, temp_ref_clone_name
        elif metric_name == "poi_density":
            geom_reference_query = self.config.validation["metrics"][metric_name]["geom_reference_query"]
            temp_ref_clone_name = geom_reference_query.split("FROM", 1)[1].strip().split()[0]
            return geom_reference_query, temp_ref_clone_name
        elif metric_name == "poi_per_people":
            geom_reference_query = self.config.validation["metrics"][metric_name]["geom_reference_query"]
            temp_ref_clone_name = geom_reference_query.split("FROM", 1)[1].strip().split()[0]
            return geom_reference_query, temp_ref_clone_name
        elif metric_name == "population_per_poi":
            geom_reference_query = self.config.validation["metrics"][metric_name]["geom_reference_query"]
            temp_ref_clone_name = geom_reference_query.split("FROM", 1)[1].strip().split()[0]
            return geom_reference_query, temp_ref_clone_name
        else:
            raise ValueError(f"Unknown metric: {metric_name}")
       
    def get_metric_clause(self, metric_name):
        """
        Get the SQL clause for a specific metric.
        
        Args:
            metric_name (str): The name of the metric for which to get the SQL clause.
        
        Returns:
            str: The SQL clause for the specified metric.
        """
        if metric_name == "poi_count":
            return f"COUNT(*) AS poi_count"
        elif metric_name == "poi_density":
            return f"""
                ROUND(CAST(COUNT(poi.category) AS NUMERIC) / (ST_Area(ST_Transform(ref.geom, 3857))::NUMERIC / 10000.0::NUMERIC), 5)::NUMERIC AS poi_density
            """
        elif metric_name == "poi_per_people":
            people_count = self.config.validation["metrics"][metric_name].get("capita_people_count", "N/A")
            return f"""
                CASE
                    WHEN ref.einwohnerzahl_ewz > 0 THEN
                        ROUND((CAST(COUNT(*) AS NUMERIC) / ref.einwohnerzahl_ewz::NUMERIC) * {people_count}::NUMERIC, 2)::NUMERIC
                    ELSE 0::NUMERIC
                END AS poi_per_people
            """
        elif metric_name == "population_per_poi":
            return f"""
                CASE
                    WHEN COUNT(*) > 0 THEN
                        (FLOOR(ref.einwohnerzahl_ewz::NUMERIC / CAST(COUNT(*) AS NUMERIC))::NUMERIC)::INT
                    ELSE NULL
                END AS population_per_poi
            """
        else:
            raise ValueError(f"Unknown metric: {metric_name}")


    def create_temp_geom_reference_table(self, db_old: Database, db_new: Database, geom_reference_query, temp_ref_clone_name):
        """
        Copy the geometry reference table from raw database to local database using ogr2ogr.

        Args:
            db_old (Database): Database connection to the old database.
            db_new (Database): Database connection to the new database.
            metric (str, optional): The metric to use for validation. Defaults to None.
            temp_ref_clone_name (str, optional): Name for the temporary reference table. Defaults to None.

        Returns:
            str: The name of the temporary geometry reference table created in the new database.
        """

        # Extract the reference geomtry table name from the yaml definition inside validation object
        print_info(f"Creating clone of '{temp_ref_clone_name}' in new database")
        print_hashtags()
        
        # Extract the new and old db credentials to parse to ogr2ogr command
        local_host = settings.LOCAL_DATABASE_URI.host
        local_db = settings.LOCAL_DATABASE_URI.path.replace("/", "")
        local_user = settings.LOCAL_DATABASE_URI.user
        local_password = settings.LOCAL_DATABASE_URI.password
        local_port = settings.LOCAL_DATABASE_URI.port
        
        # Extract the old db credentials to parse to ogr2ogr command
        raw_host = settings.RAW_DATABASE_URI.host
        raw_db = settings.RAW_DATABASE_URI.path.replace("/", "")
        raw_user = settings.RAW_DATABASE_URI.user
        raw_password = settings.RAW_DATABASE_URI.password
        raw_port = settings.RAW_DATABASE_URI.port
        
        # Run the ogr2ogr command to copy the geometry reference table from old to new database
        ogr2ogr_command = (
            f"ogr2ogr -f 'PostgreSQL' "
            f"PG:'host={local_host} dbname={local_db} user={local_user} password={local_password} port={local_port}' "
            f"PG:'host={raw_host} dbname={raw_db} user={raw_user} password={raw_password} port={raw_port}' "
            f"-nln {temp_ref_clone_name} -sql \"{geom_reference_query}\""
        )
    
        try:
            subprocess.run(ogr2ogr_command, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print_error(f"Reference geometry data table copy failed: {e}")
    
        # It returns the name of the temporary reference geometry reference table    
        return temp_ref_clone_name
    
    def drop_temp_geom_reference_table(self, db: Database, temp_ref_clone_name: str):
        """
        Drop the temporary reference geometry reference table from the database.

        Args:
            db (Database): Database connection to the database.
            temp_ref_clone_name (str): Name of the temporary reference geometry table to drop.
        """

        print_info(f"Dropping temporary reference geometry table '{temp_ref_clone_name}'")
        print_hashtags()
        
        # Execute the drop table query
        drop_query = f"DROP TABLE IF EXISTS {temp_ref_clone_name};"
        db.perform(drop_query)
    
    def generate_group_by_clause(self, geom_reference_query):
        """
        This function extracts the column names from the reference geometry query and creates a mapping dictionary for running queries on new and old database.

        Structure:
            {
                "new_column_alias": "old_column_name"
            }
        
        Returns:
            tuple: A tuple containing:
                - new_and_old_columns_mapping (dict): Mapping of new column aliases to old column names.
                - new_group_by_clause (str): Group by clause for new query.
                - old_group_by_clause (str): Group by clause for old query.
        """

        # Extract the columns part from the reference geometry query and split it into individual columns
        columns_part = geom_reference_query.split("SELECT")[1].split("FROM")[0].strip()
        group_by_list = [f"ref.{col.strip()}" for col in columns_part.split(",")]
        group_by_clause = ", ".join(group_by_list)

        return group_by_clause

    def perform_spatial_intersection(self, db, poi_table, temp_ref_clone_name, group_by_clause, metric_clause):
        """
        Perform a spatial intersection between the POI table and the temporary reference clone.
        
        Args:
            db (Database): Database connection to the database.
            poi_table (str): Name of the POI table to join with the reference geometry table.
            temp_ref_clone_name (str): Name of the temporary reference geometry table.
            group_by_clause (str): Group by clause for the query.
            metric_clause (str): Metric clause for the query.
        
        Returns:
            list: Results of the spatial join as a list of dictionaries.
        """
        
        print_info(f"Spatially joining '{poi_table}' <-> '{temp_ref_clone_name}'")
        print_hashtags()
        
        spatial_join_condition = "ST_Within(poi.geom, ref.geom)"
        spatial_join_query = f"""
            SELECT {group_by_clause}, poi.category, {metric_clause}
            FROM {temp_ref_clone_name} AS ref
            CROSS JOIN LATERAL (
                SELECT category, geom
                FROM {poi_table} AS poi
                WHERE poi.category IS NOT NULL
                AND {spatial_join_condition}
            ) AS poi
            GROUP BY {group_by_clause}, poi.category
            ORDER BY poi.category;
        """
        
        spatial_join_results = db.select(spatial_join_query)
        return spatial_join_results

    def run_metric_based_validation(self, db_old, db_new, temp_geom_clone_table, metric):
        """
        Run the core validation functions to create merged results for new and old POI tables.

        Args:
            db_old (Database): Database connection to the old database.
            db_new (Database): Database connection to the new database.
            temp_geom_clone_table (str): Name of the temporary geometry reference table.
            metric (str): The metric to use for validation.

        Returns:
            dict: A dictionary containing the spatial join results for new and old POI tables.
        """

        print_info(f"Creating merged results for new and old POI tables")
        print_hashtags()

        spatial_join_results = {
            "new": {},
            "old": {}
        }

        geom_reference_query, _ = self.get_geom_reference_table_and_query(metric)

        # Generate the group by clause
        group_by_clause = self.generate_group_by_clause(geom_reference_query)

        # Append a prefix to the metric key for new and old results
        new_metric_key = f"new_{metric}"
        old_metric_key = f"old_{metric}"

        metric_clause = self.get_metric_clause(metric)

        # Extract column names from the geom_reference_query
        columns_part = geom_reference_query.split("SELECT")[1].split("FROM")[0].strip()
        column_names = [col.strip() for col in columns_part.split(",")]

        # Prepare the keys for new and old results
        new_keys = [f"new_{col}" for col in column_names] + ["new_category", new_metric_key]
        old_keys = [f"old_{col}" for col in column_names] + ["old_category", old_metric_key]

        # Perform the spatial intersection for the new POI table and temporary reference clone table
        new_results_old = self.perform_spatial_intersection(
            db_new, self.new_poi_table, temp_geom_clone_table, group_by_clause, metric_clause
        )
        new_results = [
            dict(zip(new_keys, row))
            for row in new_results_old
        ]

        # Append the results to the dictionary of spatial joins
        spatial_join_results["new"][self.new_poi_table] = new_results

        # Do the same for each old POI table with temporary reference clone table
        for old_poi_table in self.old_poi_tables:
            old_results_old = self.perform_spatial_intersection(
                db_old, old_poi_table, temp_geom_clone_table, group_by_clause, metric_clause
            )
            old_results = [
                dict(zip(old_keys, row))
                for row in old_results_old
            ]
            spatial_join_results["old"][old_poi_table] = old_results

        return spatial_join_results

    # Helper to handle exponential notation and format floats to 5 decimal places
    def format_float(self, val):
        if isinstance(val, float):
            return float(f"{val:.5f}".rstrip('0').rstrip('.') if '.' in f"{val:.5f}" else f"{val:.5f}")
        return val

    def compare_metric_based_new_and_old_results(self, spatial_join_results, metric):
        """
        Compare the new and old results based on matching id, name, geom, and category.
        
        Args:
            spatial_join_results (dict): Dictionary containing the spatial join results for new and old POI tables.
            metric (str): The metric to use for validation.
            
        Returns:
            dict: A dictionary containing the unified results of the comparison.
        """
        
        print_info(f"Filtering pois based on matching reference geometry table fields")
        print_hashtags()
        
        # Prepare a dictionary to hold the final dataframe for export as GPKG and MD
        unified_results = {}
        # Get the new results (should be a list of dicts) from the spatial join results
        new_pois = spatial_join_results["new"].get(self.new_poi_table, [])
        
        # Extract column names from the geom_reference_query
        geom_reference_query, _ = self.get_geom_reference_table_and_query(metric)
        columns_part = geom_reference_query.split("SELECT")[1].split("FROM")[0].strip()
        column_names = [col.strip() for col in columns_part.split(",")]

        # Prepare the keys for new and old results
        new_keys_prefix = [f"new_{col}" for col in column_names]
        old_keys_prefix = [f"old_{col}" for col in column_names]

        # Category is a special case, so we add it separately since it is not in the mapping and reference query
        new_category_key = "new_category"
        new_metric_key = f"new_{metric}"
        old_category_key = "old_category"
        old_metric_key = f"old_{metric}"

        # Build a lookup for new pois: (id, name, geom, category) -> record
        new_lookup = {}
        for rec in new_pois:
            # Build the key tuple dynamically (all mapped keys + category)
            key = tuple(rec.get(k) for k in new_keys_prefix) + (rec.get(new_category_key),)
            new_lookup[key] = rec

        # This loop runs over the old results and then matches them with the new pois
        for old_table, old_pois in spatial_join_results["old"].items():
            comparison_list = []
            # For each old record, create a key and check if it exists in the new lookup
            for old_rec in old_pois:
                key = tuple(old_rec.get(k) for k in old_keys_prefix) + (old_rec.get(old_category_key),)
                new_rec = new_lookup.get(key)
                # If a matching new record is found, calculate the percentage difference
                if new_rec:
                    new_val = new_rec.get(new_metric_key, 0)
                    old_val = old_rec.get(old_metric_key, 0)
                    try:
                        # Calculate percentage difference: ((new - old) / old) * 100
                        if float(old_val) != 0:
                            difference = int(new_val) - int(old_val)
                            perc_diff = round(((float(new_val) - float(old_val)) / float(old_val)) * 100, 2)
                        else:
                            difference = None
                            perc_diff = None
                    except Exception:
                        difference = None 
                        perc_diff = None
                    
                    # Prepare the output that contains the comparison results and filters out duplicate column names either from new or old columns
                    #  id, name, category, geom come from either new or old pois
                    # new_count and old_count are the counts from new and old pois respectively
                    # perc_diff is the calculated percentage difference
                    geom_key = next((k for k in new_keys_prefix if "geom" in k), None)
                    output = {
                        "id": new_rec.get(new_keys_prefix[0]),
                        "name": new_rec.get(new_keys_prefix[1]),
                        "category": new_rec.get(new_category_key),
                        new_metric_key: self.format_float(new_val), # Use dynamic metric key
                        old_metric_key: self.format_float(old_val), # Use dynamic metric key
                        "difference": self.format_float(difference),
                        "perc_diff": self.format_float(perc_diff),
                        "geom": new_rec.get(geom_key),
                    }
                    comparison_list.append(output)
            unified_results[old_table] = comparison_list
        return unified_results

    def generate_metrics_based_gpkg_file(self, pois, output_path):
        
        """
        Export the validation results to a GPKG file, creating separate layers for each category.
        
        Args:
            pois (dict): Dictionary containing the validation results.
            output_path (str): Path to the output GPKG file.
        """
        
        print_info(f"Exporting to GPKG: {output_path}")
        print_hashtags()
        
        for old_table, recs in pois.items():
            if not recs:
                continue
            
            # Dynamically get the geometry key (find the key that contains 'geom')
            geom_key = next((k for k in recs[0].keys() if "geom" in k), None)
            category_key = next((k for k in recs[0].keys() if "category" in k), None)
            if not geom_key or not category_key:
                continue

            # Prepare the dataframe to hold the pois for export
            df = pd.DataFrame(recs)

            # Fix: Ensure numeric columns are float and fill missing values
            for col in df.columns:
                if col.startswith("new_") or col.startswith("old_") or col == "perc_diff":
                    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

            # Convert geometry column to shapely objects for spatial export
            def parse_geom(val):
                if val is None:
                    return None
                try:
                    if hasattr(val, "geom_type"):
                        return val
                    return wkb.loads(val, hex=True)
                except Exception:
                    try:
                        return wkt.loads(val)
                    except Exception:
                        return None

            df["geometry"] = df[geom_key].apply(parse_geom)

            # Sort the DataFrame by category (ascending) and perc_diff (descending)
            df = df.sort_values(by=[category_key, "perc_diff"], ascending=[True, False])

            # Export each category as a separate layer inside the GPKG file
            for category, group in df.groupby(category_key):
                table_name = old_table.split(".", 1)[1]
                layer_name = f"{table_name}-{str(category)}"
                gdf = gpd.GeoDataFrame(group.drop(columns=[geom_key]), geometry="geometry", crs="EPSG:4326")
                gdf.to_file(output_path, layer=layer_name, driver="GPKG")

    def generate_metrics_based_markdown_report(self, pois, output_path, region, metric, temp_ref_clone_name):
        
        """
        Generate a Markdown report summarizing the validation results.
        
        Args:
            pois (dict): Dictionary containing the validation results.
            output_path (str): Path to the output Markdown file.
            region (str): The region for which the report is generated.
            metric (str, optional): The metric to use for validation. Defaults to None.
        """
        
        print_info(f"Generating Report: {output_path}")
        print_hashtags()

        # Setup configuration
        new_table = self.new_poi_table
        metric_config = self.config.validation["metrics"][metric]
        thresholds = metric_config.get("thresholds", {})
        
        # Units description based on metric
        units = "Number of Features"
        people_count = metric_config.get("capita_people_count", "N/A")
        if metric == "poi_per_people":
            units = f"POIs per {people_count} people"
        elif metric == "population_per_poi":
            units = "People per POI"


        # A function definition to format rows for Markdown tables
        def format_row(row, widths):
            """Format a row for Markdown with padded column widths."""
            escaped_row = [str(val).replace("_", "\_") for val in row]
            return "| " + " | ".join(str(val).ljust(width) for val, width in zip(escaped_row, widths)) + " |\n"

        # Create the Markdown file and write the header
        with open(output_path, 'w') as f:
            # f.write(
            #     """<style>
            #     @import url('https://fonts.googleapis.com/css2?family=Barlow&display=swap');
            #     body {
            #         font-family: 'Barlow', sans-serif;
            #     }
            #     </style>\n\n"""
            # )
            f.write(f"# Validation Report - **{metric.replace('_', ' ').upper()}** Metric\n\n")

            for old_table, recs in pois.items():
                if not recs:
                    continue

                # Section Header Info
                f.write(f"### 📄 Table Details\n\n")
                f.write(f"- **Reference Geometry Table:** `{temp_ref_clone_name}`\n")
                f.write(f"- **Raw Database Table:** `{old_table}`\n")
                f.write(f"- **Local Database Table:** `{new_table}`\n")
                f.write(f"- **Region:** `{region}`\n")
                # f.write(f"- **Metric:** `{metric}`\n")
                f.write(f"- **Units:** `{units}`\n\n")

                # 🚨 Threshold Violations Table
                f.write("## 🚨 Threshold Violations\n\n")

                headers = [
                    "Category", 
                    "Admin Name", 
                    f"Old {metric.replace('_', ' ').title()}", 
                    f"New {metric.replace('_', ' ').title()}",
                    "Difference",
                    "Difference (%)", 
                    "Threshold (%)", 
                    "Region ID"
                ]
                violations = []
                summary_stats = {}

                # Sort recs by category (ascending) and perc_diff (descending)
                recs = sorted(recs, key=lambda r: (str(r.get("category", "")).lower(), -(r.get("perc_diff") if r.get("perc_diff") is not None else float('-inf'))))

                for rec in recs:
                    category = str(rec["category"]).capitalize()
                    county = rec.get("name", "")
                    # Access old_count or old_density based on 'metric'
                    old_val = float(rec.get(f"old_{metric}", 0)) # Dynamic access
                    # Access new_count or new_density based on 'metric'
                    new_val = float(rec.get(f"new_{metric}", 0)) # Dynamic access
                    difference = float(rec.get("difference", 0))
                    perc_diff = float(rec.get("perc_diff", 0))
                    threshold = thresholds.get(str(rec["category"]).lower(), thresholds.get("default", 0))
                    region_id = rec.get("id", "")

                    # Check if the difference exceeds the relevant threshold then append the record to violations
                    is_violation = perc_diff is not None and threshold is not None and abs(perc_diff) > float(threshold)
                    if is_violation:
                        violations.append([
                            category, county, 
                            self.format_float(old_val),
                            self.format_float(new_val),
                            self.format_float(difference),
                            self.format_float(perc_diff),
                            self.format_float(threshold),
                            region_id
                        ])

                    # This snippet updates the summary statistics for each category
                    cat_key = str(rec["category"]).lower()
                    summary_stats.setdefault(cat_key, {
                        "violations": 0, "old_total": 0, "new_total": 0, "diffs": []
                    })
                    if is_violation:
                        summary_stats[cat_key]["violations"] += 1
                    summary_stats[cat_key]["old_total"] += old_val
                    summary_stats[cat_key]["new_total"] += new_val
                    if perc_diff is not None:
                        summary_stats[cat_key]["diffs"].append(abs(perc_diff))

                # This snippet renders the violations table
                all_rows = [headers] + violations if violations else [headers]
                col_widths = [max(len(str(row[i])) for row in all_rows) for i in range(len(headers))]
                f.write(format_row(headers, col_widths))
                f.write("|" + "|".join("-" * (w + 2) for w in col_widths) + "|\n")
                for row in violations:
                    f.write(format_row(row, col_widths))
                f.write(f"\n**Total Violations Found:** {len(violations)}\n\n")

                # 📊 Summary Statistics Section
                f.write("## 📊 Summary Statistics\n\n")
                stats_headers = ["Statistic", "Value"]
                stats_rows = [
                    ["Admin Units Analyzed", len(set(rec.get("id") for rec in recs))],
                    ["Categories Analyzed", len(set(rec.get("category") for rec in recs))],
                    ["Combinations of Admin Units and Categories", len(recs)],
                    ["Combinations Surpassing the Violation Threshold", len(violations)],
                    ["Violation Rate", f"{(len(violations) / len(recs) * 100):.2f}%" if recs else "0.00%"]
                ]
                
                stats_col_widths = [max(len(str(row[i])) for row in [stats_headers] + stats_rows) for i in range(2)]
                f.write(format_row(stats_headers, stats_col_widths))
                f.write("|" + "|".join("-" * (w + 2) for w in stats_col_widths) + "|\n")
                for row in stats_rows:
                    f.write(format_row(row, stats_col_widths))
                f.write("\n")

                # Add a disclaimer about possible combinations
                f.write("> <span style='color: red;'>⚠️ **Disclaimer:**</span> The possible combinations of administrative units and categories represent all theoretical pairs (admin units × categories). However, not every administrative unit contains every category, so the actual number of analyzed combinations may be lower than the maximum possible. \n\n")

                temp_clone = temp_ref_clone_name.replace("_", "\\_")
                old_table = old_table.replace("_", "\\_")
                # Add a Markdown paragraph describing the summary statistics
                f.write(
                    f"**Dataset Length**: _The total count of spatial join results between {temp_clone} <-> {old_table}._  \n"
                    f"**Admin Units Analyzed**: _The number of unique administrative regions considered in the analysis._  \n"
                    f"**Categories Analyzed**: _The number of different POI categories that were analyzed within each administrative unit._  \n"
                    f"**Violations Found**: _The total number of category-specific violations exceeding the defined threshold within each administrative unit._  \n"
                    f"**Violation Rate**: _The percentage of category-specific violations exceeding the defined threshold, relative to the total categories analyzed within each administrative unit._  \n\n"
                )

                # 📋 Violation Breakdown by Category Section
                f.write("## 📋 Violation Breakdown by Category \n\n")
                breakdown_headers = [
                    "Category", 
                    "Violations", 
                    f"Old Total ({metric.replace('_', ' ').title()})", 
                    f"New Total ({metric.replace('_', ' ').title()})", 
                    "Avg Diff (%)", 
                    "Max Diff (%)"
                ]
                breakdown_rows = []
                for cat, stats in summary_stats.items():
                    avg_diff = sum(stats["diffs"]) / len(stats["diffs"]) if stats["diffs"] else 0
                    max_diff = max(stats["diffs"]) if stats["diffs"] else 0
                    breakdown_rows.append([
                        cat,
                        stats["violations"],
                        self.format_float(stats['old_total']),
                        self.format_float(stats['new_total']),
                        self.format_float(avg_diff),
                        self.format_float(max_diff)
                    ])
                breakdown_col_widths = [
                    max(len(str(row[i])) for row in [breakdown_headers] + breakdown_rows)
                    for i in range(len(breakdown_headers))
                ]
                f.write(format_row(breakdown_headers, breakdown_col_widths))
                f.write("|" + "|".join("-" * (w + 2) for w in breakdown_col_widths) + "|\n")
                for row in breakdown_rows:
                    f.write(format_row(row, breakdown_col_widths))
                f.write("\n\n---\n\n")

    # ************************************************************************************** #
    # ********************** LOWEST COMMON SUBSEQUENCE (LCS) ANALYSIS ********************** #
    # ************************************************************************************** #

    def get_lcs_similarity_sql(self):
        """
        Returns SQL snippet for LCS similarity, category similarity, and overall similarity using pg_trgm.
        Uses columns and weights from YAML config (already loaded in __init__).
        
        Args:
            None
            
        Returns:
            str: SQL snippet for calculating LCS similarity, category similarity, overall similarity, and ge
        """
        id_col, name_col, category_col, geom_col = self.poi_columns
        name_weight = self.name_weight
        category_weight = self.category_weight
        threshold_lcs = self.threshold_lcs
    
        string_similarity_query = f"""
            similarity(ref.{name_col}, comp.{name_col}) AS name_lcs,
            similarity(ref.{category_col}, comp.{category_col}) AS category_lcs,
            ROUND(
                ((similarity(ref.{name_col}, comp.{name_col}) * {name_weight}) +
                (similarity(ref.{category_col}, comp.{category_col}) * {category_weight}))::numeric,
                2
            ) AS overall_lcs,
            ROUND(ST_Distance(ref.{geom_col}::geography, comp.{geom_col}::geography)::numeric, 2) AS distance_m,
            (ROUND(
                ((similarity(ref.{name_col}, comp.{name_col}) * {name_weight}) +
                (similarity(ref.{category_col}, comp.{category_col}) * {category_weight}))::numeric,
                2
            ) >= {threshold_lcs}) AS flagged
        """

        return string_similarity_query

    def run_lcs_analysis_sql(self, db_new):
        """
        Runs LCS analysis using pure SQL with pg_trgm and PostGIS, using columns from YAML config.
        
        Args:
        db_new (Database): Database connection to the new POI database.
        
        Returns:    
            pd.DataFrame: A DataFrame containing the results of the LCS similarity analysis, including
            reference and comparison POI details, geodesic distance, LCS scores, and whether the pair is flagged as similar.
        """

        print_info(f"Performing LCS Analysis on '{self.new_poi_table}' using PSQL 'Similarity' and ST_Distance()")
        print_hashtags()

        id_col, name_col, category_col, geom_col = self.poi_columns
        similarity_sql = self.get_lcs_similarity_sql()
        table = self.new_poi_table
        search_radius = self.proximity_radius_m
        excluded_categories_clause = ", ".join([f"'{cat}'" for cat in self.excluded_categories])

        query = f"""
            SELECT
                ref.{id_col} AS ref_id,
                ref.{name_col} AS ref_name,
                ref.{category_col} AS ref_category,
                ST_AsText(ref.{geom_col}) AS ref_geom,
                comp.{id_col} AS comp_id,
                comp.{name_col} AS comp_name,
                comp.{category_col} AS comp_category,
                ST_AsText(comp.{geom_col}) AS comp_geom,
                {similarity_sql}
            FROM {table} ref
            JOIN {table} comp
                ON ref.{id_col} <> comp.{id_col}
                AND ST_DWithin(
                    ST_Transform(ref.{geom_col}, 3857),
                    ST_Transform(comp.{geom_col}, 3857),
                    {search_radius}
                )
            WHERE ref.{name_col} IS NOT NULL AND comp.{name_col} IS NOT NULL
            AND ref.{category_col} IS NOT NULL AND comp.{category_col} IS NOT NULL
            AND ref.{category_col} NOT IN ({excluded_categories_clause})
            AND comp.{category_col} NOT IN ({excluded_categories_clause})
        """
    
        print_info("Running SQL query for LCS analysis...")
        df = db_new.select(query)
        print_info(f"Query returned {len(df) if df else 0} rows.")
        print_hashtags()

        columns = [
            "ref_id", "ref_name", "ref_category", "ref_geom",
            "comp_id", "comp_name", "comp_category", "comp_geom",
            "name_lcs", "category_lcs", "overall_lcs", "distance_m", "flagged"
        ]
        lcs_df = pd.DataFrame(df, columns=columns) if df else pd.DataFrame(columns=columns)

        return lcs_df
    
    def prepare_lcs_dataframe_for_export(self, df: pd.DataFrame) -> pd.DataFrame:
        
        """
        Prepares the LCS DataFrame for export by filtering and restructuring it.
        
        Args:
            df (pd.DataFrame): DataFrame containing the results of the LCS similarity analysis.
            
        Returns:
            pd.DataFrame: A DataFrame structured for export, with grouped reference and comparison POI details,
            geodesic distance, LCS scores, and whether the pair is flagged as similar.
        """
        
        if df is None or df.empty:
            return pd.DataFrame()

        # Only keep rows where flagged is True (possible bugs)
        df = df[df['flagged'] == True].copy()
        if df.empty:
            return pd.DataFrame()

        # Do the aggregation of compared POIs based on each reference POI
        group_cols = ['ref_id', 'ref_name', 'ref_category', 'ref_geom']
        comp_cols = [col for col in df.columns if col.startswith('comp_')]
        array_group_cols = ['distance_m', 'name_lcs', 'category_lcs', 'overall_lcs', 'flagged']
        other_cols = [col for col in df.columns if col not in group_cols + comp_cols + array_group_cols]

        seen_comp_ids = set()
        seen_ref_ids = set()
        output_rows = []

        # Sort for deterministic grouping
        df = df.sort_values(group_cols).reset_index(drop=True)

        for _, row in df.iterrows():
            ref_id = row['ref_id']
            comp_id = row['comp_id']

            # If this ref_id has already been grouped as a comp_id, skip this row
            if ref_id in seen_comp_ids:
                continue

            # If this comp_id has already been a ref_id before, skip this row
            if comp_id in seen_ref_ids:
                continue

            # Group all rows with this ref_id
            group = df[df['ref_id'] == ref_id].copy()
            # Deduplicate comp_id and keep all comp_* and array_group_cols in sync
            if 'comp_id' in group.columns:
                _, unique_indices = np.unique(group['comp_id'], return_index=True)
                unique_indices = sorted(unique_indices)
                agg = {}
                for col in comp_cols + array_group_cols:
                    if col in group.columns:
                        agg[col] = group[col].iloc[unique_indices].tolist()
                for col in group_cols:
                    agg[col] = group[col].iloc[0]
                for col in other_cols:
                    agg[col] = group[col].iloc[0]
            else:
                agg = {col: group[col].tolist() for col in comp_cols + array_group_cols}
                for col in group_cols:
                    agg[col] = group[col].iloc[0]
                for col in other_cols:
                    agg[col] = group[col].iloc[0]
            output_rows.append(agg)

            # Mark all comp_ids in this group as seen
            seen_comp_ids.update(group['comp_id'].dropna().unique())
            # Mark this ref_id as seen
            seen_ref_ids.add(ref_id)

        # Ensure column order: ref columns, comp columns, array_group_cols, then other columns
        columns_order = group_cols + comp_cols + array_group_cols + other_cols
        result_df = pd.DataFrame(output_rows)
        columns_order = [col for col in columns_order if col in result_df.columns]
        result_df = result_df[columns_order]

        # Remove duplicate rows based on reference columns and other_cols (skip list columns)
        dedup_cols = group_cols + other_cols
        result_df = result_df.drop_duplicates(subset=dedup_cols, keep='first')

        return result_df
    
    def generate_lcs_based_gpkg_file(self, df: pd.DataFrame, gpkg_output_path: str):
        
        """
        Generate a single GPKG file with three layers:
        1. reference_points: Only id, name, category, and geom of ref_id.
        2. proximity_zones: Buffer zones around each ref_id using self.proximity_radius_m.
        3. duplicate_points: Only id, name, category, and geom of comp_id.
        
        Args:
            df (pd.DataFrame): DataFrame containing the results of the LCS similarity analysis.
            gpkg_output_path (str): Path to the output GPKG file.
        
        Returns:
            None: Exports the DataFrame to a GPKG file at the specified path.
        """
        
        print_info(f"Exporting LCS results to Markdown: {gpkg_output_path}")
        print_hashtags()
        
        if df is None or df.empty:
            print_warning(f"No LCS results to export for path: {gpkg_output_path}")
            return

        # --- 1. Reference Points Layer ---
        ref_points = []
        for _, row in df.iterrows():
            ref_geom = row['ref_geom']
            geom = None
            if isinstance(ref_geom, str):
                try:
                    geom = wkt.loads(ref_geom)
                except Exception:
                    try:
                        geom = wkb.loads(ref_geom, hex=True)
                    except Exception:
                        geom = None
            else:
                geom = ref_geom
            ref_points.append({
                "id": row['ref_id'],
                "name": row['ref_name'],
                "category": row['ref_category'],
                "geom": geom
            })
        ref_gdf = gpd.GeoDataFrame(ref_points, geometry="geom", crs="EPSG:4326")
        ref_gdf = ref_gdf[ref_gdf["geom"].notnull()]

        # --- 2. Proximity Zones Layer ---
        # Project to a metric CRS for accurate buffering
        ref_gdf_metric = ref_gdf.to_crs(epsg=3857)
        prox_zones = []
        for _, row in ref_gdf_metric.iterrows():
            if row.geom is not None:
                proximity_buffer_geom = row.geom.buffer(self.proximity_radius_m)
                prox_zones.append({
                    "id": row["id"],
                    "name": row["name"],
                    "category": row["category"],
                    "geom": proximity_buffer_geom
                })
        prox_gdf_metric = gpd.GeoDataFrame(prox_zones, geometry="geom", crs="EPSG:3857")
        prox_gdf_metric = prox_gdf_metric[prox_gdf_metric["geom"].notnull()]
        # Reproject back to WGS84 for export
        prox_gdf = prox_gdf_metric.to_crs(epsg=4326)

        # --- 3. Duplicate Points Layer ---
        dup_points = []
        for _, row in df.iterrows():
            comp_ids = row['comp_id'] if isinstance(row['comp_id'], list) else [row['comp_id']]
            comp_names = row['comp_name'] if isinstance(row['comp_name'], list) else [row['comp_name']]
            comp_categories = row['comp_category'] if isinstance(row['comp_category'], list) else [row['comp_category']]
            comp_geoms = row['comp_geom'] if isinstance(row['comp_geom'], list) else [row['comp_geom']]
            for cid, cname, ccat, cgeom in zip(comp_ids, comp_names, comp_categories, comp_geoms):
                geom = None
                if isinstance(cgeom, str):
                    try:
                        geom = wkt.loads(cgeom)
                    except Exception:
                        try:
                            geom = wkb.loads(cgeom, hex=True)
                        except Exception:
                            geom = None
                else:
                    geom = cgeom
                dup_points.append({
                    "id": cid,
                    "name": cname,
                    "category": ccat,
                    "geom": geom
                })
        dup_gdf = gpd.GeoDataFrame(dup_points, geometry="geom", crs="EPSG:4326")
        dup_gdf = dup_gdf[dup_gdf["geom"].notnull()]

        # --- Write all layers to a single GPKG file ---
        ref_gdf.to_file(gpkg_output_path, layer="reference_points", driver="GPKG")
        prox_gdf.to_file(gpkg_output_path, layer="proximity_zones", driver="GPKG")
        dup_gdf.to_file(gpkg_output_path, layer="duplicate_points", driver="GPKG")
    
    def generate_lcs_based_markdown_report(self, df: pd.DataFrame, md_output_path: str):

        """
        Exports the LCS results to a Markdown file.
        
        Args:
            df (pd.DataFrame): DataFrame containing the results of the LCS similarity analysis.
            md_output_path (str): Path to the output Markdown file.

        Returns:
            None: Exports the DataFrame to a Markdown file at the specified path.
        """

        print_info(f"Exporting LCS results to Markdown: {md_output_path}")
        print_hashtags()

        if df is None or df.empty:
            print_warning(f"No LCS results to export for path: {md_output_path}")
            return

        # Only keep the required columns
        cols = ['ref_id', 'ref_name', 'ref_category', 'comp_id', 'comp_name', 'comp_category', 'comp_geom', 'distance_m', 'overall_lcs']
        df = df[cols]

        # Flatten the DataFrame: one row per (ref_id, comp_id) pair
        rows = []
        for _, row in df.iterrows():
            ref_id = row['ref_id']
            ref_name = row['ref_name']
            ref_category = row['ref_category']
            comp_ids = row['comp_id'] if isinstance(row['comp_id'], list) else [row['comp_id']]
            comp_names = row['comp_name'] if isinstance(row['comp_name'], list) else [row['comp_name']]
            distances = row['distance_m'] if isinstance(row['distance_m'], list) else [row['distance_m']]
            overall_lcs = row['overall_lcs'] if isinstance(row['overall_lcs'], list) else [row['overall_lcs']]
            for cid, cname, dist, score in zip(comp_ids, comp_names, distances, overall_lcs):
                rows.append({
                    "Reference ID": ref_id,
                    "Reference Name": ref_name,
                    "Category": ref_category,
                    "Duplicate ID": cid,
                    "Duplicate Name": cname,
                    "Distance (m)": dist,
                    "Overall LCS": score
                })

        flat_df = pd.DataFrame(rows)

        # Set column widths so headers do not wrap
        headers = ["Reference ID", "Reference Name", "Category", "Duplicate ID", "Duplicate Name", "Distance (m)", "Overall LCS"]
        col_widths = [max(len(str(val)) for val in [header] + flat_df[header].astype(str).tolist()) for header in headers]

        def format_row_md(row, widths):
            return "| " + " | ".join(str(val).ljust(width) for val, width in zip(row, widths)) + " |\n"

        with open(md_output_path, "w") as f:
            f.write("# OSM POI Data Duplication Report\n\n")
            f.write("This report lists all detected duplicate POIs based on Lowest Common Subsequence (LCS) similarity method.\n\n")
            f.write("**LCS Parameters Used:**\n\n")
            f.write("- **Attributes of Interest:** Name, Category, Geometry\n")
            f.write(f"- **Proximity Radius (meters):** `{self.proximity_radius_m}`\n")
            f.write(f"- **Similarity Threshold (LCS):** `{self.threshold_lcs}`\n\n")
            f.write("## Summary of Duplicates\n\n")
            # Write header
            f.write(format_row_md(headers, col_widths))
            f.write("|" + "|".join("-" * (w + 2) for w in col_widths) + "|\n")
            # Write rows
            for _, row in flat_df.iterrows():
                f.write(format_row_md([row[h] for h in headers], col_widths))
            f.write("\n")
            
            
@timing
def validate_poi(region: str):
    """Main function to run the validation process for POI data for all configured metrics."""

    db_old = Database(settings.RAW_DATABASE_URI)
    db_new = Database(settings.LOCAL_DATABASE_URI)

    validator = PoiValidation(db_config=settings, region=region)
    all_metrics = validator.all_metrics

    if not all_metrics:
        print_error(f"No metrics defined in poi.yaml for region '{region}'. Please configure at least one metric under 'validation.metrics'.")
        return

    if region == 'europe':
        for loop_region in validator.config.regions:
            for current_metric in all_metrics:
                geom_reference_query, temp_ref_clone_name = validator.get_geom_reference_table_and_query(current_metric)
                temp_geom_clone_table = validator.create_temp_geom_reference_table(db_old, db_new, geom_reference_query, temp_ref_clone_name)
                process_poi_validation(validator, "poi", loop_region, current_metric, temp_geom_clone_table, temp_ref_clone_name, db_old, db_new)
                validator.drop_temp_geom_reference_table(db_new, temp_geom_clone_table)
    else:
        for current_metric in all_metrics:
            print_info(f"Performing calculations for metric '{current_metric}'")
            print_hashtags()
            geom_reference_query, temp_ref_clone_name = validator.get_geom_reference_table_and_query(current_metric)
            temp_geom_clone_table = validator.create_temp_geom_reference_table(db_old, db_new, geom_reference_query, temp_ref_clone_name)
            process_poi_validation(validator, "poi", region, current_metric, temp_geom_clone_table, temp_ref_clone_name, db_old, db_new)
            validator.drop_temp_geom_reference_table(db_new, temp_geom_clone_table)
            
    lcs_results_df = validator.run_lcs_analysis_sql(db_new)
    processed_lcs_df = validator.prepare_lcs_dataframe_for_export(lcs_results_df)

    lcs_gpkg_output_path = os.path.join(validator.data_dir, "poi", f"poi_validation_lcs_{region}.gpkg")
    lcs_md_output_path = os.path.join(validator.data_dir, "poi", f"poi_validation_lcs_{region}.md")

    validator.generate_lcs_based_gpkg_file(processed_lcs_df, lcs_gpkg_output_path)    
    validator.generate_lcs_based_markdown_report(processed_lcs_df, lcs_md_output_path)

    db_old.close()
    db_new.close()

def process_poi_validation(validator, dataset_type: str, region: str, metric: str, temp_geom_clone_table: str, temp_ref_clone_name: str, db_old, db_new):
    spatial_join_results = validator.run_metric_based_validation(db_old, db_new, temp_geom_clone_table, metric)
    unified_results = validator.compare_metric_based_new_and_old_results(spatial_join_results, metric)

    gpkg_output_path = os.path.join(validator.data_dir, dataset_type, f"{dataset_type}_validation_{metric}_{region}.gpkg")
    validator.generate_metrics_based_gpkg_file(unified_results, gpkg_output_path)

    md_output_path = os.path.join(validator.data_dir, dataset_type, f"{dataset_type}_validation_{metric}_{region}.md")
    validator.generate_metrics_based_markdown_report(unified_results, md_output_path, region=region, metric=metric, temp_ref_clone_name=temp_ref_clone_name)

if __name__ == "__main__":
    validate_poi()