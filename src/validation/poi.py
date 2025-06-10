import os
import re
import json
import subprocess
import time
import pandas as pd
import geopandas as gpd
from shapely import wkb, wkt

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
        self.default_metric = self.all_metrics[0] if self.all_metrics else "count"
        self.thresholds = self.config.validation["metrics"][self.default_metric]["thresholds"]
        self.geom_reference_query = self.config.validation["metrics"][self.default_metric]["geom_reference_query"]
        self.polygon_geom_table = self.geom_reference_query.split("FROM", 1)[1].strip().split()[0]

    def create_temp_geom_reference_table(self, db_old: Database, db_new: Database, temp_polygons_clone: str=None):
        """
        Copy polygons geometry reference table from old to new using ogr2ogr.

        Args:
            db_old (Database): Database connection to the old database.
            db_new (Database): Database connection to the new database.
            metric (str, optional): The metric to use for validation. Defaults to None.
            temp_polygons_clone (str, optional): Name for the temporary polygons table. Defaults to None.

        Returns:
            str: The name of the temporary polygon geometry reference table created in the new database.
        """

        # Extract the polygon geomtry table name from the yaml definition inside validation object
        print_info(f"Creating clone of {self.polygon_geom_table} in new database")
        print_hashtags()
        
        # Copy the polygon table name using the geometry reference query to the temp_polygons_clone variable 
        if temp_polygons_clone is None:
            temp_polygons_clone = self.geom_reference_query.split("FROM")[1].strip().split()[0]
    
        # Extract the new and old db credentials to parse to ogr2ogr command
        new_host = settings.LOCAL_DATABASE_URI.host
        new_db = settings.LOCAL_DATABASE_URI.path.replace("/", "")
        new_user = settings.LOCAL_DATABASE_URI.user
        new_password = settings.LOCAL_DATABASE_URI.password
        new_port = settings.LOCAL_DATABASE_URI.port
        
        # Extract the old db credentials to parse to ogr2ogr command
        old_host = settings.RAW_DATABASE_URI.host
        old_db = settings.RAW_DATABASE_URI.path.replace("/", "")
        old_user = settings.RAW_DATABASE_URI.user
        old_password = settings.RAW_DATABASE_URI.password
        old_port = settings.RAW_DATABASE_URI.port
        
        # Run the ogr2ogr command to copy the polygon geometry reference table from old to new database
        ogr2ogr_command = (
            f"ogr2ogr -f 'PostgreSQL' "
            f"PG:'host={new_host} dbname={new_db} user={new_user} password={new_password} port={new_port}' "
            f"PG:'host={old_host} dbname={old_db} user={old_user} password={old_password} port={old_port}' "
            f"-nln {temp_polygons_clone} -sql \"{self.geom_reference_query}\""
        )
    
        try:
            subprocess.run(ogr2ogr_command, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print_error(f"Polygons data table copy failed: {e}")
    
        # It returns the name of the temporary polygon geometry reference table    
        return temp_polygons_clone
    
    def drop_temp_geom_reference_table(self, db: Database, temp_polygons_clone: str):
        """
        Drop the temporary polygons geometry reference table from the database.

        Args:
            db (Database): Database connection to the database.
            temp_polygons_clone (str): Name of the temporary polygons table to drop.
        """
        
        print_info(f"Dropping temporary polygons geometry reference table {temp_polygons_clone}")
        print_hashtags()
        
        # Execute the drop table query
        drop_query = f"DROP TABLE IF EXISTS {temp_polygons_clone};"
        db.perform(drop_query)
    
    def convert_geometry_query_columns_to_dict(self):
        """
        This function extracts the column names from the polygon geometry reference query and creates a mapping dictionary for running queries on new and old database.
        
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
        
        # Extract the columns part from the polygon geometry reference query and split it into individual columns
        columns_part = self.geom_reference_query.split("SELECT")[1].split("FROM")[0].strip()
        columns = [col.strip() for col in columns_part.split(",")]

        # Build dictionary: {alias for new query: original_column for old query}
        new_and_old_columns_mapping = {}
        for col in columns:
            if " AS " in col:
                original, alias = [x.strip() for x in col.split(" AS ")]
                new_and_old_columns_mapping[alias] = original
            else:
                new_and_old_columns_mapping[col] = col

        # Generate group by clauses for new and old queries otherwise it will not work
        new_group_by_clause = ", ".join(f"poly.{k}" for k in new_and_old_columns_mapping.keys())
        old_group_by_clause = ", ".join(f"poly.{v}" for v in new_and_old_columns_mapping.values())
        
        return new_and_old_columns_mapping, new_group_by_clause, old_group_by_clause
        
    def perform_spatial_intersection(self, db, poi_table, temp_polygons_clone, group_by_clause, metric=None):
        
        """
        Perform spatial intersection between POI table and polygons table.
        
        Args:
            db (Database): Database connection to the new or old database.
            poi_table (str): Name of the POI table to perform intersection with.
            temp_polygons_clone (str): Name of the temporary polygons clone table.
            group_by_clause (str): Group by clause for the spatial join query (new or old).
            metric (str, optional): The metric to use for validation. Defaults to None.
        
        Returns:
            list: Results of the spatial intersection query.
        """
        
        # Determine which metric to use. If not provided, use the "count" metric.
        print_info(f"Performing spatial intersection between {poi_table} and {temp_polygons_clone}")
        print_hashtags()

        # Generate the spatial join query using the group by clause and poi_table for new or old database
        # spatial_join_query = f"""
        #     SELECT {group_by_clause}, poi.category, COUNT(poi.category) AS count
        #     FROM {temp_polygons_clone} AS poly JOIN {poi_table} AS poi
        #     ON ST_Within(poi.geom, poly.geom)
        #     WHERE poi.category IS NOT NULL
        #     GROUP BY {group_by_clause}, poi.category
        #     ORDER BY poi.category;
        # """
        
        spatial_join_query = f"""
            SELECT {group_by_clause}, poi.category, COUNT(*) AS count
            FROM {temp_polygons_clone} AS poly
            CROSS JOIN LATERAL (
                SELECT category
                FROM {poi_table} AS poi
                WHERE poi.category IS NOT NULL
                AND ST_Within(poi.geom, poly.geom)
            ) AS poi
            GROUP BY {group_by_clause}, poi.category
            ORDER BY poi.category;
        """
        spatial_join_results = db.select(spatial_join_query)
        
        return spatial_join_results

    def run_core_validation_functions(self, db_old, db_new, metric=None):
        
        """
        Run the core validation functions to create merged results for new and old POI tables.
        
        Args:
            db_old (Database): Database connection to the old database.
            db_new (Database): Database connection to the new database.
            metric (str, optional): The metric to use for validation. Defaults to None.
            
        Returns:
            dict: A dictionary containing spatial join results for new and old POI tables.
        """
        
        print_info(f"Creating merged results for new and old POI tables")
        print_hashtags()
        
        # Initiate the spatial join results dictionary to hold  new and old join results
        spatial_join_results = {
            "new": {},
            "old": {}
        }
        
        # Step 1: Run the actual function to create the polygon geometry reference table in new database
        temp_geom_clone_table = self.create_temp_geom_reference_table(db_old, db_new, metric)
        # Step 2: Convert the geometry query columns to a mapping dictionary
        new_and_old_columns_mapping, new_group_by_clause, old_group_by_clause = self.convert_geometry_query_columns_to_dict()
        
        # Step 3: Prepare they keys for mapping. These keys are appended in the final results for export 
        new_keys = [f"new_{k}" for k in new_and_old_columns_mapping.keys()] + ["new_category", "new_count"]
        old_keys = [f"old_{v}" for v in new_and_old_columns_mapping.values()] + ["old_category", "old_count"]

        # Step 4: Perform spatial intersection between POI table and polygons table for new database
        new_results_old = self.perform_spatial_intersection(
            db_new, self.new_poi_table, temp_geom_clone_table, new_group_by_clause, metric
        )
        
        # Step 5: Map the new results to a list of dictionaries with keys
        new_results = [
            dict(zip(new_keys, row))
            for row in new_results_old
        ]
        
        # Step 6: Store the new results in the spatial join results dictionary
        spatial_join_results["new"][self.new_poi_table] = new_results

        # Step 7: Perform spatial intersection for each old POI table in iteration with polygons table and store results in the old key of spatial join results
        for old_poi_table in self.old_poi_tables:
            old_results_old = self.perform_spatial_intersection(
                db_old, old_poi_table, temp_geom_clone_table, old_group_by_clause, metric
            )
            old_results = [
                dict(zip(old_keys, row))
                for row in old_results_old
            ]
            spatial_join_results["old"][old_poi_table] = old_results
            
        return spatial_join_results

    def compare_new_and_old_results(self, spatial_join_results):
        
        """
        Compare new and old results based on matching id, name, geom, and category.
        
        Args:
            spatial_join_results (dict): Dictionary containing spatial join results for new and old POI tables.
        
        Returns:
            dict: A filtered dictionary containing unified results with comparison of new and old pois.
        """
        
        print_info(f"Filtering pois based on matching id, name, geom, and category")
        print_hashtags()
        
        # Initialize the unified results dictionary to hold the comparison results
        unified_results = {}

        # Get the new results (should be a list of dicts) from the spatial join results
        new_pois = spatial_join_results["new"].get(self.new_poi_table, [])

        # Dynamically get the column keys from the key mapping dictionary for new and old tables
        new_and_old_columns_mapping, _, _ = self.convert_geometry_query_columns_to_dict()
        new_keys = [f"new_{k}" for k in new_and_old_columns_mapping.keys()]
        # Category is a special case, so we add it separately since it is not in the mapping and reference query
        new_category_key = "new_category"

        # Build a lookup for new pois: (id, name, geom, category) -> record
        new_lookup = {}
        for rec in new_pois:
            # Build the key tuple dynamically (all mapped keys + category)
            key = tuple(rec.get(k) for k in new_keys) + (rec.get(new_category_key),)
            new_lookup[key] = rec

        # Define the old keys based on the mapping dictionary to be matched with the new keys for filtering
        old_keys = [f"old_{v}" for v in new_and_old_columns_mapping.values()]
        old_category_key = "old_category"

        # This loop runs over the old results and then matches them with the new pois
        for old_table, old_pois in spatial_join_results["old"].items():
            comparison_list = []
            # For each old record, create a key and check if it exists in the new lookup            
            for old_rec in old_pois:
                key = tuple(old_rec.get(k) for k in old_keys) + (old_rec.get(old_category_key),)
                new_rec = new_lookup.get(key)
                # If a matching new record is found, calculate the percentage difference
                if new_rec:
                    new_count = new_rec.get("new_count", 0)
                    old_count = old_rec.get("old_count", 0)
                    try:
                        # Calculate percentage difference: ((new - old) / old) * 100
                        if float(old_count) != 0:
                            difference = round(((float(new_count) - float(old_count)) / float(old_count)) * 100, 2)
                        else:
                            difference = None  # or set to 0 or 100 if you want to handle zero division differently
                    except Exception:
                        difference = None
                    
                    # Prepare the output that contains the comparison results and filters out duplicate column names either from new or old columns
                    #  id, name, category, geom come from either new or old pois
                    # new_count and old_count are the counts from new and old pois respectively
                    # perc_diff is the calculated percentage difference
                    output = {
                        "id": new_rec.get(new_keys[0]),
                        "name": new_rec.get(new_keys[1]),
                        "category": new_rec.get(new_category_key),
                        "new_count": new_count,
                        "old_count": old_count,
                        "perc_diff": difference,
                        "geom": new_rec.get(new_keys[2]),
                    }
                    comparison_list.append(output)
            unified_results[old_table] = comparison_list

        return unified_results

    def export_pois_to_gpkg(self, pois, output_path):
        
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

            # Export each category as a separate layer inside the GPKG file
            for category, group in df.groupby(category_key):
                table_name = old_table.split(".", 1)[1]
                layer_name = f"{table_name}-{str(category)}"
                gdf = gpd.GeoDataFrame(group.drop(columns=[geom_key]), geometry="geometry", crs="EPSG:4326")
                gdf.to_file(output_path, layer=layer_name, driver="GPKG")

    def generate_markdown_report(self, pois, output_path, region, metric):
        
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
        metric = metric or self.default_metric
        metric_config = self.config.validation["metrics"][metric]
        thresholds = metric_config.get("thresholds", {})
        units = "Number of Features"

        # A function definition to format rows for Markdown tables
        def format_row(row, widths):
            """Format a row for Markdown with padded column widths."""
            return "| " + " | ".join(str(val).ljust(width) for val, width in zip(row, widths)) + " |\n"

        # Create the Markdown file and write the header
        with open(output_path, 'w') as f:
            f.write(
                """<style>
                @import url('https://fonts.googleapis.com/css2?family=Inter&display=swap');
                body {
                    font-family: 'Inter', sans-serif;
                }
                </style>\n\n"""
            )
            f.write(f"# Validation Report - **{metric.upper()}** Metric\n\n")

            for old_table, recs in pois.items():
                if not recs:
                    continue

                # Section Header Info
                f.write(f"### 📄 Table Details\n\n")
                f.write(f"- **Polygon Reference Table:** `{self.polygon_geom_table}`\n")
                f.write(f"- **Raw Database Table:** `{old_table}`\n")
                f.write(f"- **Local Database Table:** `{new_table}`\n")
                f.write(f"- **Region:** `{region}`\n")
                f.write(f"- **Metric:** `{metric}`\n")
                f.write(f"- **Units:** `{units}`\n\n")

                # 🚨 Threshold Violations Table
                f.write("## 🚨 Threshold Violations\n\n")

                headers = [
                    "Category", "County", "Old Count", "New Count",
                    "Difference (%)", "Threshold (%)", "Region ID"
                ]
                violations = []
                summary_stats = {}

                for rec in recs:
                    category = str(rec["category"]).capitalize()
                    county = rec.get("name", "")
                    old_val = float(rec.get("old_count", 0))
                    new_val = float(rec.get("new_count", 0))
                    diff = rec.get("perc_diff", 0)
                    threshold = thresholds.get(str(rec["category"]).lower(), thresholds.get("default", 0))
                    region_id = rec.get("id", "")

                    # Check if the difference exceeds the relevant threshold then append the record to violations
                    is_violation = diff is not None and threshold is not None and abs(diff) > float(threshold)
                    if is_violation:
                        violations.append([
                            category, county, f"{old_val:.2f}", f"{new_val:.2f}",
                            f"{diff:.2f}%", f"{float(threshold):.2f}%", region_id
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
                    if diff is not None:
                        summary_stats[cat_key]["diffs"].append(abs(diff))

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
                    ["Regions Analyzed", len(set(rec.get("id") for rec in recs))],
                    ["POIs Analyzed", len(recs)],
                    ["Violations Found", len(violations)],
                    ["Violation Rate", f"{(len(violations) / len(recs) * 100):.2f}%" if recs else "0.00%"]
                ]
                stats_col_widths = [max(len(str(row[i])) for row in [stats_headers] + stats_rows) for i in range(2)]
                f.write(format_row(stats_headers, stats_col_widths))
                f.write("|" + "|".join("-" * (w + 2) for w in stats_col_widths) + "|\n")
                for row in stats_rows:
                    f.write(format_row(row, stats_col_widths))
                f.write("\n")

                # 📋 Violation Breakdown by Category Section
                f.write("## 📋 Violation Breakdown by Category\n\n")
                breakdown_headers = [
                    "Category", "Violations", "Old Total", "New Total", "Avg Diff (%)", "Max Diff (%)"
                ]
                breakdown_rows = []
                for cat, stats in summary_stats.items():
                    avg_diff = sum(stats["diffs"]) / len(stats["diffs"]) if stats["diffs"] else 0
                    max_diff = max(stats["diffs"]) if stats["diffs"] else 0
                    breakdown_rows.append([
                        cat,
                        stats["violations"],
                        f"{stats['old_total']:.2f}",
                        f"{stats['new_total']:.2f}",
                        f"{avg_diff:.2f}",
                        f"{max_diff:.2f}"
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

@timing
def validate_poi(region: str, metric: str = None):
    """Main function to run the validation process for POI data."""
    if region == 'europe':
        for loop_region in Config("poi", region).regions:
            process_poi_validation("poi", loop_region, metric)
    else:
        process_poi_validation("poi", region, metric)
        
def process_poi_validation(dataset_type: str, region: str, metric: str = None):
    """Process POI validation for single or all metrics."""
    validator = PoiValidation(db_config=settings, region=region)
    
    db_old = Database(settings.RAW_DATABASE_URI)
    db_new = Database(settings.LOCAL_DATABASE_URI)
     
    spatial_join_results = validator.run_core_validation_functions(db_old, db_new, metric)
    unified_results = validator.compare_new_and_old_results(spatial_join_results)
    
    gpkg_output_path = os.path.join(validator.data_dir, dataset_type, f"{dataset_type}_validation_{region}.gpkg")
    validator.export_pois_to_gpkg(unified_results, gpkg_output_path)
    
    md_output_path = os.path.join(validator.data_dir, dataset_type, f"{dataset_type}_validation_{region}.md")
    validator.generate_markdown_report(unified_results, md_output_path, region=region, metric=metric)
    
    validator.drop_temp_geom_reference_table(db_new, validator.polygon_geom_table)


if __name__ == "__main__":
    validate_poi()