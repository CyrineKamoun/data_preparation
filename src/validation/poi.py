import os
import re
import json
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
        self.raw_poi_tables = self.config.validation["raw_poi_table"]
        self.local_poi_table = self.config.validation["local_poi_table"]
        self.all_metrics = list(self.config.validation["metrics"].keys())
        self.default_metric = self.all_metrics[0] if self.all_metrics else "count"
        self.thresholds = self.config.validation["metrics"][self.default_metric]["thresholds"]
        self.geom_reference_query = self.config.validation["metrics"][self.default_metric]["geom_reference_query"]
        self.polygon_geom_table = self.geom_reference_query.split("FROM", 1)[1].strip().split()[0]

    def create_temp_geom_reference_table(self, db_raw: Database, db_local: Database, metric: str=None, temp_polygons_clone: str=None):
        """
        Create a temporary geometry reference table in the local database by copying data from the raw database.

        Args:
            db_raw (Database): Database connection to the raw database.
            db_local (Database): Database connection to the local database.
            metric (str, optional): The metric to use for validation. Defaults to None.
            temp_polygons_clone (str, optional): Name for the temporary polygons table. Defaults to None.

        Returns:
            str: The name of the temporary polygon geometry reference table created in the local database.
        """

        # Extract the polygon geomtry table name from the yaml definition inside validation object
        print_info(f"Creating clone of {self.polygon_geom_table} in local database")
        print_hashtags()
        
        # Determine which metric to use. If not provided, use the "count" metric.
        if metric is None:
            metric = self.default_metric

        # Copy the polygon table name using the geometry reference query to the temp_polygons_clone variable 
        if temp_polygons_clone is None:
            temp_polygons_clone = self.geom_reference_query.split("FROM")[1].strip().split()[0]

        # Run the geometry reference query and store all the results in the geom_data variable
        geom_data = db_raw.select(self.geom_reference_query)

        # If not run for the first time, the table gets dropped and recreated
        drop_sql = f"DROP TABLE IF EXISTS {temp_polygons_clone};"
        db_local.perform(drop_sql)

        # Create a temporary polygon geometry reference table clone in local DB
        create_sql = f"""
        CREATE TABLE {temp_polygons_clone} (
            id TEXT,
            name TEXT,
            geom geometry
        );
        """
        db_local.perform(create_sql)

        # Insert the data copied from the raw database into the local temporary polygon geometry reference table
        insert_sql = f"INSERT INTO {temp_polygons_clone} (id, name, geom) VALUES (%s, %s, %s);"
        for row in geom_data:
            db_local.perform(insert_sql, row)
    
        # It returns the name of the temporary polygon geometry reference table    
        return temp_polygons_clone
    
    def convert_geometry_query_columns_to_dict(self):
        """
        This function extracts the column names from the polygon geometry reference query and creates a mapping dictionary for running queries on local and raw database.
        
        Structure:
            {
                "local_column_alias": "raw_column_name"
            }
        
        Returns:
            tuple: A tuple containing:
                - local_and_raw_columns_mapping (dict): Mapping of local column aliases to raw column names.
                - local_group_by_clause (str): Group by clause for local query.
                - raw_group_by_clause (str): Group by clause for raw query.
        """
        
        # Extract the columns part from the polygon geometry reference query and split it into individual columns
        columns_part = self.geom_reference_query.split("SELECT")[1].split("FROM")[0].strip()
        columns = [col.strip() for col in columns_part.split(",")]

        # Build dictionary: {alias for local query: original_column for raw query}
        local_and_raw_columns_mapping = {}
        for col in columns:
            if " AS " in col:
                original, alias = [x.strip() for x in col.split(" AS ")]
                local_and_raw_columns_mapping[alias] = original
            else:
                local_and_raw_columns_mapping[col] = col

        # Generate group by clauses for local and raw queries otherwise it will not work
        local_group_by_clause = ", ".join(f"poly.{k}" for k in local_and_raw_columns_mapping.keys())
        raw_group_by_clause = ", ".join(f"poly.{v}" for v in local_and_raw_columns_mapping.values())
        
        return local_and_raw_columns_mapping, local_group_by_clause, raw_group_by_clause
        
    def perform_spatial_intersection(self, db, poi_table, temp_polygons_clone, group_by_clause, metric=None):
        
        """
        Perform spatial intersection between POI table and polygons table.
        
        Args:
            db (Database): Database connection to the local or raw database.
            poi_table (str): Name of the POI table to perform intersection with.
            temp_polygons_clone (str): Name of the temporary polygons clone table.
            group_by_clause (str): Group by clause for the spatial join query (local or raw).
            metric (str, optional): The metric to use for validation. Defaults to None.
        
        Returns:
            list: Results of the spatial intersection query.
        """
        
        # Determine which metric to use. If not provided, use the "count" metric.
        print_info(f"Performing spatial intersection between {poi_table} and {temp_polygons_clone}")
        print_hashtags()

        # Generate the spatial join query using the group by clause and poi_table for local or raw database
        spatial_join_query = f"""
            SELECT {group_by_clause}, poi.category, COUNT(poi.category) AS count
            FROM {temp_polygons_clone} AS poly JOIN {poi_table} AS poi
            ON ST_Within(poi.geom, poly.geom)
            WHERE poi.category IS NOT NULL
            GROUP BY {group_by_clause}, poi.category
            ORDER BY poi.category;
        """
        spatial_join_results = db.select(spatial_join_query)
        
        return spatial_join_results

    def run_core_validation_functions(self, db_raw, db_local, metric=None):
        
        """
        Run the core validation functions to create merged results for local and raw POI tables.
        
        Args:
            db_raw (Database): Database connection to the raw database.
            db_local (Database): Database connection to the local database.
            metric (str, optional): The metric to use for validation. Defaults to None.
            
        Returns:
            dict: A dictionary containing spatial join results for local and raw POI tables.
        """
        
        print_info(f"Creating merged results for local and raw POI tables")
        print_hashtags()
        
        # Initiate the spatial join results dictionary to hold  local and raw join results
        spatial_join_results = {
            "local": {},
            "raw": {}
        }
        
        # Step 1: Run the actual function to create the polygon geometry reference table in local database
        temp_geom_clone_table = self.create_temp_geom_reference_table(db_raw, db_local, metric)
        # Step 2: Convert the geometry query columns to a mapping dictionary
        local_and_raw_columns_mapping, local_group_by_clause, raw_group_by_clause = self.convert_geometry_query_columns_to_dict()
        
        # Step 3: Prepare they keys for mapping. These keys are appended in the final results for export 
        local_keys = [f"local_{k}" for k in local_and_raw_columns_mapping.keys()] + ["local_category", "new_count"]
        raw_keys = [f"raw_{v}" for v in local_and_raw_columns_mapping.values()] + ["raw_category", "old_count"]

        # Step 4: Perform spatial intersection between POI table and polygons table for local database
        local_results_raw = self.perform_spatial_intersection(
            db_local, self.local_poi_table, temp_geom_clone_table, local_group_by_clause, metric
        )
        
        # Step 5: Map the local results to a list of dictionaries with keys
        local_results = [
            dict(zip(local_keys, row))
            for row in local_results_raw
        ]
        
        # Step 6: Store the local results in the spatial join results dictionary
        spatial_join_results["local"][self.local_poi_table] = local_results

        # Step 7: Perform spatial intersection for each raw POI table in iteration with polygons table and store results in the raw key of spatial join results
        for raw_poi_table in self.raw_poi_tables:
            raw_results_raw = self.perform_spatial_intersection(
                db_raw, raw_poi_table, temp_geom_clone_table, raw_group_by_clause, metric
            )
            raw_results = [
                dict(zip(raw_keys, row))
                for row in raw_results_raw
            ]
            spatial_join_results["raw"][raw_poi_table] = raw_results
            
        return spatial_join_results

    def compare_local_and_raw_results(self, spatial_join_results):
        
        """
        Compare local and raw results based on matching id, name, geom, and category.
        
        Args:
            spatial_join_results (dict): Dictionary containing spatial join results for local and raw POI tables.
        
        Returns:
            dict: A filtered dictionary containing unified results with comparison of local and raw records.
        """
        
        print_info(f"Filtering records based on matching id, name, geom, and category")
        print_hashtags()
        
        # Initialize the unified results dictionary to hold the comparison results
        unified_results = {}

        # Get the local results (should be a list of dicts) from the spatial join results
        local_records = spatial_join_results["local"].get(self.local_poi_table, [])

        # Dynamically get the column keys from the key mapping dictionary for local and raw tables
        local_and_raw_columns_mapping, _, _ = self.convert_geometry_query_columns_to_dict()
        local_keys = [f"local_{k}" for k in local_and_raw_columns_mapping.keys()]
        # Category is a special case, so we add it separately since it is not in the mapping and reference query
        local_category_key = "local_category"

        # Build a lookup for local records: (id, name, geom, category) -> record
        local_lookup = {}
        for rec in local_records:
            # Build the key tuple dynamically (all mapped keys + category)
            key = tuple(rec.get(k) for k in local_keys) + (rec.get(local_category_key),)
            local_lookup[key] = rec

        # Define the raw keys based on the mapping dictionary to be matched with the local keys for filtering
        raw_keys = [f"raw_{v}" for v in local_and_raw_columns_mapping.values()]
        raw_category_key = "raw_category"

        # This loop runs over the raw results and then matches them with the local records
        for raw_table, raw_records in spatial_join_results["raw"].items():
            comparison_list = []
            # For each raw record, create a key and check if it exists in the local lookup            
            for raw_rec in raw_records:
                key = tuple(raw_rec.get(k) for k in raw_keys) + (raw_rec.get(raw_category_key),)
                local_rec = local_lookup.get(key)
                # If a matching local record is found, calculate the percentage difference
                if local_rec:
                    new_count = local_rec.get("new_count", 0)
                    old_count = raw_rec.get("old_count", 0)
                    try:
                        # Calculate percentage difference: ((local - raw) / raw) * 100
                        if float(old_count) != 0:
                            difference = round(((float(new_count) - float(old_count)) / float(old_count)) * 100, 2)
                        else:
                            difference = None  # or set to 0 or 100 if you want to handle zero division differently
                    except Exception:
                        difference = None
                    
                    # Prepare the output that contains the comparison results and filters out duplicate column names either from local or raw columns
                    #  id, name, category, geom come from either local or raw records
                    # new_count and old_count are the counts from local and raw records respectively
                    # perc_diff is the calculated percentage difference
                    output = {
                        "id": local_rec.get(local_keys[0]),
                        "name": local_rec.get(local_keys[1]),
                        "category": local_rec.get(local_category_key),
                        "new_count": new_count,
                        "old_count": old_count,
                        "perc_diff": difference,
                        "geom": local_rec.get(local_keys[2]),
                    }
                    comparison_list.append(output)
            unified_results[raw_table] = comparison_list

        return unified_results

    def export_records_to_gpkg(self, records, output_path):
        
        """
        Export the validation results to a GPKG file, creating separate layers for each category.
        
        Args:
            records (dict): Dictionary containing the validation results.
            output_path (str): Path to the output GPKG file.
        """
        
        print_info(f"Exporting to GPKG: {output_path}")
        print_hashtags()
        
        for raw_table, recs in records.items():
            if not recs:
                continue
            
            # Dynamically get the geometry key (find the key that contains 'geom')
            geom_key = next((k for k in recs[0].keys() if "geom" in k), None)
            category_key = next((k for k in recs[0].keys() if "category" in k), None)
            if not geom_key or not category_key:
                continue

            # Prepare the dataframe to hold the records for export
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
                table_name = raw_table.split(".", 1)[1]
                layer_name = f"{table_name}-{str(category)}"
                gdf = gpd.GeoDataFrame(group.drop(columns=[geom_key]), geometry="geometry", crs="EPSG:4326")
                gdf.to_file(output_path, layer=layer_name, driver="GPKG")

    def generate_markdown_report(self, records, output_path, region, metric):
        
        """
        Generate a Markdown report summarizing the validation results.
        
        Args:
            records (dict): Dictionary containing the validation results.
            output_path (str): Path to the output Markdown file.
            region (str): The region for which the report is generated.
            metric (str, optional): The metric to use for validation. Defaults to None.
        """
        
        print_info(f"Generating Report: {output_path}")
        print_hashtags()

        # Setup configuration
        local_table = self.local_poi_table
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

            for raw_table, recs in records.items():
                if not recs:
                    continue

                # Section Header Info
                f.write(f"### 📄 Table Details\n\n")
                f.write(f"- **Polygon Reference Table:** `{self.polygon_geom_table}`\n")
                f.write(f"- **Raw Database Table:** `{raw_table}`\n")
                f.write(f"- **Local Database Table:** `{local_table}`\n")
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
                    raw_val = float(rec.get("old_count", 0))
                    local_val = float(rec.get("new_count", 0))
                    diff = rec.get("perc_diff", 0)
                    threshold = thresholds.get(str(rec["category"]).lower(), thresholds.get("default", 0))
                    region_id = rec.get("id", "")

                    # Check if the difference exceeds the relevant threshold then append the record to violations
                    is_violation = diff is not None and threshold is not None and abs(diff) > float(threshold)
                    if is_violation:
                        violations.append([
                            category, county, f"{raw_val:.2f}", f"{local_val:.2f}",
                            f"{diff:.2f}%", f"{float(threshold):.2f}%", region_id
                        ])

                    # This snippet updates the summary statistics for each category
                    cat_key = str(rec["category"]).lower()
                    summary_stats.setdefault(cat_key, {
                        "violations": 0, "raw_total": 0, "local_total": 0, "diffs": []
                    })
                    if is_violation:
                        summary_stats[cat_key]["violations"] += 1
                    summary_stats[cat_key]["raw_total"] += raw_val
                    summary_stats[cat_key]["local_total"] += local_val
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
                    ["Records Analyzed", len(recs)],
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
                    "Category", "Violations", "Raw Total", "Local Total", "Avg Diff (%)", "Max Diff (%)"
                ]
                breakdown_rows = []
                for cat, stats in summary_stats.items():
                    avg_diff = sum(stats["diffs"]) / len(stats["diffs"]) if stats["diffs"] else 0
                    max_diff = max(stats["diffs"]) if stats["diffs"] else 0
                    breakdown_rows.append([
                        cat,
                        stats["violations"],
                        f"{stats['raw_total']:.2f}",
                        f"{stats['local_total']:.2f}",
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
    
    db_raw = Database(settings.RAW_DATABASE_URI)
    db_local = Database(settings.LOCAL_DATABASE_URI)
    
    spatial_join_results = validator.run_core_validation_functions(db_raw, db_local, metric)
    unified_results = validator.compare_local_and_raw_results(spatial_join_results)
    gpkg_output_path = os.path.join(validator.data_dir, dataset_type, f"{dataset_type}_validation_{region}.gpkg")
    validator.export_records_to_gpkg(unified_results, gpkg_output_path)
    md_output_path = os.path.join(validator.data_dir, dataset_type, f"{dataset_type}_validation_{region}.md")
    validator.generate_markdown_report(unified_results, md_output_path, region=region, metric=metric)

if __name__ == "__main__":
    validate_poi()