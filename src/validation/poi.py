import json
import math
import os
import subprocess

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import wkb, wkt

from src.config.config import Config
from src.core.config import settings
from src.db.db import Database
from src.utils.utils import (
    print_error,
    print_hashtags,
    print_info,
    print_warning,
    timing,
)


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

        #self.lcs_config = self.config.validation["lcs"]
        #self.poi_columns = self.lcs_config["poi_columns"]
        #self.excluded_categories = self.lcs_config["excluded_categories"]
        #self.proximity_radius_m = self.lcs_config["search_radius_m"]
        #self.threshold_lcs = self.lcs_config["threshold_lcs"]

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
                    WHEN ref.population > 0 THEN
                        ROUND((CAST(COUNT(*) AS NUMERIC) / ref.population::NUMERIC) * {people_count}::NUMERIC, 2)::NUMERIC
                    ELSE 0::NUMERIC
                END AS poi_per_people
            """
        elif metric_name == "population_per_poi":
            return f"""
                CASE
                    WHEN COUNT(*) > 0 THEN
                        (FLOOR(ref.population::NUMERIC / CAST(COUNT(*) AS NUMERIC))::NUMERIC)::INT
                    ELSE NULL
                END AS population_per_poi
            """
        else:
            raise ValueError(f"Unknown metric: {metric_name}")

    def create_temp_geom_reference_table(self, db_old: Database, db_new: Database, metric_name: str):
        """
        Copy the geometry reference table from raw database to local database using ogr2ogr.
        """

        # Extract the reference geomtry table name from the yaml definition inside validation object
        print_info(f"Creating spatial reference table in new database")
        print_hashtags()

        ogr2ogr_command = (
            f"ogr2ogr -f 'PostgreSQL' "
            f"PG:'host={db_new.db_config.host} dbname={db_new.db_config.path.replace('/', '')} user={db_new.db_config.user} password={db_new.db_config.password} port={db_new.db_config.port}' "
            f"PG:'host={db_old.db_config.host} dbname={db_old.db_config.path.replace('/', '')} user={db_old.db_config.user} password={db_old.db_config.password} port={db_old.db_config.port}' "
            f'-nln spatial_reference_table -overwrite -sql "{self.config.validation["metrics"][metric_name]["geom_reference_query"]}"'
        )
        try:
            subprocess.run(ogr2ogr_command, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print_error(f"Reference geometry data table copy failed: {e}")

    def get_categories_from_old_tables(self, db_old):
        """
        Get distinct categories from all old POI tables.

        Returns:
            dict: Dictionary mapping old_poi_table to list of categories
        """
        
        print_info("Getting distinct categories from old POI tables")
        print_hashtags()
        
        old_table_categories = {}
        
        for old_poi_table in self.old_poi_tables:
            try:
                categories_query = f"SELECT DISTINCT category FROM {old_poi_table} WHERE category IS NOT NULL ORDER BY category"
                categories_result = db_old.select(categories_query)
                categories_list = [row[0] for row in categories_result if row[0]]
                old_table_categories[old_poi_table] = categories_list
            except Exception as e:
                print_warning(f"Error getting categories from {old_poi_table}: {e}")
                old_table_categories[old_poi_table] = []
        
        return old_table_categories

    def perform_spatial_intersection(self, db, poi_table, spatial_table_query, metric_clause, categories_filter=None):
        """
        Perform a spatial intersection between the POI table and the temporary reference clone.
        Returns:
            list: Results of the spatial join as a list of dictionaries.
        """

        print_info(f"Spatially joining '{poi_table}' with spatial layer")
        print_hashtags()

        spatial_join_condition = "ST_Within(poi.geom, ref.geom)"
        
        # Build the category filter clause
        if categories_filter:
            categories_clause = ", ".join([f"'{cat}'" for cat in categories_filter])
            category_filter = f" poi.category IN ({categories_clause})"
        else:
            category_filter = ""

        spatial_join_query = f"""SELECT ref.id, ref.name,ref.geom, ref.population, poi.category, {metric_clause} FROM ({spatial_table_query}) AS ref INNER JOIN {poi_table} AS poi ON {spatial_join_condition} WHERE {category_filter} GROUP BY ref.id, ref.name,ref.geom, ref.population , poi.category ORDER BY poi.category;"""

        spatial_join_results = db.select(spatial_join_query)
        return spatial_join_results

    def process_old_new_poi_table_data(self, db_old, db_new, old_poi_table, categories_for_this_table,metric_name):
        """Process spatial intersection for both new and old POI tables."""
        keys=['id','name','geom','population','category',metric_name]
        spatial_table_query="SELECT * FROM spatial_reference_table"
        metric_clause = self.get_metric_clause(metric_name)
        # Perform spatial intersection for NEW POI table using categories from current old table
        new_results_raw = self.perform_spatial_intersection( db_new, self.new_poi_table, spatial_table_query, metric_clause, categories_filter=categories_for_this_table )
        # Convert new_results_raw to a DataFrame
        new_results =  [dict(zip(keys, row)) for row in new_results_raw]

        # Perform spatial intersection for current OLD POI table using its own categories
        spatial_table_query=self.config.validation["metrics"][metric_name]["geom_reference_query"]
        old_results_raw = self.perform_spatial_intersection(db_old, old_poi_table, spatial_table_query, metric_clause, categories_filter=categories_for_this_table)
        old_results =  [dict(zip(keys, row)) for row in old_results_raw]
        
        return new_results, old_results

    def format_float(self, val):
        if isinstance(val, float):
            return float(
                f"{val:.5f}".rstrip("0").rstrip(".")
                if "." in f"{val:.5f}"
                else f"{val:.5f}"
            )
        return val

    def load_metric_based_validation_thresholds(self, metric):
        """Load validation thresholds from configuration."""
        try:
            metric_config = self.config.validation["metrics"][metric]
            thresholds = metric_config["thresholds"]
            
            return {
                'abs_positive': thresholds["absolute"]["positive"],
                'abs_negative': thresholds["absolute"]["negative"],
                'relative_thresholds': thresholds["relative"]
            }
        except KeyError as e:
            print_info(f"Error: Could not load configuration for metric '{metric}'. Details: {e}")
            return None

    def apply_flagging_logic(self, difference, perc_diff, old_val_float, category, thresholds):
        """Apply flagging logic based on thresholds."""
        epsilon_for_zero_count = 1e-9
        abs_positive = thresholds['abs_positive']
        abs_negative = thresholds['abs_negative']
        rel_negative, rel_positive = thresholds['relative_thresholds'].get(category, (-0.3, 0.3))

        is_flagged = False
        violation_type = "No Violation"
        violated_threshold_value = "N/A"
        
        # Check if BOTH absolute AND relative positive thresholds are crossed
        if (difference > abs_positive and old_val_float > epsilon_for_zero_count and perc_diff > rel_positive):
            is_flagged = True
            violation_type = "Both Absolute & Percentage Increase"
            violated_threshold_value = f"Abs: {self.format_float(abs_positive)}, Rel: {self.format_float(rel_positive * 100)}%"
        
        # Check if BOTH absolute AND relative negative thresholds are crossed
        elif (difference < abs_negative and old_val_float > epsilon_for_zero_count and perc_diff < rel_negative):
            is_flagged = True
            violation_type = "Both Absolute & Percentage Decrease"
            violated_threshold_value = f"Abs: {self.format_float(abs_negative)}, Rel: {self.format_float(rel_negative * 100)}%"

        return is_flagged, violation_type, violated_threshold_value

    def process_old_new_pois_comparison(self, old_pois, new_pois, metric):
        """Process comparison between old and new POI data for a single table."""
        thresholds = self.load_metric_based_validation_thresholds(metric)
        # Build a lookup for new pois: (id, name, geom, category) -> record
        new_lookup = {}
        for rec in new_pois:
            key = tuple(rec.get(k) for k in ['id','name','geom']) + (rec.get('category'),)
            new_lookup[key] = rec

        comparison_list = []
        for old_rec in old_pois:
            key = tuple(old_rec.get(k) for k in ['id','name','geom']) + (old_rec.get('category'),)
            new_rec = new_lookup.get(key)
            
            if new_rec:
                new_val = new_rec.get(metric, 0)
                old_val = old_rec.get(metric, 0)
                category = new_rec.get('category', "")
                
                # Calculate metrics
                difference = new_val - old_val
                perc_diff = round(difference / old_val, 4) if old_val != 0 else 0

                if difference is not None and perc_diff is not None:
                    is_flagged, violation_type, violated_threshold_value = self.apply_flagging_logic(difference, perc_diff, old_val, category, thresholds)
                    output = {
                        "id": new_rec.get('id'),
                        "name": new_rec.get('name'),
                        "category": new_rec.get('category'),
                        f"new_{metric}": self.format_float(new_val),
                        f"old_{metric}": self.format_float(old_val),
                         "difference": self.format_float(difference),
                         "perc_diff": self.format_float(perc_diff * 100),
                         "geom": new_rec.get('geom'),
                         "flagged": is_flagged,
                         "violation_type": violation_type,
                         "violated_threshold": violated_threshold_value,
                     }
                    comparison_list.append(output)

        return comparison_list

    def parse_geom(self,val):
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
            
    def prepare_recs_for_export(self, recs, metric, only_flagged= False):
        """Prepare DataFrame for GPKG export with proper data types and filtering."""
        df = pd.DataFrame(recs)

        # Ensure numeric columns are float and fill missing values
        for col in df.columns:
            if (
                col.startswith("new_")
                or col.startswith("old_")
                or col == "perc_diff"
                or col == "difference"
            ):
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        # Converting geom column to a geometry
        if 'geom' in df.columns:
            df["geometry"] = df['geom'].apply(self.parse_geom)

        # Filter only the flagged data
        if only_flagged:
            df = df[df["flagged"] == True]
        if df.empty:
            return df

        # Apply lower_bound and upper_bound filtering
        lower_bound = self.config.validation["metrics"][metric].get("lower_bound")
        if  lower_bound is not None:
            metric_col_new = f"new_{metric}"
            metric_col_old = f"old_{metric}"
            df = df.loc[(df[metric_col_new] > lower_bound) | (df[metric_col_old] > lower_bound)]


        return df

    def export_category_layer(self, group, old_table, category, output_path):
        """Export a single category as a layer in the GPKG file."""
        if group.empty:
            return False
            
        # Create a clean table name from old_table
        if "." in old_table:
            table_name = old_table.split(".", 1)[1]  # Remove schema prefix
        else:
            table_name = old_table
        
        # Create layer name: table_name-category
        layer_name = f"{table_name}-{str(category)}"
        
        try:
            # Create GeoDataFrame
            gdf = gpd.GeoDataFrame(
                group.drop(columns=['geom']), 
                geometry="geometry", 
                crs="EPSG:4326"
            )
            
            # Remove rows with invalid geometry
            gdf = gdf[gdf.geometry.notnull()]
            
            if not gdf.empty:
                # Export to GPKG
                gdf.to_file(output_path, layer=layer_name, driver="GPKG")
                print_info(f"Exported layer '{layer_name}' with {len(gdf)} records")
                return True
            else:
                print_warning(f"No valid geometries for category '{category}' in {old_table}")
                return False
                
        except Exception as e:
            print_warning(f"Error exporting category '{category}' for {old_table}: {e}")
            return False

    def process_table_for_gpkg(self, old_table, recs, output_path, metric):
        """Process a single table's records for GPKG export."""
        print_info(f"Processing {len(recs)} records for '{old_table}'")

        # Prepare and filter DataFrame
        df_flagged = self.prepare_recs_for_export(recs,metric,only_flagged=True)
        if df_flagged.empty:
            print_warning(f"No records remaining after filtering for {old_table}, skipping...")
            return 0

        # Sort the DataFrame by category and percentage difference
        df_flagged = df_flagged.sort_values(by=['category', "perc_diff"], ascending=[True, False])

        # Export each category as a separate layer
        categories_exported = 0
        for category, group in df_flagged.groupby('category'):
            if self.export_category_layer(group, old_table, category, output_path):
                categories_exported += 1

        print_info(f"Exported {categories_exported} category layers for {old_table}")
        print_hashtags()
        
        return categories_exported

    def generate_metrics_based_gpkg_file(self, pois, output_path, metric):
        """
        Export the validation results to a GPKG file, creating separate layers for each category and old table.
        
        """
        print_info(f"Exporting to GPKG: {output_path}")
        print_hashtags()

        total_layers_exported = 0
        
        for table, recs in pois.items():
            if not recs:
                print_warning(f"No records found for {table}, skipping...")
                continue

            total_layers_exported += self.process_table_for_gpkg(table, recs, output_path, metric)

        if total_layers_exported == 0:
            print_warning(f"No layers were exported to {output_path}")
        else:
            print_info(f"Successfully exported {total_layers_exported} total layers to {output_path}")
            print_hashtags()
            
    def format_markdown_row(self, row, widths, header=False):
        """Format a row for Markdown with padded column widths."""
        escaped_row = [str(val).replace("_", "\_") for val in row]
        new_rows= (
            "| "
            + " | ".join(
                str(val).ljust(width) for val, width in zip(escaped_row, widths)
            )
            + " |\n"
        )
        if header:
           new_rows+= ("|" + "|".join("-" * (w + 2) for w in widths) + "|\n")
        return new_rows
            
    def process_metric_violations_for_markdown(self, recs, metric):
        """Process records to extract violations and build statistics."""
        violations = []
        summary_stats = {}
        
        for rec in recs:
            category = str(rec.get("category", "")).capitalize()
            county = rec.get("name", "")
            region_id = rec.get("id", "")
            old_val = rec.get(f"old_{metric}")
            new_val = rec.get(f"new_{metric}")
            difference = rec.get("difference")
            perc_diff = rec.get("perc_diff")
            is_flagged = rec.get("flagged")
            violation_type = rec.get("violation_type")
            violated_threshold_value = rec.get("violated_threshold")


            # Only append to violations if the item is already flagged
            if is_flagged:
                violations.append([
                    category, county, old_val, new_val, difference,
                    f"{perc_diff}%", violation_type,
                    violated_threshold_value,
                    region_id,
                ])

            # Update summary statistics for each category
            cat_key = str(rec.get("category", "")).lower()
            summary_stats.setdefault(cat_key, {
                "violations": 0, "old_total": 0, "new_total": 0, "diffs": []
            })
            if is_flagged:
                summary_stats[cat_key]["violations"] += 1
            summary_stats[cat_key]["old_total"] += old_val
            summary_stats[cat_key]["new_total"] += new_val
            if perc_diff != 0:
                summary_stats[cat_key]["diffs"].append(abs(perc_diff))

        return violations, summary_stats

    def write_metric_violations_table(self, f, violations, headers):
        """Write the violations table to the markdown file."""
        if violations:
            all_rows = [headers] + violations
            col_widths = [  max(len(str(row[i])) for row in all_rows) for i in range(len(headers)) ]
            f.write(self.format_markdown_row(headers, col_widths, header=True))
            for row in violations:
                f.write(self.format_markdown_row(row, col_widths))
            f.write(f"\n**Total Significant Changes Found:** {len(violations)}\n\n")
        else:
            f.write("✅ **No significant changes detected.**\n\n")

    def write_metric_summary_statistics(self, f, recs, violations,total_analyzed_combinations):
        """Write the summary statistics section."""
        f.write("## 📊 Summary Statistics\n\n")
        stats_headers = ["Statistic", "Value"]
        stats_rows = [
            ["Admin Units Analyzed", len(set(rec.get("id") for rec in recs))],
            ["Categories Analyzed", len(set(rec.get("category") for rec in recs))],
            ["Combinations of Admin Units and Categories", total_analyzed_combinations],
            ["Combinations with Significant Changes", len(violations)],
            [ "Rate of Significant Changes",
                (
                    f"{(len(violations) / total_analyzed_combinations * 100):.2f}%"
                    if total_analyzed_combinations > 0
                    else "0.00%"
                ),
            ],
        ]

        stats_col_widths = [ max(len(str(row[i])) for row in [stats_headers] + stats_rows) for i in range(2) ]
        f.write(self.format_markdown_row(stats_headers, stats_col_widths,header=True))
        for row in stats_rows:
            f.write(self.format_markdown_row(row, stats_col_widths))
        f.write("\n")

    def write_metric_breakdown_by_category(self, f, summary_stats, metric):
        """Write the breakdown by category section."""
        f.write("## 📋 Breakdown by Category of Significant Changes\n\n")
        breakdown_headers = [
            "Category", "Significant Changes",
            f"Old Total ({metric.replace('_', ' ').title()})",
            f"New Total ({metric.replace('_', ' ').title()})",
            "Avg Diff (%)", "Max Diff (%)",
        ]
        breakdown_rows = []
        for cat, stats in summary_stats.items():
            avg_diff = sum(stats["diffs"]) / len(stats["diffs"]) if stats["diffs"] else 0
            max_diff = max(stats["diffs"]) if stats["diffs"] else 0
            breakdown_rows.append([
                cat, stats["violations"], self.format_float(stats["old_total"]),
                self.format_float(stats["new_total"]), self.format_float(avg_diff),
                self.format_float(max_diff),
            ])
        
        breakdown_col_widths = [
            max(len(str(row[i])) for row in [breakdown_headers] + breakdown_rows)
            for i in range(len(breakdown_headers))
        ]
        f.write(self.format_markdown_row(breakdown_headers, breakdown_col_widths, header=True))
        for row in breakdown_rows:
            f.write(self.format_markdown_row(row, breakdown_col_widths))
        f.write("\n\n---\n\n")

    def process_table_for_markdown(self, f, old_table, recs, metric, metric_config, new_table, temp_ref_clone_name, region, units):
        """Process a single table's records for markdown report."""
        if not recs:
            return

        # Section Header Info
        f.write(f"### 📄 Table Details\n\n")
        f.write(f"- **Reference Geometry Table:** `{temp_ref_clone_name}`\n")
        f.write(f"- **Raw Database Table:** `{old_table}`\n")
        f.write(f"- **Local Database Table:** `{new_table}`\n")
        f.write(f"- **Region:** `{region}`\n")
        f.write(f"- **Units:** `{units}`\n\n")

        f.write("## 🚨 Significant Changes Detected\n\n")

        headers = [
            "Category", "Admin Name", f"Old {metric.replace('_', ' ').title()}",
            f"New {metric.replace('_', ' ').title()}", "Difference", "Difference (%)",
            "Violation Type", "Violated Thresholds", "Region ID",  # Changed "Violated Threshold" to "Violated Thresholds"
        ]


        total_analyzed_combinations = len(recs)
        # Sort and filter records
        recs = sorted(recs, key=lambda r: (
            str(r.get("category", "")).lower(),
            -(float(str(r.get("perc_diff", "0")).replace("%", ""))
              if r.get("perc_diff") not in [None, "N/A"] else float("-inf")),
        ))

        recs = self.prepare_recs_for_export(recs, metric).to_dict(orient="records")

        # Process violations and statistics
        violations, summary_stats = self.process_metric_violations_for_markdown(recs, metric)

        # Write sections
        self.write_metric_violations_table(f, violations, headers)
        self.write_metric_summary_statistics(f, recs, violations,total_analyzed_combinations)
        
        # Add disclaimer and explanations
        f.write("> <span style='color: red;'>⚠️ **Disclaimer:**</span> The possible combinations of administrative units and categories represent all theoretical pairs (admin units × categories). However, not every administrative unit contains every category, so the actual number of analyzed combinations may be lower than the maximum possible. \n\n")

        temp_clone = temp_ref_clone_name.replace("_", "\\_")
        escaped_old_table = old_table.replace("_", "\\_") 
        f.write(
            f"**Dataset Length**: _The total count of spatial join results between {temp_clone} <-> {escaped_old_table}._  \n"
            f"**Admin Units Analyzed**: _The number of unique administrative regions considered in the analysis._  \n"
            f"**Categories Analyzed**: _The number of different POI categories that were analyzed within each administrative unit._  \n"
            f"**Combinations with Significant Changes**: _The total number of category-specific changes exceeding the defined thresholds within each administrative unit._  \n"
            f"**Rate of Significant Changes**: _The percentage of category-specific changes exceeding the defined thresholds, relative to the total combinations analyzed within each administrative unit._  \n\n"
        )
        self.write_metric_breakdown_by_category(f, summary_stats, metric)

    def generate_metrics_based_markdown_report(self, pois, output_path, region, metric, temp_ref_clone_name='spatial_reference_table'):
        """
        Generate a Markdown report summarizing the validation results.
        Processes already-flagged data from compare_metric_based_new_and_old_results.

        Args:
            pois (dict): Dictionary containing the validation results with flagging already applied.
            output_path (str): Path to the output Markdown file.
            region (str): The region for which the report is generated.
            metric (str): The metric to use for validation (e.g., "poi_count", "poi_per_people").
            temp_ref_clone_name (str): Temporary reference clone name for reporting.
        """
        print_info(f"Generating Report: {output_path}")
        print_hashtags()

        # Setup configuration
        new_table = self.new_poi_table
        metric_config = self.config.validation["metrics"][metric]

        # Units description based on metric
        units = "Number of Features"
        people_count = metric_config.get("capita_people_count", "N/A")
        if metric == "poi_per_people":
            units = f"POIs per {people_count} people"
        elif metric == "population_per_poi":
            units = "People per POI"

        # Create the Markdown file and write the header
        with open(output_path, "w") as f:
            f.write(f"# Validation Report - **{metric.replace('_', ' ').upper()}** Metric\n\n")

            for old_table, recs in pois.items():
                self.process_table_for_markdown(
                    f, old_table, recs, metric, metric_config, 
                    new_table, temp_ref_clone_name, region, units
                )

    # ************************************************************************************** #
    # ********************** LONGEST COMMON SUBSEQUENCE (LCS) ANALYSIS ********************** #
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
        # name_weight = self.name_weight
        # category_weight = self.category_weight
        threshold_lcs = self.threshold_lcs

        # string_similarity_query = f"""
        #     similarity(ref.{name_col}, comp.{name_col}) AS name_lcs,
        #     similarity(ref.{category_col}, comp.{category_col}) AS category_lcs,
        #     ROUND(
        #         ((similarity(ref.{name_col}, comp.{name_col}) * {name_weight}) +
        #         (similarity(ref.{category_col}, comp.{category_col}) * {category_weight}))::numeric,
        #         2
        #     ) AS overall_lcs,
        #     ROUND(ST_Distance(ref.{geom_col}::geography, comp.{geom_col}::geography)::numeric, 2) AS distance_m,
        #     (ROUND(
        #         ((similarity(ref.{name_col}, comp.{name_col}) * {name_weight}) +
        #         (similarity(ref.{category_col}, comp.{category_col}) * {category_weight}))::numeric,
        #         2
        #     ) >= {threshold_lcs}) AS flagged
        # """
        
        string_similarity_query = f"""
            ROUND(SIMILARITY(ref.{name_col}, comp.{name_col})::numeric, 2) AS name_lcs,
            ROUND(ST_Distance(ref.{geom_col}::geography, comp.{geom_col}::geography)::numeric, 2) AS distance_m,
            (ROUND(SIMILARITY(ref.{name_col}, comp.{name_col})::numeric, 2) >= {threshold_lcs}) AS flagged
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

        print_info(
            f"Performing LCS Analysis on '{self.new_poi_table}' using PSQL 'Similarity' and ST_Distance()"
        )
        print_hashtags()

        id_col, name_col, category_col, geom_col = self.poi_columns
        similarity_sql = self.get_lcs_similarity_sql()
        table = self.new_poi_table
        search_radius = self.proximity_radius_m
        excluded_categories_clause = ", ".join(
            [f"'{cat}'" for cat in self.excluded_categories]
        )
        threshold_lcs = self.threshold_lcs

        # query = f"""
        #     SELECT
        #         ref.{id_col} AS ref_id,
        #         ref.{name_col} AS ref_name,
        #         ref.{category_col} AS ref_category,
        #         ST_AsText(ref.{geom_col}) AS ref_geom,
        #         comp.{id_col} AS comp_id,
        #         comp.{name_col} AS comp_name,
        #         comp.{category_col} AS comp_category,
        #         ST_AsText(comp.{geom_col}) AS comp_geom,
        #         {similarity_sql}
        #     FROM {table} ref
        #     JOIN {table} comp
        #         ON ref.{id_col} <> comp.{id_col}
        #         AND ST_DWithin(
        #             ST_Transform(ref.{geom_col}, 3857),
        #             ST_Transform(comp.{geom_col}, 3857),
        #             {search_radius}
        #         )
        #     WHERE ref.{name_col} IS NOT NULL AND comp.{name_col} IS NOT NULL
        #     AND ref.{category_col} NOT IN ({excluded_categories_clause})
        #     AND comp.{category_col} NOT IN ({excluded_categories_clause})
        # """

        query = f"""
            WITH lcs_results AS (
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
                        ref.{geom_col},
                        comp.{geom_col},
                        {search_radius}
                    )
                WHERE ref.{name_col} IS NOT NULL AND comp.{name_col} IS NOT NULL
                AND ref.{category_col} = comp.{category_col}
                AND ref.{category_col} NOT IN ({excluded_categories_clause})
                AND comp.{category_col} NOT IN ({excluded_categories_clause})
            ) SELECT * FROM lcs_results WHERE name_lcs >= {threshold_lcs};
        """

        print_info("Running SQL query for LCS analysis...")
        df = db_new.select(query)
        print_hashtags()

        columns = [
            "ref_id",
            "ref_name",
            "ref_category",
            "ref_geom",
            "comp_id",
            "comp_name",
            "comp_category",
            "comp_geom",
            "name_lcs",
            # "category_lcs",
            # "overall_lcs",
            "distance_m",
            "flagged",
        ]
        lcs_df = (
            pd.DataFrame(df, columns=columns) if df else pd.DataFrame(columns=columns)
        )

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

        print_info("Preparing LCS DataFrame for export...")
        print_hashtags()

        if df is None or df.empty:
            return pd.DataFrame()

        # Do the aggregation of compared POIs based on each reference POI
        group_cols = ["ref_id", "ref_name", "ref_category", "ref_geom"]
        comp_cols = [col for col in df.columns if col.startswith("comp_")]
        array_group_cols = [
            "distance_m",
            "name_lcs",
            # "category_lcs",
            # "overall_lcs",
            "flagged",
        ]
        other_cols = [
            col
            for col in df.columns
            if col not in group_cols + comp_cols + array_group_cols
        ]

        seen_comp_ids = set()
        seen_ref_ids = set()
        output_rows = []

        # Sort for deterministic grouping
        df = df.sort_values(group_cols).reset_index(drop=True)

        for _, row in df.iterrows():
            ref_id = row["ref_id"]
            comp_id = row["comp_id"]

            # If this ref_id has already been grouped as a comp_id, skip this row
            if ref_id in seen_comp_ids:
                continue

            # If this comp_id has already been a ref_id before, skip this row
            if comp_id in seen_ref_ids:
                continue

            # Group all rows with this ref_id
            group = df[df["ref_id"] == ref_id].copy()
            # Deduplicate comp_id and keep all comp_* and array_group_cols in sync
            if "comp_id" in group.columns:
                _, unique_indices = np.unique(group["comp_id"], return_index=True)
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
            seen_comp_ids.update(group["comp_id"].dropna().unique())
            # Mark this ref_id as seen
            seen_ref_ids.add(ref_id)

        # Ensure column order: ref columns, comp columns, array_group_cols, then other columns
        columns_order = group_cols + comp_cols + array_group_cols + other_cols
        result_df = pd.DataFrame(output_rows)
        columns_order = [col for col in columns_order if col in result_df.columns]
        result_df = result_df[columns_order]

        # Remove duplicate rows based on reference columns and other_cols (skip list columns)
        dedup_cols = group_cols + other_cols
        result_df = result_df.drop_duplicates(subset=dedup_cols, keep="first")

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

        print_info(f"Exporting LCS results to GPKG: {gpkg_output_path}")
        print_hashtags()

        if df is None or df.empty:
            print_warning(f"No LCS results to export for path: {gpkg_output_path}")
            return

        # --- 1. Reference Points Layer ---
        ref_points = []
        for _, row in df.iterrows():
            ref_geom = row["ref_geom"]
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
            ref_points.append(
                {
                    "id": row["ref_id"],
                    "name": row["ref_name"],
                    "category": row["ref_category"],
                    "geom": geom,
                }
            )
        ref_gdf = gpd.GeoDataFrame(ref_points, geometry="geom", crs="EPSG:4326")
        ref_gdf = ref_gdf[ref_gdf["geom"].notnull()]

        # --- 2. Proximity Zones Layer ---
        # Project to a metric CRS for accurate buffering
        ref_gdf_metric = ref_gdf.to_crs(epsg=3857)
        prox_zones = []
        for _, row in ref_gdf_metric.iterrows():
            if row.geom is not None:
                proximity_buffer_geom = row.geom.buffer(self.proximity_radius_m * 111000)
                prox_zones.append(
                    {
                        "id": row["id"],
                        "name": row["name"],
                        "category": row["category"],
                        "geom": proximity_buffer_geom,
                    }
                )
        prox_gdf_metric = gpd.GeoDataFrame(prox_zones, geometry="geom", crs="EPSG:3857")
        prox_gdf_metric = prox_gdf_metric[prox_gdf_metric["geom"].notnull()]
        # Reproject back to WGS84 for export
        prox_gdf = prox_gdf_metric.to_crs(epsg=4326)

        # --- 3. Duplicate Points Layer ---
        dup_points = []
        for _, row in df.iterrows():
            comp_ids = (
                row["comp_id"] if isinstance(row["comp_id"], list) else [row["comp_id"]]
            )
            comp_names = (
                row["comp_name"]
                if isinstance(row["comp_name"], list)
                else [row["comp_name"]]
            )
            comp_categories = (
                row["comp_category"]
                if isinstance(row["comp_category"], list)
                else [row["comp_category"]]
            )
            comp_geoms = (
                row["comp_geom"]
                if isinstance(row["comp_geom"], list)
                else [row["comp_geom"]]
            )
            for cid, cname, ccat, cgeom in zip(comp_ids, comp_names, comp_categories, comp_geoms,strict=True):
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
                dup_points.append(
                    {"id": cid, "name": cname, "category": ccat, "geom": geom}
                )
        dup_gdf = gpd.GeoDataFrame(dup_points, geometry="geom", crs="EPSG:4326")
        dup_gdf = dup_gdf[dup_gdf["geom"].notnull()]

        # --- Write all layers to a single GPKG file ---
        prox_gdf.to_file(gpkg_output_path, layer="proximity_zones", driver="GPKG")
        dup_gdf.to_file(gpkg_output_path, layer="duplicate_points", driver="GPKG")
        ref_gdf.to_file(gpkg_output_path, layer="reference_points", driver="GPKG")

        ref_count = len(ref_gdf)
        dup_count = len(dup_gdf)

        return ref_count, dup_count

    def generate_lcs_based_markdown_report(self, db: Database, df: pd.DataFrame, ref_count, dup_count, md_output_path: str):
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
        cols = [
            "ref_id",
            "ref_name",
            "ref_category",
            "comp_id",
            "comp_name",
            "comp_category",
            "comp_geom",
            "distance_m",
            "name_lcs",
            # "overall_lcs",
        ]
        df = df[cols]

        # Flatten the DataFrame: one row per (ref_id, comp_id) pair
        rows = []
        for _, row in df.iterrows():
            ref_id = row["ref_id"]
            ref_name = row["ref_name"]
            ref_category = row["ref_category"]
            comp_ids = (
                row["comp_id"] if isinstance(row["comp_id"], list) else [row["comp_id"]]
            )
            comp_names = (
                row["comp_name"]
                if isinstance(row["comp_name"], list)
                else [row["comp_name"]]
            )
            distances = (
                row["distance_m"]
                if isinstance(row["distance_m"], list)
                else [row["distance_m"]]
            )
            # overall_lcs = (
            #     row["overall_lcs"]
            #     if isinstance(row["overall_lcs"], list)
            #     else [row["overall_lcs"]]
            # )
            name_lcs = (
                row["name_lcs"]
                if isinstance(row["name_lcs"], list)
                else [row["name_lcs"]]
            )
            
            for cid, cname, dist, score in zip(comp_ids, comp_names, distances, name_lcs, strict=True):
                rows.append(
                    {
                        "Reference ID": ref_id,
                        "Reference Name": ref_name,
                        "Category": ref_category,
                        "Duplicate ID": cid,
                        "Duplicate Name": cname,
                        "Distance (m)": dist,
                        # "Overall LCS": score,
                        "Name LCS": score,
                    }
                )

        flat_df = pd.DataFrame(rows)

        # Set column widths so headers do not wrap
        headers = [
            "Reference ID",
            "Reference Name",
            "Category",
            "Duplicate ID",
            "Duplicate Name",
            "Distance (m)",
            # "Overall LCS",
            "Name LCS",
        ]
        col_widths = [
            max(
                len(str(val)) for val in [header] + flat_df[header].astype(str).tolist()
            )
            for header in headers
        ]

        def format_row_md(row, widths):
            return (
                "| "
                + " | ".join(str(val).ljust(width) for val, width in zip(row, widths, strict=True))
                + " |\n"
            )

        total_pois_count = db.select(
            f"SELECT COUNT(*) FROM {self.new_poi_table}"
        )[0][0]
        total_pois_name_not_null_count = db.select(
            f"SELECT COUNT(*) FROM {self.new_poi_table} WHERE name IS NOT NULL"
        )[0][0]

        with open(md_output_path, "w") as f:
            f.write("# OSM POI Data Duplication Report\n\n")
            f.write(
                "This report lists all detected duplicate POIs based on Lowest Common Subsequence (LCS) similarity method.\n\n"
            )
            f.write("**LCS Parameters Used:**\n\n")
            f.write("- **Attributes of Interest:** Name, Category, Geometry\n")
            f.write(f"- **Proximity Radius (meters):** `~{40}`\n")
            f.write(f"- **Similarity Threshold (LCS):** `{self.threshold_lcs * 100 }%`\n\n")

            # Add summary of layers and counts
            f.write("## Summary of Layers\n\n")
            f.write(f"- **Total Points:** `{total_pois_count}`\n")
            f.write(f"- **Points with Name Populated:** `{total_pois_name_not_null_count}`\n")
            f.write(f"- **Reference Points Count:** `{ref_count}`\n")
            f.write(f"- **Duplicate Points Count:** `{dup_count}`\n\n")

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

    if region == "europe":
        for loop_region in validator.config.regions:
            for current_metric in all_metrics:
                print_info(f"Performing calculations for metric '{current_metric}'")
                print_hashtags()
                process_poi_validation(validator, "poi", loop_region, current_metric,  db_old, db_new)
        
    else:
        for current_metric in all_metrics:
            print_info(f"Performing calculations for metric '{current_metric}'")
            print_hashtags()
            process_poi_validation(validator, "poi", region, current_metric, db_old, db_new)
            drop_query = f"DROP TABLE IF EXISTS spatial_reference_table;"
            db_new.perform(drop_query)

#    lcs_results_df = validator.run_lcs_analysis_sql(db_new)
#    processed_lcs_df = validator.prepare_lcs_dataframe_for_export(lcs_results_df)
#    lcs_gpkg_output_path = os.path.join(validator.data_dir, "poi", f"poi_validation_lcs_{region}.gpkg")
#    lcs_md_output_path = os.path.join(validator.data_dir, "poi", f"poi_validation_lcs_{region}.md")
#    ref_count, dup_count = validator.generate_lcs_based_gpkg_file(processed_lcs_df, lcs_gpkg_output_path)
#    validator.generate_lcs_based_markdown_report(db_new, processed_lcs_df, ref_count, dup_count, lcs_md_output_path)

    db_old.close()
    db_new.close()

def process_poi_validation(validator, dataset_type: str, region: str, metric: str,  db_old, db_new):
    unified_results={}
    validator.create_temp_geom_reference_table(db_old, db_new, metric)
    old_table_categories = validator.get_categories_from_old_tables(db_old)
    for old_poi_table in validator.old_poi_tables:
        # Get categories for this specific old table
        categories_for_this_table = old_table_categories.get(old_poi_table, [])
        if not categories_for_this_table:
                print_warning(f"No categories found for {old_poi_table}, skipping...")
                continue
        # Process POI table data
        new_results, old_results = validator.process_old_new_poi_table_data(db_old, db_new, old_poi_table, categories_for_this_table, metric)
        if not new_results:
            print_warning(f"No new POI results found for {old_poi_table}")
            continue
        comparison_list = validator.process_old_new_pois_comparison( old_results, new_results, metric)
        print_info(f"Comparison results for {old_poi_table}")
        unified_results[old_poi_table] = comparison_list


    gpkg_output_path = os.path.join(validator.data_dir, dataset_type, f"{dataset_type}_validation_{metric}_{region}.gpkg")
    validator.generate_metrics_based_gpkg_file(unified_results, gpkg_output_path, metric=metric)

    md_output_path = os.path.join(validator.data_dir, dataset_type, f"{dataset_type}_validation_{metric}_{region}.md")
    validator.generate_metrics_based_markdown_report(unified_results, md_output_path, region=region, metric=metric)

if __name__ == "__main__":
    validate_poi()