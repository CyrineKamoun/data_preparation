import os
import re
import json
import time
import pandas as pd
import geopandas as gpd
from shapely.geometry import shape
from collections import defaultdict

from src.config.config import Config
from src.core.config import settings
from src.db.db import Database
from src.utils.utils import timing, print_info, print_hashtags, print_warning, print_error

class PoiValidation:
    """Validates the cloud database POIs by comparing them with the preparation database."""
    
    def __init__(self, db_config, region):
        self.db_config = db_config
        self.region = region
        self.data_dir = settings.INPUT_DATA_DIR
        self.config = Config("poi", region)
        self.raw_poi_tables = self.config.validation["raw_poi_table"]
        self.local_poi_table = self.config.validation["local_poi_table"]
        
        # Remove single metric selection - now we handle all metrics
        self.all_metrics = list(self.config.validation["metrics"].keys())
        
        # For backward compatibility, keep these but they'll be used per metric
        # Use first metric as default for any legacy code that might reference these
        self.default_metric = self.all_metrics[0] if self.all_metrics else "count"
        self.thresholds = self.config.validation["metrics"][self.default_metric]["thresholds"]
        self.geom_reference_query = self.config.validation["metrics"][self.default_metric]["geom_reference_query"]

    def cross_database_spatial_validation(self, metric_key=None):
        """Perform validation using GeoPandas spatial joins across databases."""
        
        if metric_key is None:
            metric_key = self.default_metric
        
        print_info(f"Starting cross-database validation for metric: {metric_key}")
        print_hashtags()
        # Step 1: Load polygon geometries from raw database
        polygons_gdf = self._load_polygons_geopandas(self.geom_reference_query)
        
        # Step 2: Load local POI data using unified function
        local_gdf = self._load_pois_geopandas(
            self.local_poi_table, 
            settings.LOCAL_DATABASE_URI, 
            "local"
        )
        
        if local_gdf.empty:
            print_error("No local POI data found - cannot proceed with validation")
            return {}
        
        # Step 3: Process each raw POI table
        results = {}
        for raw_table in self.raw_poi_tables:          
            # Load raw POI data using unified function
            raw_gdf = self._load_pois_geopandas(
                raw_table,
                settings.RAW_DATABASE_URI,
                "raw"
            )
            
            if raw_gdf.empty:
                print_warning(f"No data found in {raw_table}")
                continue
            
            # Perform spatial joins and comparison
            comparison_result = self._compare_pois_spatial(
                local_gdf, raw_gdf, polygons_gdf, metric_key, raw_table
            )
            
            if not comparison_result.empty:
                results[raw_table] = comparison_result
            else:
                print_warning(f"No comparison results for {raw_table}")

        return results

    def _load_polygons_geopandas(self, geom_query):
        """Load polygon geometries from raw database into GeoDataFrame."""
        
        db_raw = Database(settings.RAW_DATABASE_URI)
        
        try:
            # Use the geom_query directly from YAML - no modifications needed
            data = db_raw.select(geom_query)
            
            # Create DataFrame - columns should match the query output
            df = pd.DataFrame(data, columns=['id', 'name', 'geom'])
            
            # Convert to GeoDataFrame
            gdf = gpd.GeoDataFrame(
                df,
                geometry=gpd.GeoSeries.from_wkt(df['geom']),
                crs='EPSG:4326'
            )
            
            # Drop the WKT column as we now have proper geometry
            gdf = gdf.drop('geom', axis=1)
    
            return gdf
            
        except Exception as e:
            print_error(f"Error loading polygons: {e}")
            raise
        finally:
            db_raw.conn.close()

    def _load_pois_geopandas(self, table_name, database_uri, table_type="POI"):
        """Load POI data into GeoDataFrame - unified function."""
        
        db = Database(database_uri)
        
        query = f"""
            SELECT
                category,
                ST_AsText(geom) as geom
            FROM {table_name}
            WHERE category IS NOT NULL
            ORDER BY category
        """
        
        try:
            data = db.select(query)
            
            if not data:
                print_warning(f"No data found in {table_name}")
                return gpd.GeoDataFrame(columns=['category'], geometry=[])
            
            df = pd.DataFrame(data, columns=['category', 'geom'])
            
            # Convert to GeoDataFrame
            gdf = gpd.GeoDataFrame(
                df[['category']],
                geometry=gpd.GeoSeries.from_wkt(df['geom']),
                crs='EPSG:4326'
            )

            return gdf
            
        except Exception as e:
            print_error(f"Error loading POIs from {table_name}: {e}")
            return gpd.GeoDataFrame(columns=['category'], geometry=[])
        finally:
            db.conn.close()

    def _load_local_pois_geopandas(self):
        """Load local POI data into GeoDataFrame."""
        return self._load_pois_geopandas(
            self.local_poi_table, 
            settings.LOCAL_DATABASE_URI, 
            "local"
        )

    def _load_raw_pois_geopandas(self, raw_table):
        """Load raw POI data into GeoDataFrame."""
        return self._load_pois_geopandas(
            raw_table, 
            settings.RAW_DATABASE_URI, 
            "raw"
        )

    def _compare_pois_spatial(self, local_gdf, raw_gdf, polygons_gdf, metric_key, raw_table):
        """Perform spatial comparison between local and raw POI data."""
        
        # Spatial join: POIs within polygons
        local_joined = gpd.sjoin(local_gdf, polygons_gdf, how='left', predicate='within')
        raw_joined = gpd.sjoin(raw_gdf, polygons_gdf, how='left', predicate='within')
        
        # Remove POIs that don't fall within any region
        initial_local_count = len(local_joined)
        initial_raw_count = len(raw_joined)
        
        local_joined = local_joined.dropna(subset=['id'])  # Fix: use 'id' not 'region_id'
        raw_joined = raw_joined.dropna(subset=['id'])      # Fix: use 'id' not 'region_id'
        
        if local_joined.empty or raw_joined.empty:
            print_warning("No POIs found within regions after spatial join")
            return gpd.GeoDataFrame()
        
        # Rename id to region_id for consistency with the rest of the code
        local_joined = local_joined.rename(columns={'id': 'region_id'})
        raw_joined = raw_joined.rename(columns={'id': 'region_id'})
        
        # Calculate metrics based on type        
        if metric_key == "count":
            local_metrics = self._calculate_count_metrics(local_joined)
            raw_metrics = self._calculate_count_metrics(raw_joined)
            merge_columns = ['region_id', 'name', 'category']
            
        elif metric_key == "density":
            local_metrics = self._calculate_density_metrics(local_joined, polygons_gdf)
            raw_metrics = self._calculate_density_metrics(raw_joined, polygons_gdf)
            merge_columns = ['region_id', 'name', 'category']
            
        elif metric_key == "category_distribution":
            local_metrics = self._calculate_category_distribution(local_joined)
            raw_metrics = self._calculate_category_distribution(raw_joined)
            merge_columns = ['region_id', 'name']
        
        else:
            raise ValueError(f"Unknown metric: {metric_key}")

        # Filter common categories for category-based metrics
        if metric_key in ["count", "density"]:
            local_categories = set(local_metrics['category']) if not local_metrics.empty else set()
            raw_categories = set(raw_metrics['category']) if not raw_metrics.empty else set()
            common_categories = local_categories.intersection(raw_categories)
            
            if not common_categories:
                print_warning(f"No common categories found between local and raw data")
                return gpd.GeoDataFrame()
            
            local_metrics = local_metrics[local_metrics['category'].isin(common_categories)]
            raw_metrics = raw_metrics[raw_metrics['category'].isin(common_categories)]
        
        if local_metrics.empty or raw_metrics.empty:
            print_warning("No metrics calculated after filtering")
            return gpd.GeoDataFrame()
        
        # Merge local and raw metrics
        comparison = pd.merge(
            raw_metrics.add_suffix('_raw'),
            local_metrics.add_suffix('_local'),
            left_on=[col + '_raw' for col in merge_columns],
            right_on=[col + '_local' for col in merge_columns],
            how='inner'  # Changed from 'outer' to 'inner' to avoid unnecessary zeros
        )
        
        if comparison.empty:
            print_warning("No matching records found during merge")
            return gpd.GeoDataFrame()
        
        # Clean up column names
        for col in merge_columns:
            comparison[col] = comparison[f'{col}_raw'].combine_first(comparison[f'{col}_local'])
            comparison = comparison.drop([f'{col}_raw', f'{col}_local'], axis=1)
        
        # Calculate difference with better error handling
        metric_col_raw = f'{metric_key}_raw'
        metric_col_local = f'{metric_key}_local'
        
        if metric_col_raw not in comparison.columns or metric_col_local not in comparison.columns:
            print_error(f"Missing metric columns: {metric_col_raw}, {metric_col_local}")
            return gpd.GeoDataFrame()
        
        # Ensure numeric columns
        comparison[metric_col_raw] = pd.to_numeric(comparison[metric_col_raw], errors='coerce').fillna(0)
        comparison[metric_col_local] = pd.to_numeric(comparison[metric_col_local], errors='coerce').fillna(0)
        
        comparison['difference'] = comparison.apply(
            lambda row: (
                abs(row[metric_col_raw] - row[metric_col_local]) / row[metric_col_raw]
                if row[metric_col_raw] != 0 
                else (1.0 if row[metric_col_local] != 0 else 0.0)  # Handle case where raw=0 but local>0
            ), axis=1
        ).round(3)
        
        # Add geometry back for spatial export
        # Fix: use 'id' column from polygons_gdf to match with 'region_id'
        geometry_df = polygons_gdf[['id', 'geometry']].rename(columns={'id': 'region_id'})
        
        comparison = comparison.merge(
            geometry_df,
            on='region_id',
            how='left'
        )
        
        # Remove records without geometry
        comparison = comparison.dropna(subset=['geometry'])
        
        if comparison.empty:
            print_warning("No records with valid geometry")
            return gpd.GeoDataFrame()

        return gpd.GeoDataFrame(comparison, geometry='geometry', crs='EPSG:4326')

    def _calculate_count_metrics(self, joined_gdf):
        """Calculate count metrics by region and category."""
        
        if joined_gdf.empty:
            return pd.DataFrame(columns=['region_id', 'name', 'category', 'count'])
        
        return joined_gdf.groupby(['region_id', 'name', 'category']).size().reset_index(name='count')

    def _calculate_density_metrics(self, joined_gdf, polygons_gdf):
        """Calculate density metrics (points per 100 km²)."""
        
        if joined_gdf.empty:
            return pd.DataFrame(columns=['region_id', 'name', 'category', 'density'])
        
        # Calculate area in km²
        polygons_gdf_proj = polygons_gdf.to_crs('EPSG:3857')  # Project to calculate area
        area_km2 = polygons_gdf_proj.geometry.area / 1000000  # Convert to km²
        
        # Count points by region and category
        counts = joined_gdf.groupby(['region_id', 'name', 'category']).size().reset_index(name='count')
        
        # Add area information - Fix: use 'id' column from polygons_gdf
        area_df = pd.DataFrame({
            'region_id': polygons_gdf['id'],  # Fix: changed from 'region_id' to 'id'
            'area_km2': area_km2
        })
        
        density_result = counts.merge(area_df, on='region_id', how='left')
        
        # Calculate density (points per 100 km²) with error handling
        density_result['density'] = density_result.apply(
            lambda row: (row['count'] * 100.0) / row['area_km2'] if row['area_km2'] > 0 else 0,
            axis=1
        )
        
        return density_result[['region_id', 'name', 'category', 'density']]

    def _calculate_category_distribution(self, joined_gdf):
        """Calculate category distribution metrics."""
        
        if joined_gdf.empty:
            return pd.DataFrame(columns=['region_id', 'name', 'category_distribution', 'category'])
        
        category_dist = joined_gdf.groupby(['region_id', 'name'])['category'].nunique().reset_index(name='category_distribution')
        category_dist['category'] = 'overall'
        
        return category_dist

    def validate_all_metrics_spatial(self, dataset_type, region):
        """Validate all metrics using GeoPandas spatial joins."""
        
        all_results = {}
        
        successful_metrics = 0
        total_metrics = len(self.all_metrics)
        
        for metric_key in self.all_metrics:
            try:
                print_info(f"Processing metric: {metric_key} ({successful_metrics + 1}/{total_metrics})")
                print_hashtags()
                results = self.cross_database_spatial_validation(metric_key)
                
                if not results:
                    print_warning(f"No results generated for metric: {metric_key}")
                    all_results[metric_key] = {}
                    continue
                
                # Export results
                exported_files = 0
                for raw_table, gdf in results.items():
                    if not gdf.empty:
                        try:
                            self._export_spatial_results(gdf, raw_table, metric_key, dataset_type, region)
                            self._generate_spatial_report(gdf, raw_table, metric_key, dataset_type, region)
                            exported_files += 1
                        except Exception as export_error:
                            print_error(f"Error exporting results for {raw_table}: {export_error}")
                
                all_results[metric_key] = results
                successful_metrics += 1
  
            except Exception as e:
                print_error(f"❌ Error processing metric {metric_key}: {str(e)}")
                # Print stack trace for debugging
                import traceback
                print_error(f"Stack trace: {traceback.format_exc()}")
                all_results[metric_key] = None

        return all_results

    def _export_spatial_results(self, gdf, raw_table, metric_key, dataset_type, region):
        """Export spatial results to GPKG."""
        
        if gdf.empty:
            print_warning(f"Skipping export - no data for {raw_table} {metric_key}")
            return None
        
        try:
            # Generate clean filenames
            table_name = raw_table.split('.')[-1]
            filename = f"{table_name}_validation_{metric_key}_{region}.gpkg"
            layername = f"{table_name}_{metric_key}_validation"
            
            # Create output path
            output_dir = os.path.join(self.data_dir, dataset_type)
            output_file = os.path.join(output_dir, filename)
            
            # Ensure directory exists
            os.makedirs(output_dir, exist_ok=True)
            
            # Validate GeoDataFrame before export
            if not hasattr(gdf, 'geometry') or gdf.geometry.empty:
                print_error(f"Invalid geometry in GeoDataFrame for {raw_table}")
                return None
            
            # Clean column names for GPKG compatibility
            clean_gdf = gdf.copy()
            
            # GPKG field name restrictions (max 63 chars, no special chars)
            column_mapping = {}
            for col in clean_gdf.columns:
                if col != 'geometry':
                    # Clean column names: remove special chars, limit length
                    clean_col = re.sub(r'[^a-zA-Z0-9_]', '_', str(col))[:63]
                    if clean_col != col:
                        column_mapping[col] = clean_col
            
            if column_mapping:
                clean_gdf = clean_gdf.rename(columns=column_mapping)
            
            # Ensure CRS is set
            if clean_gdf.crs is None:
                clean_gdf = clean_gdf.set_crs('EPSG:4326')
            
            # Export to GPKG
            clean_gdf.to_file(
                output_file, 
                driver="GPKG", 
                layer=layername,
                engine='fiona'  # Explicit engine for better error handling
            )
            
            # Verify export
            file_size = os.path.getsize(output_file)
            print_info(f"✅ Exported: {output_file} ({file_size:,} bytes, {len(gdf)} records)")
            
            return output_file
            
        except Exception as e:
            print_error(f"❌ Export failed for {raw_table} {metric_key}: {e}")
            # Try fallback export without problematic columns
            try:
                fallback_gdf = gdf[['geometry']].copy()
                fallback_file = output_file.replace('.gpkg', '_geometry_only.gpkg')
                fallback_gdf.to_file(fallback_file, driver="GPKG", layer=f"{layername}_geom")
                print_warning(f"Fallback export (geometry only): {fallback_file}")
                return fallback_file
            except Exception as fallback_error:
                print_error(f"Fallback export also failed: {fallback_error}")
                return None

    def _generate_spatial_report(self, gdf, raw_table, metric_key, dataset_type, region):
        """Generate markdown report for spatial validation results."""
        
        if gdf.empty:
            print_warning(f"No data to generate report for {raw_table} {metric_key}")
            return None
        
        try:
            # Get thresholds for the specific metric
            metric_config = self.config.validation["metrics"].get(metric_key, {})
            thresholds = metric_config.get("thresholds", {"default": 0.1})
            
            # Setup output
            report_dir = os.path.join(self.data_dir, dataset_type)
            os.makedirs(report_dir, exist_ok=True)
            
            # Generate report content
            report_content = self._build_report_content(gdf, raw_table, metric_key, region, thresholds)
            
            # Save markdown file
            raw_table_name = raw_table.split('.')[-1]
            report_filename = f"{raw_table_name}_validation_{metric_key}_{region}.md"
            report_path = os.path.join(report_dir, report_filename)
            
            with open(report_path, "w", encoding='utf-8') as f:
                f.write(report_content)
            
            # Log results
            violation_count = self._count_violations(gdf, thresholds)
            
            if violation_count > 0:
                print_warning(f"⚠️  Found {violation_count} violations in {raw_table} for {metric_key}")
            else:
                print_info(f"✅ No violations found in {raw_table} for {metric_key}")
            
            print_info(f"📄 Generated report: {report_filename}")
            print_hashtags()
            
            return report_path

            
        except Exception as e:
            print_error(f"❌ Failed to generate report for {raw_table} {metric_key}: {e}")
            print_hashtags()
            return None

    def _build_report_content(self, gdf, raw_table, metric_key, region, thresholds):
        """Build the complete markdown report content."""
        
        # Header section
        content = self._build_header_section(raw_table, metric_key, region)
        
        # Violations section
        violations_content, violation_count = self._build_violations_section(gdf, metric_key, thresholds)
        content += violations_content
        
        # Summary statistics section
        content += self._build_summary_section(gdf, metric_key, violation_count)
        
        # Category breakdown (if applicable)
        if violation_count > 0 and 'category' in gdf.columns:
            content += self._build_category_breakdown(gdf, metric_key, thresholds)
        
        return content

    def _build_header_section(self, raw_table, metric_key, region):
        """Build the header section of the report."""
        
        lines = [
            f"# Validation Report - {metric_key.upper()} Metric\n\n",
            f"**Raw Table**: `{raw_table}`  \n",
            f"**Local Table**: `{self.local_poi_table}`  \n", 
            f"**Region**: `{region}`  \n",
            f"**Metric**: `{metric_key}`  \n"
        ]
        
        # Add unit information
        if metric_key == "density":
            lines.append("**Units**: Points per 100 square kilometers  \n")
        elif metric_key == "category_distribution":
            lines.append("**Units**: Number of distinct categories  \n")
        elif metric_key == "count":
            lines.append("**Units**: Number of Features  \n")
        
        lines.append("\n---\n\n")
        return "".join(lines)

    def _build_violations_section(self, gdf, metric_key, thresholds):
        """Build the violations section with one row per (category, county)."""
        lines = ["## 🚨 Threshold Violations\n\n"]
        gdf_sorted = gdf.sort_values(by=["category", "name"] if "category" in gdf.columns else ["name"])

        rows = []
        for idx, row in gdf_sorted.iterrows():
            category = row.get("category", "overall")
            difference = row.get("difference", 0)
            threshold = thresholds.get(category, thresholds.get("default", 0.1))

            if difference > threshold:
                raw_value = row.get(f"{metric_key}_raw", 0)
                local_value = row.get(f"{metric_key}_local", 0)
                region_id = row.get("region_id")
                name = row.get("name")

                rows.append({
                    "Category": category.title(),
                    "County": name,
                    "Raw Value": f"{raw_value:.2f}",
                    "Local Value": f"{local_value:.2f}",
                    "Difference (%)": f"{difference * 100:.2f}%",
                    "Threshold (%)": f"{threshold * 100:.2f}%",
                    "Region ID": region_id
                })

        if not rows:
            lines.extend([
                "🎉 **No threshold violations found!**\n\n",
                "All validation checks passed successfully.\n\n"
            ])
        else:
            # Create Markdown Table Header
            headers = ["Category", "County", "Raw Value", "Local Value", "Difference (%)", "Threshold (%)", "Region ID"]
            lines.append("| " + " | ".join(headers) + " |\n")
            lines.append("|" + "|".join(["---"] * len(headers)) + "|\n")

            # Add rows to table
            for r in rows:
                lines.append("| " + " | ".join(str(r[col]) for col in headers) + " |\n")

            lines.append(f"\n**Total Violations Found**: {len(rows)}\n\n")

        return "".join(lines), len(rows)


    def _build_summary_section(self, gdf, metric_key, violation_count):
        """Build the summary statistics section."""
        
        lines = ["## 📊 Summary Statistics\n\n"]
        
        # Calculate statistics safely
        total_regions = len(gdf['region_id'].unique()) if 'region_id' in gdf.columns else len(gdf)
        total_records = len(gdf)

        violation_rate = (violation_count / total_records * 100) if total_records > 0 else 0
        
        # Build summary table
        lines.extend([
            "| Statistic | Value |\n",
            "|-----------|-------|\n",
            f"| **Regions Analyzed** | {total_regions} |\n",
            f"| **Records Analyzed** | {total_records} |\n",
            f"| **Violations Found** | {violation_count} |\n",
            f"| **Violation Rate** | {violation_rate:.2f}% |\n"
        ])
        
        return "".join(lines)

    def _build_category_breakdown(self, gdf, metric_key, thresholds):
        """Build category breakdown section for violated categories only."""
        
        lines = ["## 📋 Violation Breakdown by Category\n\n"]
        
        # Filter only violated records
        violated_records = gdf[gdf.apply(
            lambda row: row.get('difference', 0) > thresholds.get(
                row.get('category', 'overall'), 
                thresholds.get('default', 0.1)
            ),
            axis=1
        )]
        
        if violated_records.empty:
            return ""
        
        # Group violations by category
        try:
            category_violations = violated_records.groupby('category').agg({
                'difference': ['count', 'mean', 'max'],
                f'{metric_key}_raw': 'sum',
                f'{metric_key}_local': 'sum'
            }).round(4)
            
            lines.extend([
                "| Category | Violations | Raw Total | Local Total | Avg Diff (%) | Max Diff (%) |\n",
                "|----------|------------|-----------|-------------|--------------|-------------|\n"
            ])
            
            for category in category_violations.index:
                violation_count = category_violations.loc[category, ('difference', 'count')]
                avg_diff = category_violations.loc[category, ('difference', 'mean')] * 100
                max_diff = category_violations.loc[category, ('difference', 'max')] * 100
                raw_total = category_violations.loc[category, (f'{metric_key}_raw', 'sum')]
                local_total = category_violations.loc[category, (f'{metric_key}_local', 'sum')]
                
                lines.append(
                    f"| **{category}** | {violation_count} | {raw_total:.2f} | "
                    f"{local_total:.2f} | {avg_diff:.2f}% | {max_diff:.2f}% |\n"
                )
            
            lines.append("\n")
            
        except Exception as e:
            lines.append(f"_Error generating category breakdown: {e}_\n\n")
        
        return "".join(lines)

    def _count_violations(self, gdf, thresholds):
        """Count the number of threshold violations."""
        
        if gdf.empty or 'difference' not in gdf.columns:
            return 0
        
        violation_count = 0
        for _, row in gdf.iterrows():
            category = row.get("category", "overall")
            difference = row.get("difference", 0)
            threshold = thresholds.get(category, thresholds.get("default", 0.1))
            
            if difference > threshold:
                violation_count += 1
        
        return violation_count

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
    
    if metric:
        # Validate single metric using spatial method
        if metric in validator.all_metrics:
            results = validator.cross_database_spatial_validation(metric)
            for raw_table, gdf in results.items():
                if not gdf.empty:
                    validator._export_spatial_results(gdf, raw_table, metric, dataset_type, region)
                    validator._generate_spatial_report(gdf, raw_table, metric, dataset_type, region)
        else:
            print_error(f"Metric '{metric}' not found in configuration")
            print_info(f"Available metrics: {', '.join(validator.all_metrics)}")
    else:
        # Validate all metrics using spatial method
        validator.validate_all_metrics_spatial(dataset_type, region)

if __name__ == "__main__":
    validate_poi()