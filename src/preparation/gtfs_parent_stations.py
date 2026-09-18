from src.config.config import Config
from src.core.config import settings
from src.db.db import Database
from src.utils.utils import print_info, timing


class GTFSParentStationsPreparation:
    """Identify and automatically fix missing parent stations in a GTFS dataset.

    Child stops (location_type != '1') that lack a parent_station are grouped into
    synthetic parent stations (location_type = '1') so that downstream steps such as
    stop_times_optimized and the station preparation can rely on consistent
    parent-child relationships. Two strategies are applied:

    * DHID stops: identifiers following the German "Deutschlandweite Haltestellen-ID"
      pattern (``<prefix>_de:<region>:<stop>[:...]``) are grouped by their DHID
      parent component.
    * Non-DHID stops: all remaining stops are grouped by their stop name.

    This must run before stop_times_optimized is produced (see GTFS preparation).
    """

    # DHID (Deutschlandweite Haltestellen-ID) country prefix used to detect and group
    # DHID-based stop identifiers. Stops that do not match are grouped by name instead.
    DHID_PREFIX = "de"

    def __init__(self, db: Database, region: str):
        self.db = db
        self.region = region
        self.config = Config("gtfs", region)
        self.schema = self.config.preparation["target_schema"]

    def _count(self, query: str) -> int:
        """Run a single-value COUNT query and return the result."""
        return self.db.select(query)[0][0]

    def report_issues(self, stage: str):
        """Report the number of outstanding parent station issues."""

        dhid_missing = self._count(f"""
            SELECT count(*) FROM {self.schema}.stops
            WHERE location_type != '1'
            AND parent_station IS NULL
            AND stop_id LIKE '%_{self.DHID_PREFIX}:%';
        """)

        non_dhid_missing = self._count(f"""
            SELECT count(*) FROM {self.schema}.stops
            WHERE location_type != '1'
            AND parent_station IS NULL
            AND stop_id NOT LIKE '%_{self.DHID_PREFIX}:%';
        """)

        # Stops whose parent_station references a non-existent stop_id
        orphan_refs = self._count(f"""
            SELECT count(*) FROM {self.schema}.stops s
            WHERE s.parent_station IS NOT NULL
            AND NOT EXISTS (
                SELECT 1 FROM {self.schema}.stops p WHERE p.stop_id = s.parent_station
            );
        """)

        print_info(
            f"Parent station issues ({stage} fix) in schema {self.schema}: "
            f"{dhid_missing} DHID stops without parent, "
            f"{non_dhid_missing} non-DHID stops without parent, "
            f"{orphan_refs} orphan parent_station references."
        )

    @timing
    def fix_dhid_parent_stations(self):
        """Create parent stations for DHID stops by extracting their DHID parent."""

        print_info("Producing parent stations for DHID stops...")
        sql_fix_dhid = f"""
            WITH grouped_stops AS (
                SELECT CASE WHEN parent_id = ANY(child_ids)
                        THEN parent_id || '_parent'
                        ELSE parent_id
                        END AS parent_id,
                    child_ids, child_names, num_child, parent_geom
                FROM (
                    SELECT parent_id.value AS parent_id, array_agg(stop_id) AS child_ids,
                        array_agg(stop_name) AS child_names, count(stop_id) AS num_child,
                        ST_Centroid(ST_Collect(geom)) AS parent_geom
                    FROM {self.schema}.stops,
                    LATERAL (
                        SELECT substring(stop_id from '.*_{self.DHID_PREFIX}:[^:]+:[^:]+') value
                    ) parent_id  -- Extract DHID
                    WHERE location_type != '1'  -- All stop types which are not a parent (1)
                    AND parent_station IS NULL
                    AND stop_id ~ '.*_{self.DHID_PREFIX}:[^:]+:[^:]+(:.*)?'  -- Valid DHID (+ synthetic ID)
                    GROUP BY parent_id.value
                ) sub
            ),
            create_parents AS (
                INSERT INTO {self.schema}.stops (
                    stop_id, stop_name, stop_lat, stop_lon, location_type, geom, h3_3
                )
                SELECT
                    g.parent_id,
                    g.child_names[1],
                    ST_Y(g.parent_geom),
                    ST_X(g.parent_geom),
                    '1',
                    g.parent_geom,
                    basic.to_short_h3_3(h3_lat_lng_to_cell(g.parent_geom::point, 3)::bigint)
                FROM grouped_stops g
                -- Skip parents that already exist; children are still re-pointed below
                WHERE NOT EXISTS (
                    SELECT 1 FROM {self.schema}.stops s WHERE s.stop_id = g.parent_id
                )
            ),
            update_parents AS (
                UPDATE {self.schema}.stops s
                SET parent_station = ps.parent_id
                FROM (
                    SELECT parent_id, unnest(child_ids) AS child_id
                    FROM grouped_stops
                ) ps
                WHERE s.stop_id = ps.child_id
            )
            -- Re-point stops still referencing the un-suffixed id to the new '_parent' id
            UPDATE {self.schema}.stops s
            SET parent_station = ps.stop_id
            FROM (
                SELECT stop_id FROM {self.schema}.stops WHERE stop_id LIKE '%_parent'
            ) ps
            WHERE s.parent_station = REPLACE(ps.stop_id, '_parent', '');
        """
        self.db.perform(sql_fix_dhid)

    @timing
    def fix_non_dhid_parent_stations(self):
        """Create parent stations for non-DHID stops by grouping on stop name."""

        print_info("Producing parent stations for non-DHID stops...")
        sql_fix_non_dhid = f"""
            WITH grouped_stops AS (
                SELECT stop_name || '_parent' AS parent_id, stop_name AS child_name,
                    count(*) AS num_child, ST_Centroid(ST_Collect(geom)) AS parent_geom
                FROM (
                    SELECT *
                    FROM {self.schema}.stops
                    WHERE location_type != '1'
                    AND stop_id NOT LIKE '%_{self.DHID_PREFIX}:%'
                    AND parent_station IS NULL
                    AND stop_name IS NOT NULL
                ) sub
                GROUP BY stop_name
            ),
            create_parents AS (
                INSERT INTO {self.schema}.stops (
                    stop_id, stop_name, stop_lat, stop_lon, location_type, geom, h3_3
                )
                SELECT
                    g.parent_id,
                    g.child_name,
                    ST_Y(g.parent_geom),
                    ST_X(g.parent_geom),
                    '1',
                    g.parent_geom,
                    basic.to_short_h3_3(h3_lat_lng_to_cell(g.parent_geom::point, 3)::bigint)
                FROM grouped_stops g
                WHERE NOT EXISTS (
                    SELECT 1 FROM {self.schema}.stops s WHERE s.stop_id = g.parent_id
                )
            )
            UPDATE {self.schema}.stops s
            SET parent_station = ps.parent_id
            FROM grouped_stops ps
            WHERE s.location_type != '1'
            AND s.stop_id NOT LIKE '%_{self.DHID_PREFIX}:%'
            AND s.parent_station IS NULL
            AND s.stop_name = ps.child_name;
        """
        self.db.perform(sql_fix_non_dhid)

    def run(self):
        """Run the parent station preparation."""

        print_info(f"Fixing GTFS parent stations in schema {self.schema}.")
        self.report_issues(stage="before")
        self.fix_dhid_parent_stations()
        self.fix_non_dhid_parent_stations()
        self.report_issues(stage="after")
        print_info("Preparation of GTFS parent stations is complete.")


def prepare_gtfs_parent_stations(region: str):
    print_info(f"Prepare GTFS parent stations for the region {region}.")
    db = Database(settings.LOCAL_DATABASE_URI)
    try:
        GTFSParentStationsPreparation(db=db, region=region).run()
        print_info("Finished GTFS parent stations preparation.")
    except Exception as e:
        print(e)
        raise e
    finally:
        db.close()
