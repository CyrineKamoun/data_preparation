import os
import sys
from datetime import datetime

import typer

from src.config.config import Config
from src.core.config import settings
from src.db.db import Database
from src.utils.utils import make_dir, print_hashtags, print_info, timing


class GTFSDemandRoutesPreparation:
    """Identify on-demand routes and flag those that are falsely scheduled.

    Demand-responsive services (AST / Rufbus / Ruftaxi / Anrufsammeltaxi / Bürgerbus /
    communal taxi, ...) are call-based, but feeds frequently model them with phantom
    fixed departures - often duplicate trips at identical times - which inflates the
    modelled frequency and skews accessibility analysis.

    The step works in three stages:

    1. Identify candidate on-demand routes by ``route_type`` and by name keywords.
    2. Measure frequency from ``stop_times_optimized`` (the resolved representative
       weekday/Saturday/Sunday schedule) per stop using a sliding one-hour window
       (arrivals in the window => implied headway of 60 / arrivals) and flag a route
       if, in ANY hour at ANY stop on ANY day-type, that headway drops below the
       plausible threshold ("falsely scheduled"). Measuring per hour means an hourly
       service modelled as near-simultaneous pairs is not mistaken for a high-frequency
       service; using stop_times_optimized means all concurrent service for the day is
       counted together rather than split across service_ids.
    3. Write a report document + a persistent issues table, then ask the operator to
       either reclassify ALL candidates to one canonical demand route_type (the default
       fix - normalises every on-demand service so it can be handled uniformly, e.g. a
       later step assigning a constant headway) or remove the falsely-scheduled subset,
       and apply the choice.

    Reuses the existing ``gtfs_<region>.yaml`` config (target_schema).
    """

    # Demand-responsive GTFS (extended) route types.
    DEMAND_ROUTE_TYPES = ("715", "1501")
    # Canonical demand route_type assigned to ALL on-demand routes when reclassifying,
    # so every demand-responsive service is treated uniformly (Demand & Response Bus).
    DEMAND_ROUTE_TYPE_TARGET = "715"

    # Name keywords for German demand-responsive services. AST/ALT/ALITA/RBL are
    # anchored to a word start (\m) so place names (e.g. "Astheim") do not match.
    KEYWORD_REGEX = (
        r"(\mAST([0-9 ]|\M))|Rufbus|Ruftaxi|Anrufsammeltaxi|Anrufbus|Anruf-Sammel|"
        r"Sammeltaxi|B(ü|ue)rgerbus|Rufauto|Taxibus|Linientaxi|Quartiersbus|Bedarfs|"
        r"on.?demand|\mALITA\M|\mALT\M|\mRBL\M"
    )

    # Length of the sliding window used to measure service frequency.
    FREQUENCY_WINDOW = "1 hour"
    # A route is "falsely scheduled" if, in ANY one-hour window at any stop on a
    # service day, its implied headway (60 / departures-in-window) drops below this.
    # An hourly service is fine; several departures within one hour is not.
    MAX_PLAUSIBLE_HEADWAY_MIN = 30

    def __init__(self, db: Database, region: str):
        self.db = db
        self.region = region
        self.config = Config("gtfs", region)
        self.schema = self.config.preparation["target_schema"]
        self.issues_table = f"{self.schema}.demand_route_issues"

    # ------------------------------------------------------------------ analysis

    @timing
    def analyze(self):
        """Detect on-demand routes, score their frequency, persist the issues table."""

        if not self.db.table_exists("stop_times_optimized", self.schema):
            raise RuntimeError(
                f"{self.schema}.stop_times_optimized not found. Run the gtfs preparation "
                "(prepare_gtfs) first - this step measures frequency from the resolved "
                "representative weekday/Saturday/Sunday schedule."
            )

        print_info("Identifying candidate on-demand routes (route_type + keywords)...")
        self.db.perform(f"""
            DROP TABLE IF EXISTS tmp_ondemand_routes;
            CREATE TEMP TABLE tmp_ondemand_routes AS
            SELECT r.route_id, r.route_short_name, r.route_type, a.agency_name,
                (r.route_type IN {self.DEMAND_ROUTE_TYPES}) AS by_type,
                (r.route_short_name ~ '{self.KEYWORD_REGEX}'
                 OR r.route_long_name ~ '{self.KEYWORD_REGEX}'
                 OR r.route_desc ~ '{self.KEYWORD_REGEX}') AS by_keyword
            FROM {self.schema}.routes r
            LEFT JOIN {self.schema}.agency a ON a.agency_id = r.agency_id
            WHERE r.route_type IN {self.DEMAND_ROUTE_TYPES}
               OR r.route_short_name ~ '{self.KEYWORD_REGEX}'
               OR r.route_long_name ~ '{self.KEYWORD_REGEX}'
               OR r.route_desc ~ '{self.KEYWORD_REGEX}';
            CREATE INDEX ON tmp_ondemand_routes (route_id);
        """)

        # stop_times_optimized is a Citus-distributed table (sharded on h3_3). A window
        # partitioned by route_id would force a worker-to-worker repartition, so first
        # pull the candidate rows to a local table via a plain filtered scan, then
        # compute the sliding-window frequency locally.
        print_info("Loading candidate rows from stop_times_optimized...")
        candidate_ids = [r[0] for r in self.db.select(
            "SELECT route_id FROM tmp_ondemand_routes;")]
        self.db.perform(f"""
            DROP TABLE IF EXISTS tmp_sto;
            CREATE TEMP TABLE tmp_sto AS
            SELECT route_id, stop_id, arrival_time, weekdays
            FROM {self.schema}.stop_times_optimized
            WHERE route_id = ANY(%s) AND arrival_time IS NOT NULL;
            CREATE INDEX ON tmp_sto (route_id);
        """, (candidate_ids,))

        print_info("Measuring peak hourly frequency per (stop, day-type)...")
        # Expand each row to the day-types it serves (weekdays[weekday, sat, sun]), then
        # slide a one-hour window per (route, stop, day-type) and count arrivals; the
        # implied headway is 60 / (arrivals in that hour). The peak hour is what matters.
        self.db.perform(f"""
            DROP TABLE IF EXISTS tmp_hourly;
            CREATE TEMP TABLE tmp_hourly AS
            WITH dep AS (
                SELECT route_id, stop_id, d.day_type, arrival_time
                FROM tmp_sto
                CROSS JOIN LATERAL (VALUES (1, 'weekday'), (2, 'saturday'), (3, 'sunday'))
                    AS d(idx, day_type)
                WHERE weekdays[d.idx] IS TRUE
            ),
            win AS (
                SELECT route_id, stop_id, day_type, arrival_time,
                    count(*) OVER (
                        PARTITION BY route_id, stop_id, day_type
                        ORDER BY arrival_time
                        RANGE BETWEEN CURRENT ROW AND INTERVAL '{self.FREQUENCY_WINDOW}' FOLLOWING
                    ) AS dep_in_window
                FROM dep
            )
            SELECT route_id, stop_id, day_type,
                count(*) AS n_dep,
                count(DISTINCT arrival_time) AS n_distinct,
                max(dep_in_window) AS peak_hourly_dep
            FROM win
            GROUP BY route_id, stop_id, day_type;
        """)

        print_info(f"Building issues table {self.issues_table}...")
        # Keep each route's busiest pattern (the one-hour window with the most departures).
        self.db.perform(f"""
            DROP TABLE IF EXISTS {self.issues_table};
            CREATE TABLE {self.issues_table} AS
            WITH worst_pattern AS (
                SELECT DISTINCT ON (o.route_id)
                    o.route_id, o.route_short_name, o.route_type, o.agency_name,
                    o.by_type, o.by_keyword,
                    h.stop_id AS busiest_stop_id, h.day_type AS busiest_day_type,
                    h.n_dep, h.n_distinct, h.peak_hourly_dep,
                    round(60.0 / h.peak_hourly_dep, 1) AS implied_headway_min
                FROM tmp_ondemand_routes o
                LEFT JOIN tmp_hourly h USING (route_id)
                ORDER BY o.route_id, h.peak_hourly_dep DESC NULLS LAST, h.n_dep DESC NULLS LAST
            )
            SELECT w.*,
                st.stop_name AS busiest_stop_name,
                (w.route_type NOT IN {self.DEMAND_ROUTE_TYPES}) AS mistyped,
                (w.peak_hourly_dep IS NOT NULL
                 AND 60.0 / w.peak_hourly_dep < {self.MAX_PLAUSIBLE_HEADWAY_MIN}) AS falsely_scheduled
            FROM worst_pattern w
            LEFT JOIN LATERAL (
                SELECT stop_name FROM {self.schema}.stops s
                WHERE s.stop_id = w.busiest_stop_id LIMIT 1
            ) st ON true;
        """)

    def summary(self) -> dict:
        """Return aggregate counts from the issues table."""
        row = self.db.select(f"""
            SELECT
                count(*) AS total_candidates,
                count(*) FILTER (WHERE by_type) AS by_type,
                count(*) FILTER (WHERE by_keyword) AS by_keyword,
                count(*) FILTER (WHERE falsely_scheduled) AS flagged,
                count(*) FILTER (WHERE falsely_scheduled AND mistyped) AS flagged_mistyped,
                count(*) FILTER (WHERE falsely_scheduled AND n_dep > n_distinct) AS flagged_duplicate
            FROM {self.issues_table};
        """)[0]
        flagged_trips = self.db.select(f"""
            SELECT count(*) FROM {self.schema}.trips t
            WHERE t.route_id IN (
                SELECT route_id FROM {self.issues_table} WHERE falsely_scheduled
            );
        """)[0][0]
        return {
            "total_candidates": row[0], "by_type": row[1], "by_keyword": row[2],
            "flagged": row[3], "flagged_mistyped": row[4],
            "flagged_duplicate": row[5], "flagged_trips": flagged_trips,
        }

    def write_report(self, summary: dict) -> str:
        """Write a Markdown problem report and return its path."""
        report_dir = os.path.join(settings.OUTPUT_DATA_DIR, "gtfs")
        make_dir(report_dir)
        report_path = os.path.join(report_dir, f"{self.schema}_demand_route_issues.md")

        flagged = self.db.select(f"""
            SELECT route_id, route_short_name, agency_name, route_type, mistyped,
                busiest_stop_name, busiest_day_type, peak_hourly_dep, implied_headway_min,
                n_dep, n_distinct
            FROM {self.issues_table}
            WHERE falsely_scheduled
            ORDER BY implied_headway_min ASC, peak_hourly_dep DESC;
        """)

        lines = [
            f"# GTFS demand-route issues - schema `{self.schema}`",
            "",
            f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}",
            "",
            "## Detection method",
            "",
            f"- On-demand route types: {', '.join(self.DEMAND_ROUTE_TYPES)} "
            "(Demand & Response Bus, Communal Taxi).",
            "- Name keywords: AST, Rufbus, Ruftaxi, Anrufsammeltaxi, Sammeltaxi, "
            "Bürgerbus, ALT, Linientaxi, Bedarf, ... (acronyms word-anchored).",
            "- Frequency source: stop_times_optimized (resolved representative "
            "weekday / Saturday / Sunday schedule).",
            f"- Falsely scheduled: in any one-hour window at any stop on any day-type, "
            f"the implied headway (60 / arrivals in that hour) drops below "
            f"{self.MAX_PLAUSIBLE_HEADWAY_MIN} min.",
            "",
            "## Summary",
            "",
            f"- Candidate on-demand routes: **{summary['total_candidates']}** "
            f"(by type: {summary['by_type']}, by keyword: {summary['by_keyword']}).",
            f"- Falsely scheduled (flagged): **{summary['flagged']}** "
            f"routes, **{summary['flagged_trips']}** trips.",
            f"- Of those flagged, mistyped as scheduled service: "
            f"**{summary['flagged_mistyped']}**; exhibiting duplicate departures "
            f"(more trips than distinct times): **{summary['flagged_duplicate']}**.",
            "",
            "Frequency is measured per one-hour window (peak arrivals in any hour) on "
            "the representative schedule, so an hourly service modelled as "
            "near-simultaneous pairs is not flagged.",
            "",
            "## Flagged routes",
            "",
            "| route_id | short_name | agency | type | mistyped | busiest_stop | day | peak_dep_per_hour | implied_headway_min | departures | distinct_times |",
            "|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for r in flagged:
            lines.append(
                f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {'yes' if r[4] else 'no'} "
                f"| {r[5]} | {r[6]} | {r[7]} | {r[8]} | {r[9]} | {r[10]} |"
            )
        lines.append("")

        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        return report_path

    # --------------------------------------------------------------- remediation

    @timing
    def reclassify_candidates(self) -> int:
        """Normalise every on-demand candidate to one canonical demand route_type.

        All demand-responsive services (AST, Rufbus, communal taxi, mistyped-as-bus, ...)
        are set to the same route_type so they are treated uniformly downstream - e.g. a
        later step can give them a constant headway. The falsely-scheduled frequency flag
        is left as documentation and is NOT acted on here. Both routes and (when present)
        stop_times_optimized are updated, since downstream analysis reads route_type from
        stop_times_optimized.
        """
        target = self.DEMAND_ROUTE_TYPE_TARGET
        total = self.db.select(f"SELECT count(*) FROM {self.issues_table};")[0][0]
        # issues_table.route_type is the snapshot of the original type, so it stays a
        # stable predicate even after routes.route_type is updated below.
        affected = self.db.select(f"""
            SELECT count(*) FROM {self.issues_table} WHERE route_type != '{target}';
        """)[0][0]

        self.db.perform(f"""
            UPDATE {self.schema}.routes r
            SET route_type = '{target}'
            FROM {self.issues_table} i
            WHERE r.route_id = i.route_id AND i.route_type != '{target}';
        """)

        # Keep stop_times_optimized.route_type consistent (downstream reads it from there).
        if self.db.table_exists("stop_times_optimized", self.schema):
            reclassified_ids = [r[0] for r in self.db.select(
                f"SELECT route_id FROM {self.issues_table} WHERE route_type != '{target}';")]
            if reclassified_ids:
                self.db.perform(f"""
                    UPDATE {self.schema}.stop_times_optimized
                    SET route_type = {int(target)}
                    WHERE route_id = ANY(%s);
                """, (reclassified_ids,))

        print_info(
            f"Reclassified {affected} of {total} on-demand candidate route(s) to "
            f"route_type '{target}' (routes + stop_times_optimized). "
            "All on-demand services now share one type."
        )
        return affected

    @timing
    def remove_flagged(self) -> int:
        """Delete flagged routes and their trips/stop_times from the dataset."""
        affected = self.db.select(
            f"SELECT count(*) FROM {self.issues_table} WHERE falsely_scheduled;"
        )[0][0]

        # stop_times_optimized only exists once prepare_gtfs has run.
        if self.db.table_exists("stop_times_optimized", self.schema):
            self.db.perform(f"""
                DELETE FROM {self.schema}.stop_times_optimized sto
                USING {self.issues_table} i
                WHERE sto.route_id = i.route_id AND i.falsely_scheduled;
            """)

        self.db.perform(f"""
            WITH flagged AS (
                SELECT route_id FROM {self.issues_table} WHERE falsely_scheduled
            ),
            del_stop_times AS (
                DELETE FROM {self.schema}.stop_times st
                USING (
                    SELECT t.trip_id FROM {self.schema}.trips t
                    JOIN flagged f ON f.route_id = t.route_id
                ) tr
                WHERE st.trip_id = tr.trip_id
            ),
            del_trips AS (
                DELETE FROM {self.schema}.trips t
                USING flagged f WHERE t.route_id = f.route_id
            )
            DELETE FROM {self.schema}.routes r
            USING flagged f WHERE r.route_id = f.route_id;
        """)
        print_info(f"Removed {affected} falsely-scheduled route(s) and their trips/stop_times.")
        return affected

    # -------------------------------------------------------------------- driver

    def _choose_action(self) -> str:
        """Ask the operator how to remediate flagged routes."""
        # Avoid hanging in non-interactive runs (cron, headless); allow env override.
        override = os.environ.get("GTFS_DEMAND_ROUTE_ACTION")
        if override:
            print_info(f"Using GTFS_DEMAND_ROUTE_ACTION='{override}'.")
            return override.strip().lower()
        if not sys.stdin.isatty():
            print_info(
                "Non-interactive session: skipping remediation. Review the report and "
                "re-run in a terminal, or set GTFS_DEMAND_ROUTE_ACTION=reclassify|remove."
            )
            return "skip"
        return typer.prompt(
            f"Action? [reclassify = retype ALL on-demand to {self.DEMAND_ROUTE_TYPE_TARGET}"
            " / remove = delete falsely-scheduled / skip]",
            default="skip",
        ).strip().lower()

    def run(self):
        """Run analysis, report, prompt and remediate."""
        print_info(f"Scanning GTFS demand routes in schema {self.schema}.")
        self.analyze()
        summary = self.summary()

        print_hashtags()
        print_info(
            f"On-demand candidates: {summary['total_candidates']} | "
            f"falsely scheduled: {summary['flagged']} routes "
            f"({summary['flagged_trips']} trips), "
            f"{summary['flagged_mistyped']} mistyped, "
            f"{summary['flagged_duplicate']} with duplicate departures."
        )
        report_path = self.write_report(summary)
        print_info(f"Wrote problem report to {report_path}")
        print_info(f"Inspectable issues table: {self.issues_table} (falsely_scheduled = true).")
        print_hashtags()

        if summary["total_candidates"] == 0:
            print_info("No on-demand routes found. Nothing to remediate.")
            return

        action = self._choose_action()
        if action == "reclassify":
            self.reclassify_candidates()
        elif action == "remove":
            self.remove_flagged()
        else:
            print_info("Skipping remediation. Routes left unchanged.")

        print_info("GTFS demand-route preparation is complete.")


def prepare_gtfs_demand_routes(region: str):
    print_info(f"Prepare GTFS demand routes for the region {region}.")
    db = Database(settings.LOCAL_DATABASE_URI)
    try:
        GTFSDemandRoutesPreparation(db=db, region=region).run()
        print_info("Finished GTFS demand-route preparation.")
    except Exception as e:
        print(e)
        raise e
    finally:
        db.close()
