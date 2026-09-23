import os
import sys

import typer

from src.config.config import Config
from src.core.config import settings
from src.db.db import Database
from src.utils.utils import print_hashtags, print_info, timing


class GTFSCombine:
    """Merge several national GTFS feeds into one de-duplicated combined feed.

    Cross-border service appears in more than one national feed (e.g. a Swiss SBB
    train running into Germany is in both the CH and DE feeds). This step assigns each
    cross-border agency a "home country" and, when building the combined schema, keeps
    each agency's service from its home feed only - so cross-border lines are counted
    once. Station-level consolidation of co-located stops and a few normalisations follow.

    Non-destructive: the national source schemas are never modified; de-duplication is
    applied by filtered copy into the target schema (idempotent, rebuildable).

    Phase A (this module): classify agencies -> filtered union -> border/parent-station
    consolidation + normalisations. Phase B (trip-level overlap dedup) runs after
    stop_times_optimized is built - see GTFSCombineDedup below.
    """

    # Persistent, cross-run lookup of agency_name -> home country (kept outside the
    # target schema, which is dropped/rebuilt each run). Matched case-insensitively.
    LOOKUP_TABLE = "basic.gtfs_agency_country_lookup"

    # Route types remapped for the routing engine (R5): communal taxi -> regular bus.
    ROUTE_TYPE_REMAP = {"1500": "700", "1501": "700"}

    # Seed for the lookup table - the previously hand-curated agency -> home country
    # assignments. Names are matched case-insensitively; unseeded / unmatched agencies
    # are prompted for and written back, so the table grows over time.
    LOOKUP_SEED = [
        # --- Switzerland (CH) ---
        ("Alpbus Fournier", "CH"), ("Baselland Transport", "CH"),
        ("Basler Verkehrsbetriebe", "CH"), ("Bus Ostschweiz", "CH"), ("CGN SA", "CH"),
        ("Nyon-St-Cergue-Morez", "CH"), ("Nyon-St-Cergue-Morez Ersatzverkehr", "CH"),
        ("PostAuto AG", "CH"), ("SBB GmbH", "CH"), ("SBB GmbH (Grenzverkehr)", "CH"),
        ("SBB Infrastruktur AG Bahnersatz", "CH"),
        ("Schweiz. Schifffahrtsgesellschaft Untersee und Rhein AG", "CH"),
        ("Schweizerische Bodensee-Schifffahrt AG", "CH"),
        ("Schweizerische Bundesbahnen SBB", "CH"),
        ("Service d'automobiles TRN (rvt Auto)", "CH"),
        ("Solarfährbetrieb Thomas Geiger Reichenau", "CH"), ("THURBO", "CH"),
        ("Transports Publics Genevois", "CH"),
        ("Transports Publics de la Région Nyonnaise", "CH"),
        ("Transports de Martigny et Régions (mc)", "CH"),
        ("Transports de Martigny et Régions Ersatzverkehr", "CH"),
        ("Verkehrsbetriebe Schaffhausen", "CH"),
        ("Regionale Verkehrsbetriebe Schaffhausen", "CH"), ("SBB", "CH"),
        ("TMR SA", "CH"),
        # --- Germany (DE) ---
        ("Bodensee-Schiffsbetriebe GmbH", "DE"), ("DB Regio AG Baden-Württemberg", "DE"),
        ("FPLAN BOD RAB OMP", "DE"), ("FPLAN VHB SBP", "DE"), ("NeTS Planung DB", "DE"),
        ("SWEG Südwestdeutsche Landesverkehrs-GmbH", "DE"), ("Südbadenbus", "DE"),
        ("Verkehrsverbund Hegau-Bodensee", "DE"), ("Bus VERA", "DE"),
        ("DB Fernverkehr (Codesharing)", "DE"), ("DB Fernverkehr AG", "DE"),
        ("DB Regio AG Mitte", "DE"), ("DB Regio AG Mitte Region Südwest", "DE"),
        ("DB Regio Mitte", "DE"), ("DB ZugBus Regionalverkehr Alb-Bodensee", "DE"),
        ("EUROSTAR", "DE"), ("FlixBus-de", "DE"), ("FlixTrain-de", "DE"),
        ("KVS GmbH", "DE"), ("Landkreis Konstanz", "DE"), ("Lörrach", "DE"),
        ("Queichtal Nahverkehrsgesellschaft", "DE"), ("SBG-Villingen", "DE"),
        ("SBG-Waldshut", "DE"), ("Saarbahn GmbH", "DE"),
        ("Südwestdeutsche Verkehrs-AG", "DE"), ("VVB Völklinger Verkehrsbetriebe GmbH", "DE"),
        ("vlexx", "DE"), ("vlexx1", "DE"),
        ("DB ZugBus Regionalverkehr Alb-Bodensee GmbH", "DE"), ("Saar-Mobil GmbH", "DE"),
        ("Baron Reisen GmbH", "DE"),
        # --- France (FR) ---
        ("DistriBus", "FR"), ("DISTRIBUS", "FR"), ("Jacquet Autocars", "FR"),
        ("RDTAin", "FR"), ("Réseau de transports de l'Agglomération de Thonon", "FR"),
        ("Société Nationale des Chemins de fer Français", "FR"),
        ("Transports Publics de l'agglomération d'Annemasse", "FR"), ("Evian", "FR"),
        ("SIBRA", "FR"), ("OCEdefault", "FR"), ("REGION GRAND EST", "FR"),
        ("SNCF VOYAGEURS", "FR"), ("DB + BSB Züge", "FR"), ("SNCF", "FR"),
        # --- Austria (AT) / Liechtenstein (LI) / Italy (IT) - no own feed, kept ---
        ("Verkehrsverbund Vorarlberg", "AT"), ("NeTS Planung ÖBB", "AT"),
        ("Österreichische Bundesbahnen", "AT"),
        ("Verkehrsbetrieb LIECHTENSTEINmobil", "LI"), ("Bergbahnen Malbun AG", "LI"),
        ("Trenitalia", "IT"),
    ]

    def __init__(self, db: Database, region: str):
        self.db = db
        self.region = region
        self.config = Config("gtfs", region)
        self.prep = self.config.preparation
        self.schema = self.prep["target_schema"]
        self.feeds = self.prep["merge_feeds"]                 # [{country, schema}, ...]
        self.countries = [f["country"] for f in self.feeds]
        self.buffer_m = self.prep.get("stop_buffer_m", 200)

    # -------------------------------------------------------------- lookup table

    def ensure_lookup(self):
        """Create the persistent agency->country lookup and seed it if empty."""
        schema = self.LOOKUP_TABLE.split(".")[0]
        self.db.perform(f"""
            CREATE SCHEMA IF NOT EXISTS {schema};
            CREATE TABLE IF NOT EXISTS {self.LOOKUP_TABLE} (
                agency_name text PRIMARY KEY,
                agency_country text NOT NULL
            );
        """)
        if self.db.select(f"SELECT count(*) FROM {self.LOOKUP_TABLE};")[0][0] == 0:
            print_info(f"Seeding {self.LOOKUP_TABLE} with {len(self.LOOKUP_SEED)} agencies.")
            self.db.perform(
                f"INSERT INTO {self.LOOKUP_TABLE} (agency_name, agency_country) VALUES "
                + ",".join(["(%s, %s)"] * len(self.LOOKUP_SEED))
                + " ON CONFLICT (agency_name) DO NOTHING;",
                tuple(v for row in self.LOOKUP_SEED for v in row),
            )

    def _lookup_country(self, agency_name: str):
        r = self.db.select(
            f"SELECT agency_country FROM {self.LOOKUP_TABLE} WHERE lower(agency_name) = lower(%s);",
            (agency_name,),
        )
        return r[0][0] if r else None

    def _remember(self, agency_name: str, country: str):
        self.db.perform(
            f"""INSERT INTO {self.LOOKUP_TABLE} (agency_name, agency_country) VALUES (%s, %s)
                ON CONFLICT (agency_name) DO UPDATE SET agency_country = EXCLUDED.agency_country;""",
            (agency_name, country),
        )

    # --------------------------------------------------------------- classification

    def _cross_border_agencies(self, schema: str, country: str):
        """Agencies in `schema` with non-parent stops in ANOTHER MERGED country's territory.

        Only the other merged feeds matter: an agency reaching a non-merged country
        (e.g. a DE line into CZ) can't be a cross-feed duplicate, so it's ignored.
        """
        others = [c for c in self.countries if c != country]
        return self.db.select(f"""
            SELECT DISTINCT r.agency_id, a.agency_name
            FROM {schema}.stops st
            JOIN public.nuts n ON n.levl_code = 0 AND n.cntr_code = ANY(%s)
                             AND ST_Intersects(st.geom, n.geom)
            JOIN {schema}.stop_times stt ON stt.stop_id = st.stop_id
            JOIN {schema}.trips t ON t.trip_id = stt.trip_id
            JOIN {schema}.routes r ON r.route_id = t.route_id
            JOIN {schema}.agency a ON a.agency_id = r.agency_id
            WHERE st.location_type <> '1'
            ORDER BY a.agency_name;
        """, (others,))

    def classify(self) -> dict:
        """Return {schema: [agency_ids to drop]} - agencies whose home is another merged feed."""
        drop = {}
        for feed in self.feeds:
            schema, country = feed["schema"], feed["country"]
            print_info(f"Classifying cross-border agencies in {schema} ({country})...")
            candidates = self._cross_border_agencies(schema, country)
            to_drop = []
            for agency_id, agency_name in candidates:
                home = self._lookup_country(agency_name)
                if home is None:
                    home = self._prompt_country(agency_name, country)
                    if home is None:      # non-interactive & unresolved -> handled by caller
                        raise RuntimeError(
                            f"Unclassified cross-border agency '{agency_name}' in {schema}. "
                            f"Add it to {self.LOOKUP_TABLE} or run interactively."
                        )
                    self._remember(agency_name, home)
                # Drop only if it belongs to ANOTHER feed we are merging (sourced there).
                if home != country and home in self.countries:
                    to_drop.append(agency_id)
            drop[schema] = to_drop
            print_info(f"  {len(candidates)} cross-border agencies; {len(to_drop)} sourced from another feed (dropped from {country}).")
        return drop

    def _prompt_country(self, agency_name: str, feed_country: str):
        if not sys.stdin.isatty():
            return None
        opts = "/".join(self.countries)
        ans = typer.prompt(
            f"Home country for '{agency_name}' (appears in {feed_country} feed, extends abroad) "
            f"[{opts}/other]",
            default=feed_country,
        ).strip().upper()
        return ans or feed_country

    # ------------------------------------------------------------- build combined

    @timing
    def build_schema(self):
        """(Re)create the combined schema/tables (unique stop_id, no h3 in the PK)."""
        print_info(f"Creating schema {self.schema} and tables.")
        self.db.perform(f"""
            DROP SCHEMA IF EXISTS {self.schema} CASCADE;
            CREATE SCHEMA {self.schema};

            CREATE TABLE {self.schema}.agency (
                agency_id text NOT NULL, agency_name text NOT NULL, agency_url text NOT NULL,
                agency_timezone text NOT NULL, agency_lang text NULL, agency_phone text NULL,
                agency_fare_url text NULL, agency_email text NULL, dataset_source text NULL,
                CONSTRAINT agency_pkey PRIMARY KEY (agency_id)
            );
            CREATE TABLE {self.schema}.stops (
                stop_id text NOT NULL, stop_code text NULL, stop_name text NULL, stop_desc text NULL,
                stop_lat float4 NOT NULL, stop_lon float4 NOT NULL, zone_id text NULL, stop_url text NULL,
                location_type text NULL, parent_station text NULL, stop_timezone text NULL,
                wheelchair_boarding text NULL, level_id text NULL, platform_code text NULL,
                geom public.geometry(point, 4326) NULL, buffer public.geometry(polygon, 4326) NULL,
                h3_3 int4 NULL, dataset_source text NULL,
                CONSTRAINT stops_pkey PRIMARY KEY (stop_id)
            );
            CREATE INDEX ON {self.schema}.stops USING gist (geom);
            CREATE INDEX ON {self.schema}.stops USING btree (parent_station);

            CREATE TABLE {self.schema}.routes (
                route_id text NOT NULL, agency_id text NULL, route_short_name text NULL,
                route_long_name text NULL, route_desc text NULL, route_type text NOT NULL,
                route_url text NULL, route_color text NULL, route_text_color text NULL,
                route_sort_order int4 NULL, continuous_drop_off text NULL, continuous_pickup text NULL,
                dataset_source text NULL, CONSTRAINT routes_pkey PRIMARY KEY (route_id)
            );
            CREATE TABLE {self.schema}.calendar (
                service_id text NOT NULL, monday text NOT NULL, tuesday text NOT NULL,
                wednesday text NOT NULL, thursday text NOT NULL, friday text NOT NULL,
                saturday text NOT NULL, sunday text NOT NULL, start_date date NOT NULL,
                end_date date NOT NULL, dataset_source text NULL,
                CONSTRAINT calendar_pkey PRIMARY KEY (service_id)
            );
            CREATE TABLE {self.schema}.calendar_dates (
                service_id text NOT NULL, "date" date NOT NULL, exception_type int2 NOT NULL,
                dataset_source text NULL, CONSTRAINT calendar_dates_pkey PRIMARY KEY (service_id, date)
            );
            CREATE TABLE {self.schema}.stop_times (
                trip_id text NOT NULL, arrival_time interval NULL, departure_time interval NULL,
                stop_id text NOT NULL, stop_sequence int4 NOT NULL, stop_sequence_consec int4 NULL,
                stop_headsign text NULL, pickup_type text NULL, drop_off_type text NULL,
                shape_dist_traveled float4 NULL, timepoint text NULL, h3_3 int4 NULL
            );
            CREATE TABLE {self.schema}.trips (
                trip_id text NOT NULL, route_id text NOT NULL, service_id text NOT NULL,
                trip_headsign text NULL, trip_short_name text NULL, direction_id int4 NULL,
                shape_id text NULL, wheelchair_accessible text NULL, bikes_allowed text NULL,
                CONSTRAINT trips_pkey PRIMARY KEY (trip_id)
            );
        """)

    @timing
    def copy_feed(self, feed: dict, drop_agency_ids: list):
        """Copy one national feed into the combined schema, excluding dropped agencies."""
        s, c = feed["schema"], feed["country"]
        print_info(f"Copying {s} -> {self.schema} (dataset_source={c}, dropping {len(drop_agency_ids)} agencies).")
        p = (drop_agency_ids,)
        # agency / routes: exclude dropped agencies (and their routes)
        self.db.perform(f"""
            INSERT INTO {self.schema}.agency
            SELECT agency_id, agency_name, agency_url, agency_timezone, agency_lang,
                   agency_phone, agency_fare_url, agency_email, '{c}'
            FROM {s}.agency WHERE agency_id <> ALL(%s);
        """, p)
        self.db.perform(f"""
            INSERT INTO {self.schema}.routes
            SELECT route_id, agency_id, route_short_name, route_long_name, route_desc, route_type,
                   route_url, route_color, route_text_color, route_sort_order,
                   continuous_drop_off, continuous_pickup, '{c}'
            FROM {s}.routes WHERE agency_id <> ALL(%s);
        """, p)
        # trips / stop_times: only those on kept routes/trips
        self.db.perform(f"""
            INSERT INTO {self.schema}.trips
            SELECT t.trip_id, t.route_id, t.service_id, t.trip_headsign, t.trip_short_name,
                   t.direction_id, t.shape_id, t.wheelchair_accessible, t.bikes_allowed
            FROM {s}.trips t
            WHERE EXISTS (SELECT 1 FROM {self.schema}.routes r WHERE r.route_id = t.route_id);
        """)
        self.db.perform(f"""
            INSERT INTO {self.schema}.stop_times
            SELECT st.trip_id, st.arrival_time, st.departure_time, st.stop_id, st.stop_sequence,
                   st.stop_sequence_consec, st.stop_headsign, st.pickup_type, st.drop_off_type,
                   st.shape_dist_traveled, st.timepoint, st.h3_3
            FROM {s}.stop_times st
            WHERE EXISTS (SELECT 1 FROM {self.schema}.trips t WHERE t.trip_id = st.trip_id);
        """)
        # calendar / calendar_dates: copied whole (orphan services are harmless)
        self.db.perform(f"""
            INSERT INTO {self.schema}.calendar
            SELECT service_id, monday, tuesday, wednesday, thursday, friday, saturday, sunday,
                   start_date, end_date, '{c}' FROM {s}.calendar;
        """)
        self.db.perform(f"""
            INSERT INTO {self.schema}.calendar_dates
            SELECT service_id, "date", exception_type, '{c}' FROM {s}.calendar_dates;
        """)
        # stops: DISTINCT ON (stop_id) guards the source-side H3 duplicate bug
        self.db.perform(f"""
            INSERT INTO {self.schema}.stops
            SELECT DISTINCT ON (stop_id) stop_id, stop_code, stop_name, stop_desc, stop_lat, stop_lon,
                   zone_id, stop_url, location_type, parent_station, stop_timezone, wheelchair_boarding,
                   level_id, platform_code, geom, NULL::geometry, h3_3, '{c}'
            FROM {s}.stops ORDER BY stop_id;
        """)

    @timing
    def add_stop_times_fk(self):
        self.db.perform(f"""
            CREATE INDEX ON {self.schema}.stop_times USING btree (stop_id);
            CREATE INDEX ON {self.schema}.stop_times USING btree (trip_id);
            CREATE INDEX ON {self.schema}.trips USING btree (route_id);
            CREATE INDEX ON {self.schema}.trips USING btree (service_id);
            ALTER TABLE {self.schema}.routes ADD CONSTRAINT routes_agency_fk
                FOREIGN KEY (agency_id) REFERENCES {self.schema}.agency(agency_id);
            ALTER TABLE {self.schema}.trips ADD CONSTRAINT trips_route_fk
                FOREIGN KEY (route_id) REFERENCES {self.schema}.routes(route_id);
            ALTER TABLE {self.schema}.stop_times ADD CONSTRAINT stop_times_stop_fk
                FOREIGN KEY (stop_id) REFERENCES {self.schema}.stops(stop_id);
        """)

    # --------------------------------------------------------------- normalisation

    @timing
    def normalise(self):
        print_info(f"Buffering stops ({self.buffer_m} m) and merging co-located stations across feeds.")
        self.db.perform(f"""
            UPDATE {self.schema}.stops
            SET buffer = ST_Buffer(geom::geography, {self.buffer_m})::geometry;
        """)
        # Index the buffer geometry and refresh planner stats BEFORE the spatial
        # re-parent join. Right after a bulk load the tables have no statistics, so the
        # planner otherwise picks an O(n^2) plan for ST_Intersects(geom, buffer) and the
        # UPDATE runs for many hours. (Run manually - as the original SQL was - autovacuum
        # ANALYSEd between steps and hid this; a back-to-back automated run must do it.)
        self.db.perform(f"""
            CREATE INDEX IF NOT EXISTS stops_buffer_gist ON {self.schema}.stops USING gist (buffer);
            ANALYZE {self.schema}.stops;
        """)
        # Re-parent foreign platform stops onto the nearest host-country parent station
        # within its buffer. Host countries are tried in merge order and earlier wins
        # ties: a stop already re-parented to a prior host (its parent now belongs to a
        # different feed than the stop itself) is skipped, so a later host can't override.
        for feed in self.feeds:
            host = feed["country"]
            self.db.perform(f"""
                UPDATE {self.schema}.stops
                SET parent_station = sub.new_parent_station
                FROM (
                    SELECT s2.stop_id,
                        (array_agg(s1.stop_id ORDER BY ST_Distance(s2.geom::geography, s1.geom::geography)))[1]
                            AS new_parent_station
                    FROM public.nuts n, {self.schema}.stops s1, {self.schema}.stops s2
                    WHERE n.cntr_code = '{host}' AND n.levl_code = 0
                      AND s1.location_type = '1' AND s1.dataset_source = '{host}'
                      AND ST_Intersects(s1.geom, n.geom)
                      AND s2.location_type <> '1' AND s2.dataset_source <> '{host}'
                      AND ST_Intersects(s2.geom, s1.buffer)
                      AND NOT EXISTS (
                          SELECT 1 FROM {self.schema}.stops pp
                          WHERE pp.stop_id = s2.parent_station
                            AND pp.dataset_source <> s2.dataset_source
                      )
                    GROUP BY s2.stop_id
                ) sub
                WHERE stops.stop_id = sub.stop_id;
            """)

        print_info("Cleaning orphan stops, normalising route types, checking integrity.")
        # Collection artifact: the loader stamps the dataset id-prefix into parent_station
        # on rows that weren't completely empty, so some stations (location_type='1') carry
        # a bogus parent_station referencing nothing (notably ~all CH stations). A station
        # must never have a parent - null it out (this is the original CH fix, generalised
        # to every feed). Without it these are dangling refs that fail the integrity assert.
        self.db.perform(f"""
            UPDATE {self.schema}.stops
            SET parent_station = NULL
            WHERE location_type = '1' AND parent_station IS NOT NULL;
        """)
        # Orphan child stops (no stop_times), then childless parent stations.
        self.db.perform(f"""
            DELETE FROM {self.schema}.stops
            WHERE (location_type IS NULL OR location_type IN ('', '0'))
              AND NOT EXISTS (SELECT 1 FROM {self.schema}.stop_times st WHERE st.stop_id = stops.stop_id);
        """)
        self.db.perform(f"""
            DELETE FROM {self.schema}.stops parent_stations
            WHERE location_type = '1'
              AND NOT EXISTS (SELECT 1 FROM {self.schema}.stops s WHERE s.parent_station = parent_stations.stop_id)
              AND NOT EXISTS (SELECT 1 FROM {self.schema}.stop_times st WHERE st.stop_id = parent_stations.stop_id);
        """)
        # Communal taxi -> bus (R5)
        for src, tgt in self.ROUTE_TYPE_REMAP.items():
            self.db.perform(f"UPDATE {self.schema}.routes SET route_type = '{tgt}' WHERE route_type = '{src}';")

        # Integrity: no parent_station pointing at a missing stop_id.
        dangling = self.db.select(f"""
            SELECT count(*) FROM {self.schema}.stops s
            WHERE s.parent_station IS NOT NULL
              AND NOT EXISTS (SELECT 1 FROM {self.schema}.stops p WHERE p.stop_id = s.parent_station);
        """)[0][0]
        if dangling:
            raise RuntimeError(f"{dangling} stops reference a non-existent parent_station in {self.schema}.")
        print_info("Integrity OK: no dangling parent_station references.")

    # -------------------------------------------------------------------- driver

    def run(self):
        print_info(f"Combining feeds {self.countries} into {self.schema}.")
        self.ensure_lookup()
        drop = self.classify()
        print_hashtags()
        self.build_schema()
        for feed in self.feeds:
            self.copy_feed(feed, drop.get(feed["schema"], []))
        self.add_stop_times_fk()
        self.normalise()
        print_hashtags()
        print_info(
            f"Phase A done. Next: run prepare_gtfs for region '{self.region}' to build "
            "stop_times_optimized, then re-run gtfs_combine and choose Phase B."
        )


class GTFSCombineDedup:
    """Phase B of the merge: trip-level overlap de-duplication.

    Runs AFTER `prepare_gtfs` has built `<schema>.stop_times_optimized` on the combined
    feed. Two passes:

    1. Cross-source duplicate trips - the same physical trip listed in more than one
       national feed. Matched by (origin parent-station name, origin arrival_time,
       destination parent-station name, destination arrival_time), evaluated per
       day-type (weekdays[weekday/sat/sun]) so only trips running on the same day-type
       are compared. One trip per duplicate group is kept, the rest deleted.
    2. Hardcoded, feed-gated long-distance removals - specific cross-border long-distance
       services that survive the earlier passes (e.g. SBB TGV/TER, DB Fernverkehr trains
       reaching France), each gated on the relevant feeds being in the merge.

    Matching relies on Phase A having consolidated co-located parent stations, so the same
    physical station shares a name across feeds.
    """

    METADATA_TABLE = "temporal.trip_metadata"
    # Local (non-distributed) staging tables. stop_times_optimized is a Citus table
    # distributed by h3_3; ranking/filtering/deleting it by trip_id directly forces a
    # full cross-shard repartition that spills tens of GB to disk (per day-type). We
    # stage the needed rows/ids into these plain local tables so the heavy steps run as
    # ordinary local Postgres operations, and the distributed table is only ever touched
    # by a simple trip_id filter (broadcast to shards, no repartition).
    LOCAL_STOP_TIMES = "temporal.stop_times_day_local"
    DUP_TRIPS = "temporal.dup_trip_ids"
    LD_ROUTES = "temporal.ld_routes"
    LD_TRIPS = "temporal.ld_trip_ids"
    DAY_TYPES = [(1, "weekday"), (2, "saturday"), (3, "sunday")]

    def __init__(self, db: Database, region: str):
        self.db = db
        self.region = region
        self.config = Config("gtfs", region)
        self.prep = self.config.preparation
        self.schema = self.prep["target_schema"]
        self.countries = [f["country"] for f in self.prep["merge_feeds"]]

    def _trip_count(self) -> int:
        return self.db.select(f"SELECT count(*) FROM {self.schema}.trips;")[0][0]

    # ------------------------------------------------- cross-source duplicate trips

    def _build_trip_metadata(self, day_idx: int):
        """Origin/destination stop + arrival time per trip, for one day-type.

        Staged through a LOCAL (non-distributed) table. `stop_times_optimized` is a Citus
        table distributed by `h3_3`, but we must rank rows PARTITION BY `trip_id`. Running
        the window function on the distributed table forces Citus to repartition the whole
        table by trip_id - a tens-of-GB on-disk shuffle, repeated per day-type, that
        overruns the disk. Pulling the day's slice into a plain local table (a simple
        pushed-down scan, no shuffle) turns the ranking into an ordinary local sort.
        """
        self.db.perform(f"""
            CREATE SCHEMA IF NOT EXISTS temporal;
            DROP TABLE IF EXISTS {self.LOCAL_STOP_TIMES};
            CREATE TABLE {self.LOCAL_STOP_TIMES} AS
            SELECT trip_id, stop_id, arrival_time
            FROM {self.schema}.stop_times_optimized
            WHERE weekdays[{day_idx}] IS TRUE;
            CREATE INDEX ON {self.LOCAL_STOP_TIMES} (trip_id);
            ANALYZE {self.LOCAL_STOP_TIMES};
        """)
        self.db.perform(f"""
            DROP TABLE IF EXISTS {self.METADATA_TABLE};
            CREATE TABLE {self.METADATA_TABLE} AS
            WITH ranked AS (
                SELECT trip_id, stop_id, arrival_time,
                    ROW_NUMBER() OVER (PARTITION BY trip_id ORDER BY arrival_time ASC  NULLS LAST) AS rn_asc,
                    ROW_NUMBER() OVER (PARTITION BY trip_id ORDER BY arrival_time DESC NULLS LAST) AS rn_desc
                FROM {self.LOCAL_STOP_TIMES}
            )
            SELECT t.route_id, t.trip_id,
                o.stop_id AS o_stop_id, o.arrival_time AS o_arrival_time,
                d.stop_id AS d_stop_id, d.arrival_time AS d_arrival_time
            FROM {self.schema}.trips t
            JOIN ranked o ON t.trip_id = o.trip_id AND o.rn_asc  = 1
            JOIN ranked d ON t.trip_id = d.trip_id AND d.rn_desc = 1;
            CREATE INDEX ON {self.METADATA_TABLE} (trip_id);
            DROP TABLE IF EXISTS {self.LOCAL_STOP_TIMES};
        """)

    def _delete_overlaps(self):
        """Delete one trip from each cross-source (origin,dest) x (times) duplicate group.

        The duplicate set is computed from local tables only (metadata, stops, routes) and
        staged into a local table, then applied per target. stop_times_optimized is
        distributed, so it's cleared with a plain trip_id filter (recursively planned from
        the local id set, broadcast to shards - no repartition) instead of a single
        statement that mixes the distributed table with local CTEs.
        """
        self.db.perform(f"""
            DROP TABLE IF EXISTS {self.DUP_TRIPS};
            CREATE TABLE {self.DUP_TRIPS} AS
            SELECT (array_agg(DISTINCT m.trip_id))[1] AS trip_id
            FROM {self.METADATA_TABLE} m
            JOIN {self.schema}.stops so  ON so.stop_id  = m.o_stop_id
            JOIN {self.schema}.stops sd  ON sd.stop_id  = m.d_stop_id
            JOIN {self.schema}.stops pso ON pso.stop_id = so.parent_station
            JOIN {self.schema}.stops psd ON psd.stop_id = sd.parent_station
            JOIN {self.schema}.routes r  ON r.route_id  = m.route_id
            GROUP BY pso.stop_name, m.o_arrival_time, psd.stop_name, m.d_arrival_time
            HAVING count(DISTINCT r.dataset_source) > 1;
            CREATE INDEX ON {self.DUP_TRIPS} (trip_id);
        """)
        # Distributed table: filter delete (no repartition). Then the local tables.
        self.db.perform(f"""
            DELETE FROM {self.schema}.stop_times_optimized
            WHERE trip_id IN (SELECT trip_id FROM {self.DUP_TRIPS});
        """)
        self.db.perform(f"""
            DELETE FROM {self.schema}.stop_times
            WHERE trip_id IN (SELECT trip_id FROM {self.DUP_TRIPS});
        """)
        self.db.perform(f"""
            DELETE FROM {self.schema}.trips
            WHERE trip_id IN (SELECT trip_id FROM {self.DUP_TRIPS});
            DROP TABLE IF EXISTS {self.DUP_TRIPS};
        """)

    @timing
    def dedup_overlapping_trips(self):
        print_info("Removing cross-source duplicate trips (per day-type).")
        for day_idx, label in self.DAY_TYPES:
            before = self._trip_count()
            self._build_trip_metadata(day_idx)
            self._delete_overlaps()
            print_info(f"  {label}: removed {before - self._trip_count()} duplicate trips.")

    # ---------------------------------------------------- config-driven removals

    def _delete_routes(self, desc: str, routes_sql: str):
        """Cascade-delete a set of routes (+ their trips/stop_times) and report.

        Same distributed-safe staging as _delete_overlaps: the target routes and their
        trip_ids are collected into local tables, the distributed stop_times_optimized is
        cleared by a trip_id filter, then the local tables are cleaned.
        """
        self.db.perform(f"""
            DROP TABLE IF EXISTS {self.LD_ROUTES};
            CREATE TABLE {self.LD_ROUTES} AS {routes_sql};
            CREATE INDEX ON {self.LD_ROUTES} (route_id);
        """)
        n_routes = self.db.select(f"SELECT count(*) FROM {self.LD_ROUTES};")[0][0]
        before = self._trip_count()
        self.db.perform(f"""
            DROP TABLE IF EXISTS {self.LD_TRIPS};
            CREATE TABLE {self.LD_TRIPS} AS
            SELECT t.trip_id FROM {self.schema}.trips t
            JOIN {self.LD_ROUTES} rr ON t.route_id = rr.route_id;
            CREATE INDEX ON {self.LD_TRIPS} (trip_id);
        """)
        self.db.perform(f"""DELETE FROM {self.schema}.stop_times_optimized
            WHERE trip_id IN (SELECT trip_id FROM {self.LD_TRIPS});""")
        self.db.perform(f"""DELETE FROM {self.schema}.stop_times
            WHERE trip_id IN (SELECT trip_id FROM {self.LD_TRIPS});""")
        self.db.perform(f"""DELETE FROM {self.schema}.trips
            WHERE trip_id IN (SELECT trip_id FROM {self.LD_TRIPS});""")
        self.db.perform(f"""DELETE FROM {self.schema}.routes r USING {self.LD_ROUTES} rr
            WHERE r.route_id = rr.route_id;""")
        print_info(f"  {desc}: removed {n_routes} routes, {before - self._trip_count()} trips.")
        self.db.perform(f"DROP TABLE IF EXISTS {self.LD_TRIPS}; DROP TABLE IF EXISTS {self.LD_ROUTES};")

    @timing
    def remove_long_distance(self):
        """Specific cross-border long-distance duplicate removals.

        These are the same physical international train listed under different operators
        in each feed, truncated at the border - so the endpoint-based trip dedup above
        can't match them. Hardcoded for now, each gated on the relevant feeds being in
        the merge (a future segment-level matcher should make these unnecessary).
        """
        countries = set(self.countries)
        print_info("Applying cross-border long-distance removals.")

        # SBB-branded TGV / TER in the CH feed duplicate the French SNCF services.
        if "CH" in countries:
            self._delete_routes(
                "SBB TGV/TER (CH)",
                f"""SELECT r.route_id
                    FROM {self.schema}.routes r
                    JOIN {self.schema}.agency a ON a.agency_id = r.agency_id
                    WHERE r.dataset_source = 'CH'
                      AND a.agency_name = 'Schweizerische Bundesbahnen SBB'
                      AND r.route_short_name = ANY(ARRAY['TGV', 'TER'])""",
            )

        # DB Fernverkehr trains reaching France duplicate the FR long-distance services.
        if "DE" in countries and "FR" in countries:
            self._delete_routes(
                "DB Fernverkehr reaching FR (DE)",
                f"""SELECT DISTINCT r.route_id
                    FROM {self.schema}.stops s
                    JOIN public.nuts n ON n.cntr_code = 'FR' AND n.levl_code = '1'
                                     AND ST_Intersects(s.geom, n.geom)
                    JOIN {self.schema}.stop_times st ON st.stop_id = s.stop_id
                    JOIN {self.schema}.trips t ON t.trip_id = st.trip_id
                    JOIN {self.schema}.routes r ON r.route_id = t.route_id
                    JOIN {self.schema}.agency a ON a.agency_id = r.agency_id
                    WHERE s.dataset_source = 'DE' AND a.agency_name = 'DB Fernverkehr AG'""",
            )

    def run(self):
        if not self.db.table_exists("stop_times_optimized", self.schema):
            raise RuntimeError(
                f"{self.schema}.stop_times_optimized not found. Run prepare_gtfs for region "
                f"'{self.region}' first - Phase B de-duplicates from the resolved schedule."
            )
        print_info(f"Phase B: trip-level overlap de-duplication on {self.schema}.")
        print_hashtags()
        self.dedup_overlapping_trips()
        self.remove_long_distance()
        self.db.perform(
            f"DROP TABLE IF EXISTS {self.METADATA_TABLE};"
            f"DROP TABLE IF EXISTS {self.LOCAL_STOP_TIMES};"
            f"DROP TABLE IF EXISTS {self.DUP_TRIPS};"
            f"DROP TABLE IF EXISTS {self.LD_TRIPS};"
            f"DROP TABLE IF EXISTS {self.LD_ROUTES};"
        )
        print_hashtags()
        print_info("Finished Phase B.")


def _choose_phase() -> str:
    """Ask which phase to run. Phase A builds the combined feed; Phase B (after
    stop_times_optimized) does trip-level de-duplication."""
    override = os.environ.get("GTFS_COMBINE_PHASE", "").strip().upper()
    if override in ("A", "B"):
        print_info(f"Using GTFS_COMBINE_PHASE={override}.")
        return override
    if not sys.stdin.isatty():
        raise RuntimeError(
            "Non-interactive session: set GTFS_COMBINE_PHASE=A (build+normalise, before "
            "stop_times_optimized) or =B (trip dedup, after stop_times_optimized)."
        )
    ans = typer.prompt(
        "Which phase? [A = build + normalise combined feed (before stop_times_optimized) / "
        "B = trip-level dedup (after stop_times_optimized)]",
        default="A",
    ).strip().upper()
    if ans not in ("A", "B"):
        raise typer.Abort()
    return ans


def prepare_gtfs_combine(region: str):
    print_info(f"Combine GTFS feeds for region {region}.")
    phase = _choose_phase()
    db = Database(settings.LOCAL_DATABASE_URI)
    try:
        if phase == "A":
            print_info("Running PHASE A (build combined feed + normalise).")
            GTFSCombine(db=db, region=region).run()
            print_info(
                "Phase A done. Next: run prepare_gtfs (region combined) to build "
                "stop_times_optimized, then re-run this and choose Phase B."
            )
        else:
            print_info("Running PHASE B (trip-level de-duplication).")
            GTFSCombineDedup(db=db, region=region).run()
        print_info("Finished combining GTFS feeds.")
    except Exception as e:
        print(e)
        raise e
    finally:
        db.close()
