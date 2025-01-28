from src.config.config import Config
from src.core.config import settings
from src.db.db import Database
from src.utils.utils import create_table_dump, print_info, timing


class GTFS:

    def __init__(self, db: Database, region: str):
        self.db = db
        self.region = region
        self.config = Config("gtfs", region)
        self.small_bulk = 100
        self.large_bulk = 10000
        self.schema = self.config.preparation["target_schema"]


    @timing
    def prepare_shape_dist_region(self):
        """Prepare distance travelled of the gtfs shapes per specified region."""

        # Create result table
        sql_create_shape_dist_region = f"""
        DROP TABLE IF EXISTS {self.schema}.shape_dist_region;
        CREATE TABLE {self.schema}.shape_dist_region
        (
            region_id TEXT,
            shape_id TEXT,
            shape_dist_traveled float,
            h3_3 integer
        );
        SELECT create_distributed_table('{self.schema}.shape_dist_region', 'h3_3');
        """
        self.db.perform(sql_create_shape_dist_region)

        # Get region ids and names
        regions = self.db.select(self.config.preparation["regions_query"])
        cnt_regions = len(regions)
        cnt = 0

        # Loop through regions and calculate shape_dist_traveled
        for region in regions:
            cnt += 1
            id, name = region[0], region[1]

            # Create temporary table with region
            sql_create_table_region = f"""DROP TABLE IF EXISTS region_subdivided;
            CREATE TABLE region_subdivided
            (
                id TEXT,
                name TEXT,
                geom geometry,
                h3_3 integer
            );
            SELECT create_distributed_table('public.region_subdivided', 'h3_3');

            INSERT INTO region_subdivided
            WITH region AS
            (
                SELECT * FROM public.nuts
                WHERE nuts_id = '{id}' AND nuts_name = '{name}'
            )
            ,border_points AS
            (
                SELECT ((ST_DUMPPOINTS(geom)).geom)::point AS geom
                FROM region
            ),
            h3_ids AS
            (
                SELECT DISTINCT basic.to_short_h3_3(h3_lat_lng_to_cell(geom, 3)::bigint) AS h3_3,
                ST_SETSRID(h3_cell_to_boundary(h3_lat_lng_to_cell(geom, 3))::geometry, 4326) AS geom
                FROM border_points
            )
            SELECT r.nuts_id, r.nuts_name, ST_SUBDIVIDE(ST_Intersection(h.geom, r.geom), 20) AS geom, h.h3_3
            FROM h3_ids h, region r;
            CREATE INDEX ON region_subdivided USING GIST(h3_3, geom);
            """
            self.db.perform(sql_create_table_region)

            sql_get_shape_dist_traveled = f"""INSERT INTO {self.schema}.shape_dist_region(region_id, shape_id, shape_dist_traveled, h3_3)
            SELECT n.id, s.shape_id, max(shape_dist_traveled) - min(shape_dist_traveled) AS shape_dist_traveled, n.h3_3
            FROM {self.schema}.shapes s, public.region_subdivided n
            WHERE ST_Intersects(s.geom, n.geom)
            AND s.h3_3 = n.h3_3
            GROUP BY n.h3_3, n.id, shape_id;
            DROP TABLE IF EXISTS region_subdivided; """
            self.db.perform(sql_get_shape_dist_traveled)

            print_info(f"Finished processing region {name}. There are {cnt} out of {cnt_regions} regions processed.")

        # Create index
        self.db.perform(f"CREATE INDEX ON {self.schema}.shape_dist_region (h3_3, shape_id);")
        self.db.perform(f"CREATE INDEX ON {self.schema}.shape_dist_region (h3_3, region_id);")

    @timing
    def prepare_stop_times(self):
        """Prepare stop_times table."""

        # Create undistributed shape_dist_region table
        self.db.perform(
            f"""
            DROP TABLE IF EXISTS {self.schema}.undistributed_shape_dist_region;
            CREATE TABLE {self.schema}.undistributed_shape_dist_region AS
            SELECT *
            FROM {self.schema}.shape_dist_region;
            CREATE INDEX ON {self.schema}.undistributed_shape_dist_region (shape_id);
            """
        )

        # Create result table
        sql_create_stop_times_optimized = f"""
            DROP TABLE IF EXISTS {self.schema}.stop_times_optimized;
            CREATE TABLE {self.schema}.stop_times_optimized (
                id serial4 NOT NULL,
                trip_id text NULL,
                route_id text NULL,
                arrival_time interval NULL,
                stop_id text NULL,
                route_type smallint NULL,
                weekdays _bool NULL,
                h3_3 integer NOT NULL
            );
            SELECT create_distributed_table('{self.schema}.stop_times_optimized', 'h3_3');
        """
        self.db.perform(sql_create_stop_times_optimized)

        # Create helper columns in routes for loop
        sql_create_routes_helper = (
            f"""ALTER TABLE {self.schema}.routes ADD COLUMN IF NOT EXISTS loop_id serial;"""
        )
        self.db.perform(sql_create_routes_helper)

        # Get max loop_id from routes
        sql_get_max_loop_id = f"""SELECT MAX(loop_id) FROM {self.schema}.routes;"""
        max_loop_id = self.db.select(sql_get_max_loop_id)[0][0]

        # Run processing in batches of routes to avoid memory issues
        for i in range(0, max_loop_id, self.small_bulk):
            sql_get_date_with_max_trips = f"""
                DROP TABLE IF EXISTS {self.schema}.dates_max_trips;
                CREATE TABLE {self.schema}.dates_max_trips AS
                WITH date_series AS (
                    SELECT generate_series::date AS check_date_tue,
                        (generate_series + INTERVAL '4 days')::date AS check_date_sat,
                        (generate_series + INTERVAL '5 days')::date AS check_date_sun
                    FROM generate_series(DATE '{self.config.preparation["start_date"]}',
                                            DATE '{self.config.preparation["start_date"]}' +
                                            INTERVAL '{self.config.preparation["num_weeks"]} weeks',
                                            INTERVAL '7 days')
                ),
                trip_cnt AS
                (
                    SELECT
                        j.route_id,
                        s.check_date_tue, j.cnt_trips_tue,
                        s.check_date_sat, j.cnt_trips_sat,
                        s.check_date_sun, j.cnt_trips_sun
                    FROM date_series s
                    CROSS JOIN LATERAL
                    (
                        SELECT sub.*
                        FROM (
                            SELECT route_id from
                                {self.schema}.routes r
                            WHERE r.loop_id > {i} AND r.loop_id <= {i+self.small_bulk}
                        ) r
                        CROSS JOIN LATERAL
                        (
                            select route_id,
                                sum(cnt_trips_tue) cnt_trips_tue,
                                sum(cnt_trips_sat) cnt_trips_sat,
                                sum(cnt_trips_sun) cnt_trips_sun
                            from
                            (
                                    select
                                        sum(c.tuesday::integer) as cnt_trips_tue,
                                        0 as cnt_trips_sat,
                                        0 as cnt_trips_sun,
                                        t.route_id
                                    FROM {self.schema}.trips t,
                                        (
                                            select c.*, cd.exception_type from
                                                {self.schema}.calendar c
                                            left outer join
                                                {self.schema}.calendar_dates cd
                                            on cd.service_id = c.service_id
                                                and cd.date = s.check_date_tue
                                        ) c
                                    WHERE t.route_id = r.route_id
                                    AND t.service_id = c.service_id
                                    and (c.exception_type is null or c.exception_type != 2)
                                    AND s.check_date_tue >= start_date
                                    AND s.check_date_tue <= end_date
                                    GROUP BY t.route_id
                                UNION ALL
                                    SELECT
                                        0 as cnt_trips_tue,
                                        sum(c.saturday::integer) as cnt_trips_sat,
                                        0 as cnt_trips_sun,
                                        t.route_id
                                    FROM {self.schema}.trips t,
                                        (
                                            select c.*, cd.exception_type from
                                                {self.schema}.calendar c
                                            left outer join
                                                {self.schema}.calendar_dates cd
                                            on cd.service_id = c.service_id
                                                and cd.date = s.check_date_sat
                                        ) c
                                    WHERE t.route_id = r.route_id
                                    AND t.service_id = c.service_id
                                    and (c.exception_type is null or c.exception_type != 2)
                                    AND s.check_date_sat >= start_date
                                    AND s.check_date_sat <= end_date
                                    GROUP BY t.route_id
                                union all
                                    select
                                        0 as cnt_trips_tue,
                                        0 as cnt_trips_sat,
                                        sum(c.sunday::integer) as cnt_trips_sun,
                                        t.route_id
                                    FROM {self.schema}.trips t,
                                        (
                                            select c.*, cd.exception_type from
                                                {self.schema}.calendar c
                                            left outer join
                                                {self.schema}.calendar_dates cd
                                            on cd.service_id = c.service_id
                                                and cd.date = s.check_date_sun
                                        ) c
                                    WHERE t.route_id = r.route_id
                                    AND t.service_id = c.service_id
                                    and (c.exception_type is null or c.exception_type != 2)
                                    AND s.check_date_sun >= start_date
                                    AND s.check_date_sun <= end_date
                                    GROUP BY t.route_id
                                UNION ALL
                                    SELECT
                                        sum(cd.exception_type) cnt_trips_tue,
                                        0 as cnt_trips_sat,
                                        0 as cnt_trips_sun,
                                        t.route_id
                                    FROM {self.schema}.trips t, {self.schema}.calendar_dates cd
                                    WHERE t.route_id = r.route_id
                                    AND t.service_id = cd.service_id
                                    AND cd.exception_type = 1
                                    AND cd.date = s.check_date_tue
                                    GROUP BY t.route_id
                                UNION ALL
                                    SELECT
                                        0 as cnt_trips_tue,
                                        sum(cd.exception_type) as cnt_trips_sat,
                                        0 as cnt_trips_sun,
                                        t.route_id
                                    FROM {self.schema}.trips t, {self.schema}.calendar_dates cd
                                    WHERE t.route_id = r.route_id
                                    AND t.service_id = cd.service_id
                                    AND cd.exception_type = 1
                                    AND cd.date = check_date_sat
                                    GROUP BY t.route_id
                                UNION ALL
                                    SELECT
                                        0 as cnt_trips_tue,
                                        0 as cnt_trips_sat,
                                        sum(cd.exception_type) cnt_trips_sun,
                                        t.route_id
                                    FROM {self.schema}.trips t, {self.schema}.calendar_dates cd
                                    WHERE t.route_id = r.route_id
                                    AND t.service_id = cd.service_id
                                    AND cd.exception_type = 1
                                    AND cd.date = check_date_sun
                                    GROUP BY t.route_id
                            ) route_trips
                            group by route_id
                        ) sub
                        where sub.cnt_trips_tue > 0 or sub.cnt_trips_sat > 0 or sub.cnt_trips_sun > 0
                    ) j
                ),
                route_mode_trips AS (
                    SELECT
                        r.route_id,
                        MODE() WITHIN GROUP (ORDER BY cnt_trips_tue) AS mode_trips_tue,
                        MODE() WITHIN GROUP (ORDER BY cnt_trips_sat) AS mode_trips_sat,
                        MODE() WITHIN GROUP (ORDER BY cnt_trips_sun) AS mode_trips_sun
                    FROM (
                        SELECT DISTINCT route_id
                        FROM trip_cnt
                    ) r
                    LEFT JOIN trip_cnt t ON r.route_id = t.route_id
                    GROUP BY r.route_id
                ),
                date_tue_mode_trips AS (
                    select rmt.route_id, rmt.mode_trips_tue, min(tc.check_date_tue) as mode_date_tue
                    from
                    route_mode_trips rmt join trip_cnt tc
                    on tc.route_id = rmt.route_id and tc.cnt_trips_tue = rmt.mode_trips_tue
                    group by rmt.route_id, rmt.mode_trips_tue

                ),
                date_sat_mode_trips AS (
                    select rmt.route_id, rmt.mode_trips_sat, min(tc.check_date_sat) as mode_date_sat
                    from
                    route_mode_trips rmt join trip_cnt tc
                    on tc.route_id = rmt.route_id and tc.cnt_trips_sat = rmt.mode_trips_sat
                    group by rmt.route_id, rmt.mode_trips_sat

                ),
                date_sun_mode_trips AS (
                    select rmt.route_id, rmt.mode_trips_sun, min(tc.check_date_sun) as mode_date_sun
                    from
                    route_mode_trips rmt join trip_cnt tc
                    on tc.route_id = rmt.route_id and tc.cnt_trips_sun = rmt.mode_trips_sun
                    group by rmt.route_id, rmt.mode_trips_sun

                )
                SELECT r.*,
                    d_tue.mode_trips_tue, d_tue.mode_date_tue as date_tue,
                    d_sat.mode_trips_sat, d_sat.mode_date_sat as date_sat,
                    d_sun.mode_trips_sun, d_sun.mode_date_sun as date_sun
                FROM date_tue_mode_trips d_tue,
                    date_sat_mode_trips d_sat,
                    date_sun_mode_trips d_sun,
                    {self.schema}.routes r
                WHERE d_tue.route_id = r.route_id
                    and d_sat.route_id = r.route_id
                    and d_sun.route_id = r.route_id;
            """

            self.db.perform(sql_get_date_with_max_trips)
            self.db.perform(f"CREATE INDEX ON {self.schema}.dates_max_trips (route_id);")

            # Select relevant trips with relevant route information and save them into a new table
            sql_create_trips_weekday = f"""DROP TABLE IF EXISTS {self.schema}.temp_trips_weekday;
            CREATE TABLE {self.schema}.temp_trips_weekday AS
            WITH t AS (
                SELECT t.trip_id, t.service_id, t.shape_id, t.trip_headsign, r.*
                FROM {self.schema}.trips t
                INNER JOIN {self.schema}.dates_max_trips r ON t.route_id = r.route_id
            ),
            cal AS (
                select c.*, cd.inactive_dates
                from {self.schema}.calendar c
                left outer join (
                    select service_id, array_agg("date") as inactive_dates
                    from {self.schema}.calendar_dates where exception_type = 2 group by service_id
                ) cd
                on cd.service_id = c.service_id
            )
            SELECT trip_id, route_id, service_id, route_type, trip_headsign, shape_id,
                ARRAY[CASE WHEN 'true' = ANY(array_agg(weekday)) THEN 'true'::boolean ELSE 'false'::boolean END,
                        CASE WHEN 'true' = ANY(array_agg(sat)) THEN 'true'::boolean ELSE 'false'::boolean END,
                        CASE WHEN 'true' = ANY(array_agg(sun)) THEN 'true'::boolean ELSE 'false'::boolean END] AS weekdays,
                ARRAY[CASE WHEN 'true' = ANY(array_agg(weekday)) THEN (ARRAY_AGG(date_tue))[1]::date ELSE NULL END,
                        CASE WHEN 'true' = ANY(array_agg(sat)) THEN (ARRAY_AGG(date_sat))[1]::date ELSE NULL END,
                        CASE WHEN 'true' = ANY(array_agg(sun)) THEN (ARRAY_AGG(date_sun))[1]::date ELSE NULL END] AS weekday_dates
            FROM (
                    SELECT t.*,
                        'true' as weekday,
                        'false' as sat,
                        'false' as sun
                    FROM t INNER JOIN cal
                    ON t.service_id = cal.service_id
                        AND t.date_tue >= cal.start_date
                        AND t.date_tue <= cal.end_date
                        AND cal.tuesday = '1'
                        AND (cal.inactive_dates is null or (NOT (t.date_tue = ANY (cal.inactive_dates))))
                UNION
                    SELECT t.*,
                        'true' as weekday,
                        'false' as sat,
                        'false' as sun
                    FROM t INNER JOIN {self.schema}.calendar_dates cd
                    ON t.service_id = cd.service_id
                        AND t.date_tue = cd.date
                        AND cd.exception_type = 1
                UNION
                    SELECT t.*,
                        'false' as weekday,
                        'true' as sat,
                        'false' as sun
                    FROM t INNER JOIN cal
                    ON t.service_id = cal.service_id
                        AND t.date_sat >= cal.start_date
                        AND t.date_sat <= cal.end_date
                        AND cal.saturday = '1'
                        AND (cal.inactive_dates is null or (NOT (t.date_sat = ANY (cal.inactive_dates))))
                UNION
                    SELECT t.*,
                        'false' as weekday,
                        'true' as sat,
                        'false' as sun
                    FROM t INNER JOIN {self.schema}.calendar_dates cd
                    ON t.service_id = cd.service_id
                        AND t.date_sat = cd.date
                        AND cd.exception_type = 1
                UNION
                    SELECT t.*,
                        'false' as weekday,
                        'false' as sat,
                        'true' as sun
                    FROM t INNER JOIN cal
                    ON t.service_id = cal.service_id
                        AND t.date_sun >= cal.start_date
                        AND t.date_sun <= cal.end_date
                        AND cal.sunday = '1'
                        AND (cal.inactive_dates is null or (NOT (t.date_sun = ANY (cal.inactive_dates))))
                UNION
                    SELECT t.*,
                        'false' as weekday,
                        'false' as sat,
                        'true' as sun
                    FROM t INNER JOIN {self.schema}.calendar_dates cd
                    ON t.service_id = cd.service_id
                        AND t.date_sun = cd.date
                        AND cd.exception_type = 1
            ) trips_combined
            GROUP BY trip_id, route_id, service_id, route_type, trip_headsign, shape_id;
            ALTER TABLE {self.schema}.temp_trips_weekday ADD COLUMN id serial;
            ALTER TABLE {self.schema}.temp_trips_weekday ADD PRIMARY KEY (id);
            CREATE INDEX ON {self.schema}.temp_trips_weekday (trip_id);
            CREATE INDEX ON {self.schema}.temp_trips_weekday (shape_id);"""
            self.db.perform(sql_create_trips_weekday)

            # Create distributed table for temp_trips_weekday
            sql_create_temp_trips_weekday_distributed = f"""
                DROP TABLE IF EXISTS {self.schema}.temp_trips_weekday_distributed;
                CREATE TABLE {self.schema}.temp_trips_weekday_distributed
                (
                    trip_id TEXT,
                    route_id TEXT,
                    route_type SMALLINT,
                    trip_headsign TEXT,
                    shape_id TEXT,
                    weekdays bool[],
                    h3_3 integer
                );
                SELECT create_distributed_table('{self.schema}.temp_trips_weekday_distributed', 'h3_3');
            """
            self.db.perform(sql_create_temp_trips_weekday_distributed)

            # Insert data into the distributed table
            sql_insert_temp_trips_weekday_distributed = f"""
                INSERT INTO {self.schema}.temp_trips_weekday_distributed
                SELECT t.trip_id, t.route_id, t.route_type::text::smallint, t.trip_headsign, t.shape_id, t.weekdays, j.h3_3
                FROM {self.schema}.temp_trips_weekday t
                CROSS JOIN LATERAL
                (
                    SELECT DISTINCT s.h3_3
                    FROM {self.schema}.undistributed_shape_dist_region s
                    WHERE t.shape_id = s.shape_id
                ) j;
                CREATE INDEX ON {self.schema}.temp_trips_weekday_distributed (h3_3, trip_id);
            """
            self.db.perform(sql_insert_temp_trips_weekday_distributed)

            # Create temporary table to be cleaned
            sql_create_stop_times_to_clean = f"""
                DROP TABLE IF EXISTS {self.schema}.stop_times_to_clean;
                CREATE TABLE {self.schema}.stop_times_to_clean AS
                SELECT st.trip_id, st.arrival_time, stop_id, route_type::text::smallint, weekdays, w.route_id, w.trip_headsign, st.h3_3
                FROM {self.schema}.stop_times st
                LEFT JOIN {self.schema}.temp_trips_weekday_distributed w
                ON st.trip_id = w.trip_id
                WHERE st.h3_3 = w.h3_3;
            """
            self.db.perform(sql_create_stop_times_to_clean)

            # Join stop_times with temp_trips_weekday_distributed and insert into stop_times_optimized
            sql_insert_stop_times_optimized = f"""
                INSERT INTO {self.schema}.stop_times_optimized(trip_id, route_id, stop_id, arrival_time, weekdays, route_type,  h3_3)
                SELECT (ARRAY_AGG(trip_id))[1], (ARRAY_AGG(route_id))[1], stop_id, arrival_time, weekdays, (ARRAY_AGG(route_type))[1],  h3_3
                FROM {self.schema}.stop_times_to_clean
                GROUP BY stop_id, arrival_time, weekdays, h3_3;
            """
            self.db.perform(sql_insert_stop_times_optimized)

            print_info(
                f"Finished processing routes {i} to {i+self.small_bulk} out of {max_loop_id}."
            )

        # Clean up temporary tables
        self.db.perform(f"DROP TABLE IF EXISTS {self.schema}.stop_times_to_clean;")
        self.db.perform(f"DROP TABLE IF EXISTS {self.schema}.dates_max_trips;")
        self.db.perform(f"DROP TABLE IF EXISTS {self.schema}.temp_trips_weekday;")
        self.db.perform(f"DROP TABLE IF EXISTS {self.schema}.temp_trips_weekday_distributed;")
        self.db.perform(f"DROP TABLE IF EXISTS {self.schema}.undistributed_shape_dist_region;")

    @timing
    def add_indices(self):
        """Add indices to the stop_times_optimized table."""

        # Creating indices one my one to monitor progress
        self.db.perform(f"""ALTER TABLE {self.schema}.stop_times_optimized ADD PRIMARY KEY (h3_3, id);""")
        print_info("Added primary key to stop_times_optimized.")
        self.db.perform(f"""CREATE INDEX ON {self.schema}.stop_times_optimized (h3_3, stop_id, arrival_time);""")
        print_info("Added index to stop_times_optimized (h3_3, stop_id, arrival_time).")
        self.db.perform(f"""CREATE INDEX ON {self.schema}.stop_times_optimized (h3_3, trip_id);""")
        print_info("Added index to stop_times_optimized (h3_3, trip_id).")


    def run(self):
        """Run the gtfs preparation."""

        self.prepare_shape_dist_region()
        self.prepare_stop_times()
        self.add_indices()


def prepare_gtfs(region: str):
    print_info(f"Prepare GTFS data for the region {region}.")
    db = Database(settings.LOCAL_DATABASE_URI)
    #db_rd = Database(settings.RAW_DATABASE_URI)

    try:
        GTFS(db=db, region=region).run()
        db.close()
        print_info("Finished GTFS preparation.")
    except Exception as e:
        print(e)
        raise e
    finally:
        db.close()
