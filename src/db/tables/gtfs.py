class GtfsTables:
    def __init__(self, schema):
        self.schema = schema

    def sql_create_table(self) -> dict:
        """SQL queries to create GTFS tables."""

        sql_create_table_agency = f"""
            CREATE TABLE {self.schema}.agency (
                agency_id text NOT NULL,
                agency_name text NOT NULL,
                agency_url text NOT NULL,
                agency_timezone text NOT NULL,
                agency_lang text NULL,
                agency_phone text NULL,
                agency_fare_url text NULL,
                agency_email text NULL
            );
        """
        sql_create_table_stops = f"""
            CREATE TABLE {self.schema}.stops (
                stop_id text NOT NULL,
                stop_code text NULL,
                stop_name text NULL,
                stop_desc text NULL,
                stop_lat float4 NOT NULL,
                stop_lon float4 NOT NULL,
                zone_id text NULL,
                stop_url text NULL,
                location_type text NULL,
                parent_station text NULL,
                stop_timezone text NULL,
                wheelchair_boarding text NULL,
                level_id text NULL,
                platform_code text NULL,
                geom public.geometry(point, 4326) NULL,
                h3_3 int4 NULL
            );
        """

        sql_create_table_routes = f"""
            CREATE TABLE {self.schema}.routes (
                route_id text NOT NULL,
                agency_id text NULL,
                route_short_name text NULL,
                route_long_name text NULL,
                route_desc text NULL,
                route_type text NOT NULL,
                route_url text NULL,
                route_color text NULL,
                route_text_color text NULL,
                route_sort_order int4 NULL,
                continuous_drop_off text NULL,
                continuous_pickup text NULL
            );
        """

        sql_create_table_trips = f"""
            CREATE TABLE {self.schema}.trips (
                trip_id text NOT NULL,
                route_id text NOT NULL,
                service_id text NOT NULL,
                trip_headsign text NULL,
                trip_short_name text NULL,
                direction_id int4 NULL,
                block_id text NULL,
                shape_id text NULL,
                wheelchair_accessible text NULL,
                bikes_allowed text NULL
            );"""

        sql_create_table_stop_times = f"""
            CREATE TABLE {self.schema}.stop_times (
                trip_id text NOT NULL,
                arrival_time interval NULL,
                departure_time interval NULL,
                stop_id text NOT NULL,
                stop_sequence int4 NOT NULL,
                stop_sequence_consec int4 NULL,
                stop_headsign text NULL,
                pickup_type text NULL,
                drop_off_type text NULL,
                shape_dist_traveled float4 NULL,
                timepoint text NULL,
                h3_3 int4 NULL
            );"""

        sql_create_table_calendar = f"""
            CREATE TABLE {self.schema}.calendar (
                service_id text NOT NULL,
                monday text NOT NULL,
                tuesday text NOT NULL,
                wednesday text NOT NULL,
                thursday text NOT NULL,
                friday text NOT NULL,
                saturday text NOT NULL,
                sunday text NOT NULL,
                start_date date NOT NULL,
                end_date date NOT NULL
            );"""

        sql_create_table_calendar_dates = f"""
            CREATE TABLE {self.schema}.calendar_dates (
                service_id text NOT NULL,
                date date NOT NULL,
                exception_type int2 NOT NULL
            );"""

        sql_create_table_shapes = f"""
            CREATE TABLE {self.schema}.shapes (
                shape_id text NULL,
                shape_pt_lat float4 NOT NULL,
                shape_pt_lon float4 NOT NULL,
                shape_pt_sequence int4 NULL,
                geom public.geometry(point, 4326) NULL,
                shape_dist_traveled float4 NULL,
                h3_3 int4 NULL
            );
        """

        return {
            "agency": sql_create_table_agency,
            "stops": sql_create_table_stops,
            "routes": sql_create_table_routes,
            "trips": sql_create_table_trips,
            "stop_times": sql_create_table_stop_times,
            "calendar": sql_create_table_calendar,
            "calendar_dates": sql_create_table_calendar_dates,
            "shapes": sql_create_table_shapes
        }

    def sql_select_table(self) -> dict:
        """SQL queries to select data from GTFS tables."""

        sql_select_table_agency = f"""
            SELECT agency_id, agency_name, agency_url, agency_timezone, agency_lang, agency_phone, agency_fare_url
            FROM {self.schema}.agency
        """

        sql_select_table_stops = f"""
            SELECT stop_name, parent_station, stop_code, zone_id, stop_id, stop_desc, stop_lat, stop_lon,stop_url, location_type, stop_timezone, wheelchair_boarding, level_id, platform_code
            FROM {self.schema}.stops
        """

        sql_select_table_routes = f"""
            SELECT route_long_name, route_short_name, agency_id, route_desc, route_type, route_id, route_color, route_text_color, route_sort_order
            FROM {self.schema}.routes
        """

        sql_select_table_trips = f"""
            SELECT route_id, service_id, trip_headsign, trip_short_name, direction_id, block_id, shape_id, trip_id, wheelchair_accessible, bikes_allowed
            FROM {self.schema}.trips
        """

        sql_select_table_stop_times = f"""
            SELECT trip_id, arrival_time::text, departure_time::text, stop_id, stop_sequence, stop_headsign, pickup_type, drop_off_type, shape_dist_traveled, timepoint
            FROM {self.schema}.stop_times
        """

        sql_select_table_calendar = f"""
            SELECT monday, tuesday, wednesday, thursday, friday, saturday, sunday, REPLACE(start_date::text, '-', '') as start_date, REPLACE(end_date::text, '-', '') as end_date, service_id
            FROM {self.schema}.calendar
        """

        sql_select_table_calendar_dates = f"""
            SELECT service_id, exception_type, REPLACE(date::text, '-', '') as date
            FROM {self.schema}.calendar_dates
        """

        sql_select_table_shapes = f"""
            SELECT shape_id, shape_pt_lat, shape_pt_lon, shape_pt_sequence
            FROM {self.schema}.shapes
        """

        return {
            "agency": sql_select_table_agency,
            "stops": sql_select_table_stops,
            "routes": sql_select_table_routes,
            "trips": sql_select_table_trips,
            "stop_times": sql_select_table_stop_times,
            "calendar": sql_select_table_calendar,
            "calendar_dates": sql_select_table_calendar_dates,
            "shapes": sql_select_table_shapes
        }
