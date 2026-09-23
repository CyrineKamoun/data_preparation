DROP TYPE IF EXISTS output_segment CASCADE;
CREATE TYPE output_segment AS (
	id text, overture_id text, length_m float8, length_3857 float8,
	class_ text, subclass text, impedance_slope float8, impedance_slope_reverse float8,
	impedance_surface float8, coordinates_3857 json, maxspeed_forward integer,
	maxspeed_backward integer, "source" text, source_index integer,
	target text, target_index integer, geom public.geometry(linestring, 4326),
    h3_3 integer, h3_6 integer
);
