DROP TYPE IF EXISTS output_segment CASCADE;
CREATE TYPE output_segment AS (
	id text, overture_id text, length_m float8, length_3857 float8,
	class_ text, subclass text, impedance_slope float8, impedance_slope_reverse float8,
	impedance_surface float8, coordinates_3857 json, maxspeed_forward integer,
	maxspeed_backward integer, "source" text, source_index integer,
	target text, target_index integer, geom public.geometry(linestring, 4326),
    h3_3 integer, h3_6 integer
);


DROP FUNCTION IF EXISTS basic.classify_segment;
CREATE OR REPLACE FUNCTION basic.classify_segment(
	segment_id TEXT,
	cycling_surfaces JSONB,
	default_speed_limits JSONB
)
RETURNS VOID
AS $$
DECLARE
	input_segment record;
	new_sub_segment output_segment;
	output_segment output_segment;
	split_geometry record;

	sub_segments output_segment[] = '{}';
	output_segments output_segment[] = '{}';
	
	source_conn_location float;
	target_conn_location float;

	mph_kmph_conv_factor float = 1.60934;
	car_modes text[] = ARRAY['vehicle', 'motor_vehicle', 'car'];
	maxspeed_forward_list float[];
	maxspeed_backward_list float[];
	speed_limit jsonb;
	restriction jsonb;
	maxspeed_forward int;
	maxspeed_backward int;
BEGIN
	-- Select relevant input segment
	SELECT
		id,
		subtype,
		connectors::jsonb AS connectors,
		geometry,
		class,
		subclass,
		names::jsonb->>'primary' AS name,
		road_surface::jsonb AS road_surface,
		access_restrictions::jsonb AS access_restrictions,
		speed_limits::jsonb AS speed_limits
	INTO input_segment
	FROM temporal.segments
	WHERE id = segment_id;

	-- Skip this segment if the subtype is not road
	IF input_segment.subtype != 'road' THEN
		RETURN;
	END IF;

	-- Check if segment needs to be split into sub-segments
	IF jsonb_array_length(input_segment.connectors) > 2 THEN
		-- Split segment into sub-segments
		FOR i IN 1..(jsonb_array_length(input_segment.connectors) - 1) LOOP
			-- Initialize sub-segment primary properties
			new_sub_segment.id = input_segment.id || '_sub_' || i-1;
			SELECT ST_LineLocatePoint(input_segment.geometry, geometry) INTO source_conn_location FROM temporal.connectors WHERE id = (input_segment.connectors[i-1]->>'connector_id');
			SELECT ST_LineLocatePoint(input_segment.geometry, geometry) INTO target_conn_location FROM temporal.connectors WHERE id = (input_segment.connectors[i]->>'connector_id');
			
			-- Handle rare cases with invalid connector locations
			CONTINUE WHEN source_conn_location > target_conn_location;
			
			new_sub_segment.geom = ST_LineSubstring(
				input_segment.geometry,
				source_conn_location,
				target_conn_location
			);
			new_sub_segment.source = (input_segment.connectors[i-1]->>'connector_id');
			new_sub_segment.target = (input_segment.connectors[i]->>'connector_id');

			-- TODO Handle linear split surface for sub-segment
			-- TODO Handle linear split speed limits for sub-segment
			-- TODO Handle linear split flags for sub-segment

			sub_segments = array_append(sub_segments, new_sub_segment);
		END LOOP;
	ELSE
		-- Initialize segment primary properties
		new_sub_segment.id = input_segment.id;
		new_sub_segment.geom = input_segment.geometry;
		new_sub_segment.source = (input_segment.connectors[0]->>'connector_id');
		new_sub_segment.target = (input_segment.connectors[1]->>'connector_id');

		-- TODO Handle linear split surface for segment
		-- TODO Handle linear split speed limits for segment
		-- TODO Handle linear split flags for segment

		sub_segments = array_append(sub_segments, new_sub_segment);
	END IF;

	-- Process speed limits
	IF jsonb_array_length(input_segment.speed_limits) > 0 THEN
		FOR speed_limit IN SELECT * FROM jsonb_array_elements(input_segment.speed_limits) LOOP
			-- Speed limits may have directionality, check for this
			IF speed_limit ? 'when' AND (speed_limit->'when') ? 'heading' THEN
				IF ((speed_limit->'when')->>'heading') = 'forward' THEN
					-- Speed limit applies in the forward direction only
					IF ((speed_limit->'max_speed')->>'unit') = 'km/h' THEN
						maxspeed_forward_list := array_append(maxspeed_forward_list, ((speed_limit->'max_speed')->>'value')::float);
					ELSE
						maxspeed_forward_list := array_append(maxspeed_forward_list, ((speed_limit->'max_speed')->>'value')::float * mph_kmph_conv_factor);
					END IF;
				ELSE
					-- Speed limit applies in the backward direction only
					IF ((speed_limit->'max_speed')->>'unit') = 'km/h' THEN
						maxspeed_backward_list := array_append(maxspeed_backward_list,  ((speed_limit->'max_speed')->>'value')::float);
					ELSE
						maxspeed_backward_list := array_append(maxspeed_backward_list,  ((speed_limit->'max_speed')->>'value')::float * mph_kmph_conv_factor);
					END IF;
				END IF;
			ELSE
				-- Speed limit applies in both directions
				IF ((speed_limit->'max_speed')->>'unit') = 'km/h' THEN
					maxspeed_forward_list := array_append(maxspeed_forward_list, ((speed_limit->'max_speed')->>'value')::float);
					maxspeed_backward_list := array_append(maxspeed_backward_list,  ((speed_limit->'max_speed')->>'value')::float);
				ELSE
					maxspeed_forward_list := array_append(maxspeed_forward_list, ((speed_limit->'max_speed')->>'value')::float * mph_kmph_conv_factor);
					maxspeed_backward_list := array_append(maxspeed_backward_list,  ((speed_limit->'max_speed')->>'value')::float * mph_kmph_conv_factor);
				END IF;
			END IF;
		END LOOP;

		SELECT round(avg(value))::int
		INTO maxspeed_forward
		FROM unnest(maxspeed_forward_list) AS value;

		SELECT round(avg(value))::int
		INTO maxspeed_backward
		FROM unnest(maxspeed_backward_list) AS value;
	END IF;

	-- Set default speed limits if none were specified in the data
	IF maxspeed_forward IS NULL THEN
		maxspeed_forward = (default_speed_limits->>input_segment.class)::int;
	END IF;

	IF maxspeed_backward IS NULL THEN
		maxspeed_backward = (default_speed_limits->>input_segment.class)::int;
	END IF;

	-- Process access restrictions
	FOR restriction IN SELECT * FROM jsonb_array_elements(input_segment.access_restrictions) LOOP
		IF restriction->>'access_type' = 'denied' THEN
			-- Restrictions regarding access
			IF (restriction->'when') IS NULL THEN
				maxspeed_forward = NULL;
				maxspeed_backward = NULL;
				EXIT;
			END IF;

			-- Restrictions regarding the direction of travel
			IF (restriction->'when')->>'heading' = 'forward' THEN
				maxspeed_forward = NULL;
			ELSIF (restriction->'when')->>'heading' = 'backward' THEN
				maxspeed_backward = NULL;
			END IF;

			-- Restrictions regarding the type of vehicle
			IF (restriction->'when')->'mode' ?| car_modes THEN
				maxspeed_forward = NULL;
				maxspeed_backward = NULL;
				EXIT;
			END IF;
		-- Ignore these becuase they are not always defined consistently
		-- ELSIF restriction->>'access_type' = 'allowed' THEN
			-- Restrictions regarding the type of vehicle
		--	IF (restriction->'when') ? 'mode' AND NOT (restriction->'when')->'mode' ?| car_modes THEN
		--		maxspeed_forward = NULL;
		--		maxspeed_backward = NULL;
		--		EXIT;
		--	END IF;
		END IF;
	END LOOP;

	-- Clip sub-segments to fit into h3_3 and h3_6 cells
	SELECT basic.clip_segments(sub_segments, 6) INTO output_segments;
	SELECT basic.clip_segments(output_segments, 3) INTO output_segments;

	-- Loop through final output segments
	FOREACH output_segment IN ARRAY output_segments LOOP
		-- Set remaining properties for every output segment, these are derived from primary properties
		output_segment.overture_id = input_segment.id;
		output_segment.length_m = ST_Length(output_segment.geom::geography);
		output_segment.length_3857 = ST_Length(ST_Transform(output_segment.geom, 3857));
		output_segment.coordinates_3857 = ((ST_AsGeoJson(ST_Transform(output_segment.geom, 3857)))::jsonb)['coordinates'];
		output_segment.class_ = input_segment.class;
		output_segment.subclass = input_segment.subclass;
		output_segment.maxspeed_forward = maxspeed_forward;
		output_segment.maxspeed_backward = maxspeed_backward;
		output_segment.h3_3 = basic.to_short_h3_3(h3_lat_lng_to_cell(ST_Centroid(output_segment.geom)::point, 3)::bigint);
		output_segment.h3_6 = basic.to_short_h3_6(h3_lat_lng_to_cell(ST_Centroid(output_segment.geom)::point, 6)::bigint);

		-- Drop this segment if it isn't within the bounds of the h3_3 and h3_6 grids
		IF NOT EXISTS (SELECT 1 FROM basic.h3_6_grid WHERE h3_short = output_segment.h3_6)
			OR NOT EXISTS (SELECT 1 FROM basic.h3_3_grid WHERE h3_short = output_segment.h3_3) THEN
			CONTINUE;
		END IF;

		-- Temporarily set the following properties here, but eventually handle linear split values above
		IF jsonb_array_length(input_segment.road_surface) > 0 THEN
			output_segment.impedance_surface = (cycling_surfaces ->> (input_segment.road_surface[0]->>'value'))::float;
		END IF;

		-- Insert processed output segment data into table
        INSERT INTO basic.segment (
				overture_id, length_m, length_3857, class_,
				subclass, impedance_slope, impedance_slope_reverse,
				impedance_surface, coordinates_3857, maxspeed_forward,
				maxspeed_backward, source, target, geom, h3_3, h3_6
        )
        VALUES (
			output_segment.overture_id, output_segment.length_m, output_segment.length_3857, output_segment.class_,
			output_segment.subclass, output_segment.impedance_slope, output_segment.impedance_slope_reverse,
			output_segment.impedance_surface, output_segment.coordinates_3857, output_segment.maxspeed_forward,
			output_segment.maxspeed_backward, output_segment.source_index, output_segment.target_index,
			output_segment.geom, output_segment.h3_3, output_segment.h3_6
        );
    END LOOP;
END
$$
LANGUAGE plpgsql
PARALLEL SAFE;
