DROP FUNCTION IF EXISTS basic.clip_segments;
CREATE OR REPLACE FUNCTION basic.clip_segments(
	input_segments output_segment[],
	h3_resolution integer
)
RETURNS output_segment[]
AS $$
DECLARE
	new_sub_segment output_segment;
	output_segment output_segment;
	intersecting_cells record;
	split_geometry record;

	output_segments output_segment[] = '{}';
BEGIN

-- Loop through sub-segments and clip to H3 index region
FOREACH new_sub_segment IN ARRAY input_segments LOOP
	-- Find intersecting H3 index cells for the specified resolution
	IF h3_resolution = 3 THEN
		SELECT count(g.h3_short) AS count, ST_Union(g.h3_boundary) AS geom
		INTO intersecting_cells
		FROM basic.h3_3_grid g
		WHERE ST_Intersects(new_sub_segment.geom, g.h3_boundary);
	ELSIF h3_resolution = 6 THEN
		SELECT count(g.h3_short) AS count, ST_Union(g.h3_boundary) AS geom
		INTO intersecting_cells
		FROM basic.h3_6_grid g
		WHERE ST_Intersects(new_sub_segment.geom, g.h3_boundary);
	END IF;

	-- Check if segment extends further than H3 index boundary and splitting is necessary
	IF intersecting_cells.count = 0 THEN
		-- Splitting this segment is not necessary, just process its source/target connectors

		-- If source connector doesn't already exist in output table, insert it
		INSERT INTO basic.connector (id, osm_id, geom, h3_3, h3_6)
		SELECT id, NULL, geometry, basic.to_short_h3_3(h3_lat_lng_to_cell(geometry::point, 3)::bigint),
			basic.to_short_h3_6(h3_lat_lng_to_cell(geometry::point, 6)::bigint)
		FROM temporal.connectors
		WHERE id = new_sub_segment.source
		ON CONFLICT DO NOTHING;

		-- Get serial index of new or existing source connector
		SELECT index FROM basic.connector WHERE id = new_sub_segment.source
		INTO new_sub_segment.source_index;

		-- If target connector doesn't already exist in output table, insert it
		INSERT INTO basic.connector (id, osm_id, geom, h3_3, h3_6)
		SELECT id, NULL, geometry, basic.to_short_h3_3(h3_lat_lng_to_cell(geometry::point, 3)::bigint),
			basic.to_short_h3_6(h3_lat_lng_to_cell(geometry::point, 6)::bigint)
		FROM temporal.connectors
		WHERE id = new_sub_segment.target
		ON CONFLICT DO NOTHING;

		-- Get serial index of new or existing target connector
		SELECT index FROM basic.connector WHERE id = new_sub_segment.target
		INTO new_sub_segment.target_index;

		output_segments = array_append(output_segments, new_sub_segment);
	ELSE
		-- Loop through split segment geometry to create final segments & connectors
		FOR split_geometry IN SELECT * FROM (
			WITH split_segment AS (
				SELECT (ST_Dump(ST_Split(new_sub_segment.geom, intersecting_cells.geom))).geom
			)
			SELECT
				ROW_NUMBER() OVER () AS row_index,
				sc.count AS row_count,
				ST_StartPoint(ss.geom) AS source,
				ST_EndPoint(ss.geom) AS target,
				ss.geom
			FROM split_segment ss,
			LATERAL (SELECT count(*) FROM split_segment) sc
		) sub LOOP
			-- TODO Handle linear split surface for split segments
			-- TODO Handle linear split speed limits for split segments
			-- TODO Handle linear split flags for split segments
			
			output_segment.id = new_sub_segment.id || '_clip_' || split_geometry.row_index;
			output_segment.geom = split_geometry.geom;
			output_segment.impedance_surface = new_sub_segment.impedance_surface;
			output_segment.maxspeed_forward = new_sub_segment.maxspeed_forward;
			output_segment.tags = new_sub_segment.tags;

			IF split_geometry.row_index = 1 THEN
				output_segment.source = new_sub_segment.source;
				output_segment.target = 'connector.' || output_segment.id;

				-- If source connector doesn't already exist in output table, insert it
				INSERT INTO basic.connector (id, osm_id, geom, h3_3, h3_6)
				SELECT id, NULL, geometry, basic.to_short_h3_3(h3_lat_lng_to_cell(geometry::point, 3)::bigint),
					basic.to_short_h3_6(h3_lat_lng_to_cell(geometry::point, 6)::bigint)
				FROM temporal.connectors
				WHERE id = output_segment.source
				ON CONFLICT DO NOTHING;

				-- Get serial index of new or existing source connector
				SELECT index FROM basic.connector WHERE id = output_segment.source
				INTO output_segment.source_index;

				-- Create new target connector for split segment
				INSERT INTO basic.connector (id, osm_id, geom, h3_3, h3_6)
				VALUES (
					output_segment.target, NULL, split_geometry.target,
					basic.to_short_h3_3(h3_lat_lng_to_cell(split_geometry.target::point, 3)::bigint),
					basic.to_short_h3_6(h3_lat_lng_to_cell(split_geometry.target::point, 6)::bigint)
				);

				-- Get serial index of new target connector
				SELECT index FROM basic.connector WHERE id = output_segment.target
				INTO output_segment.target_index;
			ELSIF split_geometry.row_index > 1 AND split_geometry.row_index < split_geometry.row_count THEN
				output_segment.source = 'connector.' || new_sub_segment.id || '_clip_' || (split_geometry.row_index - 1);
				output_segment.target = 'connector.' || output_segment.id;

				-- Get serial index of source connector created by previous split segment
				SELECT index FROM basic.connector WHERE id = output_segment.source
				INTO output_segment.source_index;

				-- Create new target connector for split segment
				INSERT INTO basic.connector (id, osm_id, geom, h3_3, h3_6)
				VALUES (
					output_segment.target, NULL, split_geometry.target,
					basic.to_short_h3_3(h3_lat_lng_to_cell(split_geometry.target::point, 3)::bigint),
					basic.to_short_h3_6(h3_lat_lng_to_cell(split_geometry.target::point, 6)::bigint)
				);

				-- Get serial index of new target connector
				SELECT index FROM basic.connector WHERE id = output_segment.target
				INTO output_segment.target_index;
			ELSE
				output_segment.source = 'connector.' || new_sub_segment.id || '_clip_' || (split_geometry.row_index - 1);
				output_segment.target = new_sub_segment.target;

				-- Get serial index of source connector created by previous split segment
				SELECT index FROM basic.connector WHERE id = output_segment.source
				INTO output_segment.source_index;
				
				-- If target connector doesn't already exist in output table, insert it
				INSERT INTO basic.connector (id, osm_id, geom, h3_3, h3_6)
				SELECT id, NULL, geometry, basic.to_short_h3_3(h3_lat_lng_to_cell(geometry::point, 3)::bigint),
					basic.to_short_h3_6(h3_lat_lng_to_cell(geometry::point, 6)::bigint)
				FROM temporal.connectors
				WHERE id = output_segment.target
				ON CONFLICT DO NOTHING;

				-- Get serial index of new or existing target connector
				SELECT index FROM basic.connector WHERE id = output_segment.target
				INTO output_segment.target_index;
			END IF;

			output_segments = array_append(output_segments, output_segment);
		END LOOP;
	END IF;
END LOOP;

RETURN output_segments;

END
$$ LANGUAGE plpgsql;
