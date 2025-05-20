BEGIN;
SET LOCAL check_function_bodies TO FALSE;
DROP FUNCTION IF EXISTS basic.get_idw_values;
CREATE OR REPLACE FUNCTION basic.get_idw_values(geom geometry, buffer_distance float)
RETURNS TABLE (dp_geom geometry, distance float, val float)
 LANGUAGE sql
AS $function$
	SELECT dp.geom, ST_DISTANCE(r.geom,dp.geom) distance, val
	FROM  
	(
		SELECT geom, st_clip(d.rast, st_buffer(geom, buffer_distance), 0.0) AS rast 	
		FROM dem d
		WHERE d.rast && st_buffer(geom, buffer_distance)
	) r
	, LATERAL ST_PixelAsCentroids(rast, 1) AS dp
	WHERE ST_DISTANCE(r.geom, dp.geom) != 0
	ORDER BY r.geom <-> dp.geom 
	LIMIT 3

$function$;
COMMIT;
