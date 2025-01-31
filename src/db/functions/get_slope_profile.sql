DROP FUNCTION IF EXISTS public.get_slope_profile;
CREATE OR REPLACE FUNCTION public.get_slope_profile(way_geom geometry, length_meters float, length_degree float, interval_ float default 10, dem_resolution float default 30)
RETURNS TABLE(elevs float[], linkLength float, lengthInterval float)
LANGUAGE plpgsql
AS $function$
DECLARE 
	translation_m_degree NUMERIC;
	dump_points geometry[];
BEGIN
	IF length_meters = 0 THEN
		length_meters = 1;
	END IF;

	translation_m_degree = length_degree/length_meters;
	dem_resolution = dem_resolution * translation_m_degree;

	IF length_meters > (2*interval_) THEN
		SELECT ARRAY(SELECT (ST_DUMP(ST_Lineinterpolatepoints(way_geom, interval_/length_meters))).geom)
		INTO dump_points;
	ELSEIF length_meters > interval_ AND length_meters < (2*interval_) THEN
		SELECT ARRAY(SELECT ST_LineInterpolatePoint(way_geom,0.5))
		INTO dump_points;
		interval_ = length_meters/2;
	ELSE
		SELECT ARRAY(SELECT NULL::geometry)
		INTO dump_points;
	END IF;

	RETURN query
	WITH points AS 
	(
		SELECT ROW_NUMBER() OVER() cnt, geom, length_meters 
		FROM (
			SELECT st_startpoint(way_geom) AS geom
			UNION ALL 
			SELECT unnest(dump_points)
			UNION ALL 
			SELECT st_endpoint(way_geom) 
		) x
	)
	SELECT array_agg(elev) elevs, length_meters, interval_ 
	FROM 
	(
		SELECT SUM(idw.val/(idw.distance/translation_m_degree))/SUM(1/(idw.distance/translation_m_degree))::real AS elev
		FROM points p, get_idw_values(geom, dem_resolution) idw
		WHERE p.geom IS NOT NULL 
		GROUP BY cnt 
		ORDER BY cnt 
	) x;

END;
$function$;

/*
EXPLAIN ANALYZE 
SELECT s.*
FROM ways, LATERAL get_slope_profile(the_geom, length_m, length) s
LIMIT 1
*/