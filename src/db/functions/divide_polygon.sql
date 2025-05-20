-- DROP FUNCTION basic.divide_polygon(geometry, int4, int4);

CREATE OR REPLACE FUNCTION basic.divide_polygon(input_geom geometry, num_cells_x integer, num_cells_y integer)
 RETURNS TABLE(geom geometry)
 LANGUAGE plpgsql
AS $function$
BEGIN
    RETURN QUERY
    WITH 
        bounds AS (
            SELECT 
                ST_XMin(ST_Envelope(input_geom)) AS xmin,
                ST_XMax(ST_Envelope(input_geom)) AS xmax,
                ST_YMin(ST_Envelope(input_geom)) AS ymin,
                ST_YMax(ST_Envelope(input_geom)) AS ymax
        ),
        x_splits AS (
            SELECT 
                generate_series(1, num_cells_x) AS part,
                ST_XMin(input_geom) + (generate_series(1, num_cells_x) - 1) * ((ST_XMax(input_geom) - ST_XMin(input_geom)) / num_cells_x) AS x_start,
                ST_XMin(input_geom) + generate_series(1, num_cells_x) * ((ST_XMax(input_geom) - ST_XMin(input_geom)) / num_cells_x) AS x_end
            FROM bounds
        ),
        y_splits AS (
            SELECT 
                generate_series(1, num_cells_y) AS part,
                ST_YMin(input_geom) + (generate_series(1, num_cells_y) - 1) * ((ST_YMax(input_geom) - ST_YMin(input_geom)) / num_cells_y) AS y_start,
                ST_YMin(input_geom) + generate_series(1, num_cells_y) * ((ST_YMax(input_geom) - ST_YMin(input_geom)) / num_cells_y) AS y_end
            FROM bounds
        ),
        grid AS (
            SELECT 
                ST_MakePolygon(
                    ST_GeomFromText(
                        FORMAT(
                            'LINESTRING(%s %s, %s %s, %s %s, %s %s, %s %s)',
                            x_splits.x_start, y_splits.y_start,
                            x_splits.x_start, y_splits.y_end,
                            x_splits.x_end, y_splits.y_end,
                            x_splits.x_end, y_splits.y_start,
                            x_splits.x_start, y_splits.y_start
                        ),
						4326
                    )
                ) AS cell
            FROM x_splits, y_splits
        )
    SELECT 
        ST_Intersection(input_geom, cell) AS geom
    FROM 
        grid;
END;
$function$
;
