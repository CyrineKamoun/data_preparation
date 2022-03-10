sql_queries = {
    "accidents": '''
        DROP TABLE IF EXISTS temporal.accidents;
        CREATE TABLE temporal.accidents as 
        SELECT a.* 
        FROM public.germany_accidents a, temporal.study_area s
        WHERE ST_Intersects(a.geom,s.geom)
    '''
    ,
    "landuse": '''
        DROP TABLE IF EXISTS temporal.landuse; 
        DO $$                  
        BEGIN 
                IF  ( SELECT count(*) 
                FROM public.landuse_atkis l, (SELECT ST_UNION(geom) AS geom FROM temporal.study_area) s
                WHERE ST_Intersects(l.geom, s.geom)
                ) = 0    
                THEN
                        CREATE TABLE temporal.landuse AS 
                        SELECT l.objart_txt::text AS landuse, l.geom 
                        FROM public.dlm250_polygon l, (SELECT ST_UNION(geom) AS geom FROM temporal.study_area) s
                        WHERE ST_Intersects(l.geom, s.geom)
                        AND objart_txt IN ('AX_Siedlungsflaeche','AX_FlaecheBesondererFunktionalerPraegung','AX_Friedhof','AX_IndustrieUndGewerbeflaeche','AX_Landwirtschaft',
                        'AX_Siedlungsflaeche','AX_SportFreizeitUndErholungsflaeche');
                ELSE
                        CREATE TABLE temporal.landuse AS 
                        SELECT l.objart_txt::text AS landuse, l.geom 
                        FROM public.landuse_atkis l, (SELECT ST_UNION(geom) AS geom FROM temporal.study_area) s
                        WHERE ST_Intersects(l.geom, s.geom);
                END IF;
        END
        $$ ;
        ALTER TABLE temporal.landuse ADD COLUMN gid serial;
        CREATE INDEX ON temporal.landuse(gid);
        CREATE INDEX ON temporal.landuse USING GIST(geom);''',
    "landuse_additional": '''DROP TABLE IF EXISTS temporal.landuse_additional;
            CREATE TABLE temporal.landuse_additional AS 
            SELECT u.class_2018::text AS landuse, u.geom  
            FROM public.urban_atlas u, (SELECT ST_UNION(geom) AS geom FROM temporal.study_area) s
            WHERE ST_Intersects(u.geom, s.geom)
            AND u.class_2018 NOT IN ('Fast transit roads and associated land', 'Other roads and associated land');
            ALTER TABLE temporal.landuse_additional ADD COLUMN gid serial;
            CREATE INDEX ON temporal.landuse_additional(gid);
            CREATE INDEX ON temporal.landuse_additional USING GIST(geom);''',
    "pois": '''DROP TABLE IF EXISTS buffer_study_area;
        CREATE TEMP TABLE buffer_study_area AS 
        SELECT ST_BUFFER(ST_UNION(geom), 0.027) AS geom 
        FROM temporal.study_area;

        DROP TABLE IF EXISTS temporal.pois;
        CREATE TABLE temporal.pois as 
        SELECT p.* 
        FROM public.pois_fused p, buffer_study_area s
        WHERE ST_Intersects(p.geom,s.geom);''',
    "buildings_custom": '''DROP TABLE IF EXISTS temporal.buildings_custom;
            CREATE TABLE temporal.buildings_custom AS 
            SELECT b.ags, (ST_DUMP(b.geom)).geom  
            FROM public.germany_buildings b, (SELECT ST_UNION(geom) AS geom FROM temporal.study_area) s
            WHERE ST_Intersects(b.geom, s.geom);
            ALTER TABLE temporal.buildings_custom ADD COLUMN gid serial;
            CREATE INDEX ON temporal.buildings_custom(gid);
            CREATE INDEX ON temporal.buildings_custom USING GIST(geom);''',
    "geographical_names": '''DROP TABLE IF EXISTS temporal.geographical_names;
            CREATE TABLE temporal.geographical_names AS 
            SELECT g.* 
            FROM public.germany_geographical_names_points g, temporal.study_area s 
            WHERE ST_Intersects(g.geom,s.geom);
            CREATE INDEX ON temporal.geographical_names(id);
            CREATE INDEX ON temporal.geographical_names USING GIST(geom);''',
    "census": '''DROP TABLE IF EXISTS grid;
            DROP TABLE IF EXISTS temporal.census;
            CREATE TEMP TABLE grid AS 
            SELECT DISTINCT g.id, g.geom
            FROM public.germany_grid_100_100 g, temporal.study_area s
            WHERE ST_Intersects(s.geom,g.geom);

            ALTER TABLE grid ADD PRIMARY KEY(id);

            CREATE TABLE temporal.census AS 
            WITH first_group AS 
            (
                SELECT g.id, REPLACE(merkmal,'"','') AS merkmal, jsonb_object(array_agg(c.auspraegung_text), array_agg(c.anzahl)::TEXT[]) AS demography
                FROM grid g, public.germany_census_demography_2011 c
                WHERE g.id = c.gitter_id_100m
                GROUP BY g.id, merkmal
            ),
            second_group AS 
            (
                SELECT id, jsonb_object_agg(merkmal, demography)::text AS demography 
                FROM first_group
                GROUP BY id
            )
            SELECT g.id, CASE WHEN f.id IS NULL THEN NULL ELSE demography::text END AS demography , g.geom
            FROM grid g 
            LEFT JOIN second_group f 
            ON g.id = f.id;

            ALTER TABLE temporal.census ADD COLUMN pop integer; 
            UPDATE temporal.census  
            SET pop = (demography::jsonb -> ' INSGESAMT' ->> 'Einheiten insgesamt')::integer;
            CREATE INDEX ON temporal.census(id);
            CREATE INDEX ON temporal.census USING GIST(geom);
            ''',
    "study_area": None

}


