import time
from threading import Thread

import psycopg2

from src.db.db import Database
from src.utils.utils import print_error, print_info


class ProcessSegments(Thread):

    def __init__(
            self,
            thread_id: int,
            db_connection: Database,
            get_next_h3_index,
            cycling_surfaces: dict,
            default_speed_limits: dict,
        ):
        super().__init__(group=None, target=self)

        self.thread_id = thread_id
        self.db_connection = db_connection
        self.db_cursor = db_connection.cursor()
        self.get_next_h3_index = get_next_h3_index
        self.cycling_surfaces = cycling_surfaces
        self.default_speed_limits = default_speed_limits


    def run(self):
        """Process segment data for this H3 index region"""

        h3_short = self.get_next_h3_index()
        while h3_short is not None:
            # Get all segment IDs for this H3_3 index
            # Ensure segments are within valid H3_6 cells as well
            sql_get_segment_ids = f"""
                SELECT s.id
                FROM temporal.segments s, basic.h3_3_grid g1, basic.h3_6_grid g2
                WHERE ST_Intersects(ST_Centroid(s.geometry), g1.h3_geom)
                    AND ST_Intersects(ST_Centroid(s.geometry), g2.h3_geom)
                    AND g1.h3_short = '{h3_short}';
            """
            segment_ids = self.db_cursor.execute(sql_get_segment_ids)
            segment_ids = self.db_cursor.fetchall()

            # Process each segment
            for index in range(len(segment_ids)):
                id = segment_ids[index]
                sql_classify_segment = f"""
                    SELECT basic.classify_segment(
                        '{id[0]}',
                        '{self.cycling_surfaces}'::jsonb,
                        '{self.default_speed_limits}'::jsonb
                    );
                """
                try:
                    self.db_cursor.execute(sql_classify_segment)

                    # Commit changes to DB once every 1000 segments
                    # This significantly improves performance
                    if index % 1000 == 0:
                        self.db_connection.commit()
                # TODO In the event of a fatal exception, rollback all segments/connectors for this H3_3 region
                except psycopg2.Error as e:
                    if "deadlock detected" in str(e):
                        print_error(f"Thread {self.thread_id}: Deadlock detected, retrying.")
                        time.sleep(1)
                        try:
                            self.db_connection.commit()
                            continue
                        except Exception as e:
                            print_error(f"Thread {self.thread_id}: Failed to resolve deadlock, error: {e}.")
                    else:
                        print_error(f"Thread {self.thread_id} failed to process H3 index {h3_short}, error: {e}.")
                    break
                except Exception as e:
                    print_error(f"Thread {self.thread_id} failed to process H3 index {h3_short}, error: {e}.")
                    break

            h3_short = self.get_next_h3_index()


class ComputeImpedance(Thread):

    def __init__(
            self,
            thread_id: int,
            db_connection,
            get_next_h3_index,
        ):
        super().__init__(group=None, target=self)

        self.thread_id = thread_id
        self.db_connection = db_connection
        self.db_cursor = db_connection.cursor()
        self.get_next_h3_index = get_next_h3_index


    def run(self):
        """Update slope impedance data for this H3 index region"""

        h3_short = self.get_next_h3_index()
        while h3_short is not None:
            sql_update_impedance = f"""
                WITH segment AS (
                    SELECT id, length_m, geom, h3_6
                    FROM basic.segment
                    WHERE h3_6 = {h3_short}
                    AND impedance_slope IS NULL
                )
                UPDATE basic.segment AS sp
                SET impedance_slope = COALESCE(c.imp, 0), impedance_slope_reverse = COALESCE(c.rs_imp, 0)
                FROM segment,
                LATERAL basic.get_slope_profile(segment.geom, segment.length_m, ST_LENGTH(segment.geom)) s,
                LATERAL basic.compute_impedances(s.elevs, s.linklength, s.lengthinterval) c
                WHERE sp.h3_6 = segment.h3_6
                AND sp.id = segment.id;
            """
            try:
                self.db_cursor.execute(sql_update_impedance)
                self.db_connection.commit()
            except Exception as e:
                print_error(f"Thread {self.thread_id} failed to update impedances for H3 index {h3_short}, error: {e}.")
                break

            h3_short = self.get_next_h3_index()
