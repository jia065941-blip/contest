"""Generated measured HP tree. See tree_spec.json for provenance and limits."""
import importlib.util
import math
from pathlib import Path

HERE=Path(__file__).resolve().parent
_s=importlib.util.spec_from_file_location('hp_legal_features',HERE/'features.py')
features=importlib.util.module_from_spec(_s);_s.loader.exec_module(features)

def _numeric(value):
    return not isinstance(value,bool) and isinstance(value,(int,float)) and math.isfinite(value)

def latch_H(f):
    if f.get('prior_latched') is True:
        return True
    else:
        if f.get('role') == 'H':
            if f.get('binding_matches') is True:
                if f.get('own_track_present') is True:
                    if _numeric(f.get('age_s')) and f.get('age_s') >= 0:
                        if _numeric(f.get('age_s')) and f.get('age_s') <= 2:
                            if f.get('vertical_speed_valid') is True:
                                if _numeric(f.get('vertical_speed_mps')) and f.get('vertical_speed_mps') < 0:
                                    if _numeric(f.get('range_m')) and f.get('range_m') <= 2000:
                                        return True
                                    else:
                                        return False
                                else:
                                    return False
                            else:
                                return False
                        else:
                            return False
                    else:
                        return False
                else:
                    return False
            else:
                return False
        else:
            return False

def clear_H(f):
    if f.get('role') == 'H':
        if f.get('binding_matches') is True:
            if f.get('own_track_present') is True:
                if _numeric(f.get('age_s')) and f.get('age_s') >= 0:
                    if _numeric(f.get('age_s')) and f.get('age_s') <= 2:
                        if f.get('clear_active') is True:
                            return True
                        else:
                            return False
                    else:
                        return False
                else:
                    return False
            else:
                return False
        else:
            return False
    else:
        return False

def predict_M(f):
    if f.get('role') == 'M':
        if f.get('binding_matches') is True:
            if f.get('own_track_present') is True:
                if _numeric(f.get('age_s')) and f.get('age_s') > 2:
                    if _numeric(f.get('age_s')) and f.get('age_s') <= 180:
                        if _numeric(f.get('range_m')) and f.get('range_m') <= 5000:
                            if f.get('velocity_history_valid') is True:
                                if _numeric(f.get('estimator_dt_s')) and f.get('estimator_dt_s') >= 1:
                                    if _numeric(f.get('estimator_dt_s')) and f.get('estimator_dt_s') <= 5:
                                        if _numeric(f.get('speed_mps')) and f.get('speed_mps') > 0:
                                            if _numeric(f.get('speed_mps')) and f.get('speed_mps') <= 30:
                                                if _numeric(f.get('stale_displacement_m')) and f.get('stale_displacement_m') > 100:
                                                    if _numeric(f.get('home_distance_m')) and f.get('home_distance_m') > 100:
                                                        if _numeric(f.get('homeward_dot')) and f.get('homeward_dot') > 0:
                                                            if _numeric(f.get('cross_track_m')) and f.get('cross_track_m') <= 100:
                                                                if f.get('home_time_valid') is True:
                                                                    if _numeric(f.get('home_time_s')) and f.get('home_time_s') > 0:
                                                                        if f.get('point_in_map') is True:
                                                                            return 'WAYPOINT'
                                                                        else:
                                                                            return 'KEEP'
                                                                    else:
                                                                        return 'KEEP'
                                                                else:
                                                                    return 'KEEP'
                                                            else:
                                                                return 'KEEP'
                                                        else:
                                                            return 'KEEP'
                                                    else:
                                                        return 'KEEP'
                                                else:
                                                    return 'KEEP'
                                            else:
                                                return 'KEEP'
                                        else:
                                            return 'KEEP'
                                    else:
                                        return 'KEEP'
                                else:
                                    return 'KEEP'
                            else:
                                return 'KEEP'
                        else:
                            return 'KEEP'
                    else:
                        return 'KEEP'
                else:
                    return 'KEEP'
            else:
                return 'KEEP'
        else:
            return 'KEEP'
    else:
        return 'KEEP'

def update_h_latch(ctx,prior_latched=False):
    f,_=features.project(ctx,prior_latched)
    return latch_H(f)


def choose(ctx):
    f,point=features.project(ctx)
    action=predict_M(f)
    waypoint={'action':action}
    if action=='WAYPOINT':waypoint['point']=point
    return {'clear_maneuver':clear_H(f),'waypoint':waypoint}
