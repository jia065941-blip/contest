"""Legal feature calculations for the measured HP behavior tree.

No map identity, outcome, true target position or native target velocity is read.
Coordinates are authorized received/public coordinates, not identity gates.
"""
import math


def distance(a,b):
    return math.hypot((a[0]-b[0])*111320*math.cos(math.radians((a[1]+b[1])/2)),(a[1]-b[1])*110570)


def project(ctx,prior_latched=False):
    track=ctx.get('own_track');v=track.get('velocity_en_mps') if track else None
    dt=track.get('estimator_dt_s') if track else None
    valid=v is not None and dt is not None and 1<=dt<=5 and len(v)==2 and all(math.isfinite(x) for x in v)
    age=ctx['time_s']-track['timestamp_s'] if track else 0.
    speed=math.hypot(*v) if valid else 0.
    position=track['position'] if track else ctx['position'];home=ctx['public_track']['position']
    dx=(home[0]-position[0])*111320*math.cos(math.radians((home[1]+position[1])/2));dy=(home[1]-position[1])*110570
    dot=dx*v[0]+dy*v[1] if valid else 0.
    cross=abs(dx*v[1]-dy*v[0])/speed if speed>0 else 0.
    home_time=dot/(speed*speed) if speed>0 else 0.
    home_time_valid=math.isfinite(home_time) and speed>0
    fraction=min(age/home_time,1.) if home_time_valid and home_time>0 else 0.
    point=[position[0]+fraction*(home[0]-position[0]),position[1]+fraction*(home[1]-position[1])]
    bounds=ctx['map_bounds'];vz=ctx.get('vertical_speed_mps')
    features=dict(role=ctx['role'],prior_latched=bool(prior_latched),binding_matches=bool(ctx['binding_matches']),
        own_track_present=track is not None,age_s=age,vertical_speed_valid=vz is not None and math.isfinite(vz),
        vertical_speed_mps=vz if vz is not None and math.isfinite(vz) else 0.,range_m=distance(ctx['position'],position),
        clear_active=bool(ctx.get('clear_active',False)),velocity_history_valid=valid,estimator_dt_s=dt if dt is not None else 0.,
        speed_mps=speed,stale_displacement_m=speed*age,home_distance_m=math.hypot(dx,dy),homeward_dot=dot,
        cross_track_m=cross,home_time_valid=home_time_valid,home_time_s=home_time,
        point_in_map=all(math.isfinite(x) for x in point) and bounds[0]<=point[0]<=bounds[1] and bounds[2]<=point[1]<=bounds[3])
    return features,point
