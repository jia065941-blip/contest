"""Per-actor chronological adapter for the measured HP tree.

Keep one memory per fixed original assignment. H pool: original public sentries;
M pool: original public mothers. Unsupported/new bindings use baseline behavior.
This module proposes official action changes; the existing native receipt and
matched-branch validation wrappers remain required for experimental deployment.
"""
import copy
import importlib.util
import math
from pathlib import Path

HERE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('distilled_hp_tree',HERE/'decision_tree.py')
tree=importlib.util.module_from_spec(s);s.loader.exec_module(tree)


def step(observation,memory=None):
    """Input contains authorized own state/received track/public original target.

    Required fields: role, binding_matches, time_s, position, own_track (or None),
    public_track and map_bounds. Track requires position and timestamp_s only.
    No evaluator state or map/seed identifier is used.
    """
    memory=copy.deepcopy(memory or {})
    ctx=copy.deepcopy(observation);now=ctx['time_s'];height=ctx['position'][2]
    previous=memory.get('previous_own');vz=None
    if previous and 1<=now-previous['time_s']<=5:
        vz=(height-previous['height'])/(now-previous['time_s'])
    memory['previous_own']=dict(time_s=now,height=height)
    ctx['vertical_speed_mps']=vz
    # Recompute from the authorized chronological positions; ignore any
    # supplied native velocity or evaluator-derived estimate.
    track=ctx.get('own_track')
    if track:
        track=dict(position=list(track['position']),timestamp_s=track['timestamp_s'],velocity_en_mps=None)
        history=memory.setdefault('track_history',[]);stamp=track['timestamp_s']
        if not history or stamp>history[-1]['timestamp_s']:
            history.append(dict(position=list(track['position']),timestamp_s=stamp))
            history[:]=[h for h in history if stamp-h['timestamp_s']<=5]
        older=[h for h in history if 1<=stamp-h['timestamp_s']<=5]
        if older:
            prior=older[-1];dt=stamp-prior['timestamp_s'];a,b=prior['position'],track['position']
            track['velocity_en_mps']=[(b[0]-a[0])*111320*math.cos(math.radians((a[1]+b[1])/2))/dt,(b[1]-a[1])*110570/dt]
            track['estimator_dt_s']=dt
        ctx['own_track']=track
    ctx['clear_active']=tree.update_h_latch(ctx,memory.get('h_latched',False))
    memory['h_latched']=ctx['clear_active']
    return tree.choose(ctx),memory
