"""Frozen legal alternate binding enumeration; no learning or result input."""
import copy
from . import cooperation as compiler, feature_projector as project

def alternatives(ctx,original):
    """Same original eligibility pool, same A and trigger, alternate H binding B."""
    A,D=original[0]['unit'],original[0]['target']
    S=ctx['units'][A]['target']
    totals=compiler._budget(ctx,ctx['baseline_orders'])
    candidates=[]
    for uid,v in ctx['units'].items():
        Q=v.get('target')
        if uid in (A,original[1]['unit']) or v['role']!='H' or uid in ctx['baseline_orders'] or Q in (S,D):continue
        req=ctx['public_requirements'].get(Q)
        if not req or req['type'] not in (9400,9600):continue
        if totals.get(Q,0)-compiler.damage_budget('H',req['type'])+1e-8<req['initial_health']:continue
        eta=compiler._eta(v,ctx['targets'][S])
        if eta>=v['remaining_s']-60:continue
        roles=[copy.deepcopy(original[0]),{'unit':uid,'target':S,'duty':'接替A原目标；从独立盈余任务Q改派至S'}]
        compiler.validate_roles(roles,ctx)
        if compiler.coverage_audit({'orders':ctx['baseline_orders']},roles,3,ctx)['quality_guard_violated']:continue
        candidates.append({'roles':roles,'features':project.features(ctx,roles)})
    return candidates
