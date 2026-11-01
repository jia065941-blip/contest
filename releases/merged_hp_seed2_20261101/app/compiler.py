"""Map actor-legal public input into Astra plan; bind anonymous local refs to IDs."""
from collections import Counter
import math
from common import HERE,read,digest

ROLES={21000:'H',21001:'M',21002:'L'}


def public_case(path):
    profile=read(path)['imagineProfile'];deployment=read(path.with_name('deployment_area.json'))
    own=sorted([r['entity'] for r in profile['entityList'] if r['entity']['entityType'] in ROLES and r['entity']['sideId']==0],key=lambda e:e['id'])
    public=sorted([r['entity'] for r in profile['entityList'] if r['entity']['entityType'] in (9400,9600)],key=lambda e:(e['entityType'],e['lla']['x'],e['lla']['y']))
    counts=Counter(e['entityType'] for e in own)
    case=dict(map_bounds=profile['mapArea'],deployment=dict(coordinates=deployment['coordinates'],coordinatesHM=deployment['coordinatesHM'],
        altitude=deployment['launch_altitude_external_m'],mode=deployment['mode']),
        own_inventory=[dict(role=role,type=typ,count=counts[typ],initial_health_values=sorted({e['survivePoints'] for e in own if e['entityType']==typ})) for typ,role in ROLES.items()],
        public_targets=[dict(ref=f'T{i+1:03d}',type=e['entityType'],position=e['lla'],initial_health=e['survivePoints']) for i,e in enumerate(public)])
    units={f'{role}{i:03d}':e['id'] for typ,role in ROLES.items() for i,e in enumerate(r for r in own if r['entityType']==typ)}
    targets={f'T{i+1:03d}':e['id'] for i,e in enumerate(public)}
    return case,units,targets


def inside(point,poly):
    x,y=point;hit=False
    for a,b in zip(poly,poly[1:]+poly[:1]):
        ax,ay=a;bx,by=b
        cross=(x-ax)*(by-ay)-(y-ay)*(bx-ax)
        if abs(cross)<1e-10 and min(ax,bx)-1e-10<=x<=max(ax,bx)+1e-10 and min(ay,by)-1e-10<=y<=max(ay,by)+1e-10:return True
        if (ay>y)!=(by>y) and x<(bx-ax)*(y-ay)/(by-ay)+ax:hit=not hit
    return hit


def compile_case(path,arm):
    from design.planner_v2 import compile_plan
    case,units,targets=public_case(path)
    # No test score or private map index enters compile_plan.
    plan=compile_plan(case,arm)
    assert set(u['unit_key'] for u in plan['units'])==set(units)
    assert len(plan['units'])==len(units)
    rows=[]
    for u in plan['units']:
        assert u['role']==u['unit_key'][0]
        p=u['deployment'];assert len(p)==3 and all(math.isfinite(x) for x in p) and p[2]==case['deployment']['altitude']
        polygons=case['deployment']['coordinatesHM' if u['role'] in ('H','M') else 'coordinates']
        assert any(inside(p[:2],poly) for poly in polygons)
        task=u['task']
        if task['kind']=='attack':
            assert u['role'] in ('H','M') and task['target_ref'] in targets
        else:
            assert task['kind']=='search'
            for x,y in task['waypoints']:
                b=case['map_bounds'];assert b['lonMin']<=x<=b['lonMax'] and b['latMin']<=y<=b['latMax']
        rows.append([0,units[u['unit_key']],*p])
    rows.sort(key=lambda r:r[1]);rows.append([1,-1,0,0,0])
    compiled=dict(plan=plan,unit_ids=units,target_ids=targets,deployment_rows=rows,public_case=case,candidate_id=arm)
    proof=dict(public_input_sha256=digest(case),plan_sha256=digest(plan),deployment_sha256=digest(rows),
        units=len(units),public_targets=len(targets),all_units_once=True,all_deployments_legal=True,
        anonymous_input_only=True,no_hidden_fields=True)
    return compiled,proof
