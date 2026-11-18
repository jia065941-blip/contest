"""Public-information-only Astra v2 planner. All lengths metres; time seconds.
No simulator, private scenario, historical evaluation, or Q-label policy access.
compile_plan(case, candidate_id); step_group(plan,current,targets,step,memory).
"""
from __future__ import annotations
import math
import copy
from functools import lru_cache

VERSION = 'pku_initial_astra_v2'
# These are design hypotheses, not measured survival rates.
PROFILES = [
 dict(id='A00',name='完整摧毁边际预算',mechanism='以加一H或成组M的完整摧毁概率增量配置余量，统一H天眼窗，L三机首批',mother_bias=1.,sentry_bias=1.,h_survival=.84,m_survival=.88,window_mode='balanced',arrival_fraction=.50,m_lag=18,hidden_batch=3,hidden_topup=2,hidden_spread=24,search_spacing=3600,search_rear_fraction=.22,loss_mode='expected',refresh=2),
 dict(id='A01',name='母巢五权重保险',mechanism='提高母巢失败代价，余H优先补母巢一员损失余量，L四机首批减少未知HP下打残',mother_bias=1.65,sentry_bias=1.,h_survival=.80,m_survival=.88,window_mode='mother',arrival_fraction=.55,m_lag=12,hidden_batch=4,hidden_topup=2,hidden_spread=28,search_spacing=3600,search_rear_fraction=.20,loss_mode='one_loss',refresh=1),
 dict(id='A02',name='哨卫通道先压制',mechanism='按公开进攻线附近哨卫密度加权摧毁预算与到达提前量，削弱潜在附属防御但不宣称已压制',mother_bias=1.,sentry_bias=1.60,h_survival=.82,m_survival=.87,window_mode='sentry',arrival_fraction=.45,m_lag=8,hidden_batch=3,hidden_topup=3,hidden_spread=20,search_spacing=3400,search_rear_fraction=.24,loss_mode='expected',refresh=1),
 dict(id='A03',name='窗口偏差容忍',mechanism='显式提高普通伤害而非窗口伤害的组合价值，攻击窗边缘目标用M补足普通伤害',mother_bias=1.20,sentry_bias=1.,h_survival=.85,m_survival=.90,window_mode='balanced',arrival_fraction=.50,m_lag=10,hidden_batch=4,hidden_topup=2,hidden_spread=30,search_spacing=3600,search_rear_fraction=.25,loss_mode='fallback',refresh=1),
 dict(id='A04',name='暗巢早发现远端覆盖',mechanism='选择仍能覆盖H主攻击的最早窗口，L更前置且首批三机，优先无法晚到的远端暗巢',mother_bias=1.,sentry_bias=1.,h_survival=.85,m_survival=.88,window_mode='early',arrival_fraction=.32,m_lag=14,hidden_batch=3,hidden_topup=2,hidden_spread=18,search_spacing=3200,search_rear_fraction=.14,loss_mode='expected',refresh=2),
 dict(id='A05',name='晚窗高速重击',mechanism='让最远母巢进入窗口中心，H重击末端与L前出发现共享天眼，L四机先打并保留后续搜索',mother_bias=1.25,sentry_bias=1.,h_survival=.83,m_survival=.88,window_mode='late',arrival_fraction=.64,m_lag=0,hidden_batch=4,hidden_topup=3,hidden_spread=24,search_spacing=3600,search_rear_fraction=.28,loss_mode='expected',refresh=1),
 dict(id='A06',name='M后续到达与H传感伴随',mechanism='对M晚到超过窗口的目标保留最后一H作为传感伴随，先行H争取重击，预算按混合普通/天眼重新核算',mother_bias=1.15,sentry_bias=1.,h_survival=.86,m_survival=.90,window_mode='balanced',arrival_fraction=.52,m_lag=6,hidden_batch=3,hidden_topup=2,hidden_spread=32,search_spacing=3500,search_rear_fraction=.24,loss_mode='escort',refresh=1),
 dict(id='A07',name='长航程目标独立闭环',mechanism='远目标增加H保险，减少依赖过期坐标的M；暗巢按剩余寿命最紧先派单',mother_bias=1.1,sentry_bias=1.,h_survival=.80,m_survival=.86,window_mode='deep',arrival_fraction=.56,m_lag=20,hidden_batch=4,hidden_topup=2,hidden_spread=26,search_spacing=3600,search_rear_fraction=.18,loss_mode='deep',refresh=1),
 dict(id='A08',name='本群损失重建与留一搜索',mechanism='提高成员损失下完整预算，群分裂只用当前成员重建覆盖，L每活跃搜索带保留一机避免全员转攻',mother_bias=1.1,sentry_bias=1.15,h_survival=.77,m_survival=.82,window_mode='balanced',arrival_fraction=.50,m_lag=22,hidden_batch=3,hidden_topup=3,hidden_spread=34,search_spacing=3300,search_rear_fraction=.30,loss_mode='one_loss',refresh=1),
 dict(id='A09',name='M成组收尾与母巢角度退化预算',mechanism='以母巢M单发2.4和H普通9.6作退化预算，M成组而非零碎分散，减少普通窗外打残',mother_bias=1.3,sentry_bias=1.,h_survival=.84,m_survival=.91,window_mode='balanced',arrival_fraction=.48,m_lag=0,hidden_batch=4,hidden_topup=2,hidden_spread=22,search_spacing=3600,search_rear_fraction=.26,loss_mode='fallback',refresh=1),
 dict(id='A10',name='暗巢两段补充证据门控',mechanism='暗巢首批五机提高未知HP容忍度，仅新鲜存活记录触发两机补充，避免按命令回执重复灌入',mother_bias=1.,sentry_bias=1.15,h_survival=.84,m_survival=.88,window_mode='early',arrival_fraction=.42,m_lag=15,hidden_batch=5,hidden_topup=2,hidden_spread=38,search_spacing=3800,search_rear_fraction=.26,loss_mode='expected',refresh=2),
 dict(id='A11',name='全目标最差余量补强',mechanism='在边际收益相近时先补最薄弱完整摧毁预算，H与M分散到达避免单枚爆炸同时带走整组，L宽覆盖后集中',mother_bias=1.15,sentry_bias=1.1,h_survival=.81,m_survival=.85,window_mode='balanced',arrival_fraction=.50,m_lag=30,hidden_batch=4,hidden_topup=3,hidden_spread=40,search_spacing=3400,search_rear_fraction=.32,loss_mode='maxmin',refresh=1),
]
SPEED={'H':200.,'M':120.,'L':100.}

def profile(cid):
    return copy.deepcopy(next(p for p in PROFILES if p['id']==cid))

def ecef(p):
    lon,lat=map(math.radians,p[:2]); z=p[2] if len(p)>2 else 0.
    a=6378137.; e2=6.6943799901413165e-3
    n=a/math.sqrt(1-e2*math.sin(lat)**2)
    return ((n+z)*math.cos(lat)*math.cos(lon),(n+z)*math.cos(lat)*math.sin(lon),(n*(1-e2)+z)*math.sin(lat))

def distance(a,b):
    return math.dist(ecef(a),ecef(b))

def _inside(p,poly):
    x,y=p[:2]; yes=False
    for a,b in zip(poly,poly[1:]+poly[:1]):
        cross=(x-a[0])*(b[1]-a[1])-(y-a[1])*(b[0]-a[0])
        if abs(cross)<1e-10 and min(a[0],b[0])-1e-10<=x<=max(a[0],b[0])+1e-10 and min(a[1],b[1])-1e-10<=y<=max(a[1],b[1])+1e-10:return True
        if (a[1]>y)!=(b[1]>y) and x<(b[0]-a[0])*(y-a[1])/(b[1]-a[1])+a[0]: yes=not yes
    return yes

def legal(p,polys):return any(_inside(p,v) for v in polys)

def _samples(polys):
    out=[]
    for poly in polys:
        for a,b in zip(poly,poly[1:]+poly[:1]):
            for j in range(41):out.append((a[0]+(b[0]-a[0])*j/40,a[1]+(b[1]-a[1])*j/40,3000.))
        xmin=min(v[0] for v in poly);xmax=max(v[0] for v in poly);ymin=min(v[1] for v in poly);ymax=max(v[1] for v in poly)
        for j in range(1,12):
            for k in range(1,12):
                p=(xmin+(xmax-xmin)*j/12,ymin+(ymax-ymin)*k/12,3000.)
                if _inside(p,poly):out.append(p)
    return sorted(set(out))

def _nearest(p,polys):
    if legal(p,polys):return [p[0],p[1],3000.]
    # Closest segment projection in local lon/lat metric; legality rechecked.
    c=math.cos(math.radians(p[1])); cand=[]
    for poly in polys:
        for a,b in zip(poly,poly[1:]+poly[:1]):
            dx=(b[0]-a[0])*c;dy=b[1]-a[1]
            t=max(0.,min(1.,(((p[0]-a[0])*c)*dx+(p[1]-a[1])*dy)/(dx*dx+dy*dy or 1)))
            q=[a[0]+t*(b[0]-a[0]),a[1]+t*(b[1]-a[1]),3000.]
            cand.append(q)
    return min(cand,key=lambda q:distance(p,q))

def _eta(d,role):return d/SPEED[role]*1.06+12.

def _point_for_eta(tgt,role,wanted,polys,samples,used,rank):
    # Continuous edge roots + all polygon grids, then collision-aware closest ETA.
    desired=max(0.,(wanted-12.)/1.06)*SPEED[role]
    cand=list(samples)
    for poly in polys:
        for a,b in zip(poly,poly[1:]+poly[:1]):
            # Convex distance along a segment may cross radius twice.
            prev=[*a,3000.]; prevf=distance(prev,tgt)-desired
            for j in range(1,9):
                q=[a[0]+(b[0]-a[0])*j/8,a[1]+(b[1]-a[1])*j/8,3000.]
                fq=distance(q,tgt)-desired
                if prevf*fq<=0:
                    l=list(prev);r=list(q);fl=prevf
                    for _ in range(24):
                        mid=[(l[0]+r[0])/2,(l[1]+r[1])/2,3000.];fm=distance(mid,tgt)-desired
                        if fl*fm<=0:r=mid
                        else:l=mid;fl=fm
                    cand.append(mid)
                prev=q;prevf=fq
    # 350 m preferred deployment separation avoids <=100m common blast footprint.
    # It is a soft tactical constraint, never an invented validity condition.
    def key(q):
        sep=min((distance(q,u) for u in used[-100:]),default=10000)
        angle=math.atan2(q[1]-tgt[1],(q[0]-tgt[0])*math.cos(math.radians(tgt[1])))
        desired_side=1 if rank%2 else -1
        return (abs(_eta(distance(q,tgt),role)-wanted)+max(0,350-sep)/25, max(0,-desired_side*angle)*.01,q[0],q[1])
    return list(min(cand,key=key))

@lru_cache(maxsize=20000)
def _p_kill(h,m,hp,dh,dm,ph,pm):
    return sum(math.comb(h,a)*ph**a*(1-ph)**(h-a)*math.comb(m,b)*pm**b*(1-pm)**(m-b)
               for a in range(h+1) for b in range(m+1) if a*dh+b*dm+1e-8>=hp)

def _value(t,h,m,p):
    sat=t['window_feasible']; hp=t['initial_health']; mother=t['type']==9400
    dh=20. if sat else (9.6 if mother else 12.);dm=2.4 if mother else 3.
    ph=p['h_survival'];pm=p['m_survival']
    if p['loss_mode']=='deep':pm=max(.60,pm-.12*t['nearest_h_eta_s']/700);ph=max(.65,ph-.04*t['nearest_h_eta_s']/700)
    probability=_p_kill(h,m,hp,dh,dm,ph,pm)
    ordinary=_p_kill(h,m,hp,9.6 if mother else 12.,dm,ph,pm)
    weight=(5*p['mother_bias'] if mother else 2*p['sentry_bias'])
    if not mother and p['window_mode']=='sentry':weight*=1+min(.30,t['corridor_density']*.03)
    if p['loss_mode']=='deep':weight*=1+.25*t['nearest_h_eta_s']/700
    fallback_weight=.28 if p['loss_mode']=='fallback' else .07
    one_loss=(max(0,h-1)*dh+m*dm>=hp)
    robustness=.10*one_loss if p['loss_mode']=='one_loss' else 0.
    if p['loss_mode']=='maxmin': probability=math.sqrt(max(0,probability))
    return weight*((1-fallback_weight)*probability+fallback_weight*ordinary+robustness+.001*min(1,(h*dh+m*dm)/hp))

def _allocate(targets,inventory,p):
    # First protect every target with a complete intact-window package.
    alloc={t['ref']:{'H':2 if t['type']==9400 else 1,'M':0} for t in targets}
    used=sum(v['H'] for v in alloc.values())
    # Generic scarcity fallback: don't scatter incomplete H fragments.
    if used>inventory['H']:
        alloc={t['ref']:{'H':0,'M':0} for t in targets}
        for t in sorted(targets,key=lambda t:-(5 if t['type']==9400 else 2)/(1+t['nearest_h_eta_s']/1000)):
            need=2 if t['type']==9400 else 1
            if need<=inventory['H']-sum(v['H'] for v in alloc.values()):alloc[t['ref']]['H']=need
    remh=inventory['H']-sum(v['H'] for v in alloc.values());remm=inventory['M']
    while remh or remm:
        best=None
        for t in targets:
            a=alloc[t['ref']];base=_value(t,a['H'],a['M'],p)
            for dh in range(min(2,remh)+1):
                for dm in range(min(8,remm)+1):
                    if not dh+dm:continue
                    gain=_value(t,a['H']+dh,a['M']+dm,p)-base
                    # Bundle upgrades prevent zero marginal one-M deadlocks.
                    cost=dh*(inventory['M']/max(1,inventory['H']))+dm
                    key=(gain/cost,gain,-cost,-t['nearest_h_eta_s'],t['ref'])
                    if best is None or key>best[0]:best=(key,t['ref'],dh,dm)
        _,ref,dh,dm=best;alloc[ref]['H']+=dh;alloc[ref]['M']+=dm;remh-=dh;remm-=dm
    return alloc

def compile_plan(case,candidate_id):
    p=profile(candidate_id);inv={r['role']:int(r['count']) for r in case['own_inventory']}
    hm=case['deployment']['coordinatesHM'];lp=case['deployment']['coordinates'];samples=_samples(hm);bounds=case['map_bounds']
    targets=[]
    for old in case['public_targets']:
        t=copy.deepcopy(old);pos=old['position'];t['position']=[pos['x'],pos['y'],pos.get('z',0.)]
        ds=[distance(q,t['position']) for q in samples]
        nearest=distance(_nearest(t['position'],hm),t['position'])
        t['nearest_h_eta_s']=_eta(nearest,'H');t['farthest_h_eta_s']=_eta(max(ds),'H');t['nearest_m_eta_s']=_eta(nearest,'M')
        t['corridor_density']=sum(1 for q in case['public_targets'] if q['type']==9600 and distance(t['position'],[q['position']['x'],q['position']['y'],0])<25000)
        targets.append(t)
    # Global schedule uses only public inputs. Candidate windows every 5 seconds.
    def window_value(start):
        score=0.
        for t in targets:
            lo=t['nearest_h_eta_s'];hi=t['farthest_h_eta_s'];arr=max(lo,min(hi,start+180*p['arrival_fraction']))
            inside=start+12<=arr<start+168
            w=5*p['mother_bias'] if t['type']==9400 else 2*p['sentry_bias']
            if p['window_mode']=='deep':w*=1+.4*lo/700
            score+=w*(1 if inside else .1)-.00001*w*arr
        # Earlier discovery and more L lifetime are real opportunity gains.
        late_penalty=.0008 if p['window_mode']=='early' else .0002
        if p['window_mode']=='late':late_penalty=-.00003
        return score-late_penalty*start
    latest=min(1000,max(t['nearest_h_eta_s'] for t in targets)+180)
    trigger=max(range(0,int(latest)+1,5),key=lambda s:(window_value(s),-s))
    for t in targets:
        aim=trigger+180*p['arrival_fraction']
        if p['window_mode']=='sentry' and t['type']==9600:aim-=18
        t['scheduled_h_eta_s']=max(t['nearest_h_eta_s'],min(t['farthest_h_eta_s'],aim))
        t['window_feasible']=trigger+8<=t['scheduled_h_eta_s']<trigger+172
    alloc=_allocate(targets,inv,p);units=[];used=[];counter={'H':0,'M':0,'L':0}
    for t in targets:
        a=alloc[t['ref']]
        for role in ('H','M'):
            for j in range(a[role]):
                key=f'{role}{counter[role]:03d}';counter[role]+=1
                jitter=(j-(a[role]-1)/2)*(4 if role=='H' else 5)
                aim=t['scheduled_h_eta_s']+jitter+(p['m_lag'] if role=='M' else 0)
                escort=p['loss_mode']=='escort' and role=='H' and j==a['H']-1 and a['H']>=2 and a['M']>0 and t['nearest_m_eta_s']>t['scheduled_h_eta_s']+70
                if escort:aim=max(aim,t['nearest_m_eta_s']+8)
                dep=_point_for_eta(t['position'],role,aim,hm,samples,used,j);used.append(dep);eta=_eta(distance(dep,t['position']),role)
                task={'kind':'attack','target_ref':t['ref'],'binding':'primary' if j==0 else 'reinforcement','role':role,'planned_eta_s':eta,'sensor_escort':escort}
                units.append({'unit_key':key,'role':role,'deployment':dep,'task':task,'target_eta_s':eta,'binding':task['binding']})
        assigned=[u for u in units if u['task'].get('target_ref')==t['ref']]
        hs=[u for u in assigned if u['role']=='H'];ms=[u for u in assigned if u['role']=='M']
        inwin=sum(trigger<=u['target_eta_s']<trigger+180 for u in hs)
        mother=t['type']==9400;dh=16 if mother else 12;dm=4 if mother else 3
        ordinary=dh*a['H']+dm*a['M'];floor=(.6 if mother else 1)*ordinary
        window=20*inwin+(a['H']-inwin)*dh*(.6 if mother else 1)+a['M']*dm*(.6 if mother else 1)
        t.update(allocation=a,estimated={'ordinary_nominal_damage':ordinary,'ordinary_angle_floor_damage':floor,'window_angle_floor_damage':window,'one_H_lost_window_floor_damage':window-(20 if inwin else dh*(.6 if mother else 1)),'one_M_lost_window_floor_damage':window-dm*(.6 if mother else 1),'h_in_window':inwin,'complete_intact_window_budget':window+1e-8>=t['initial_health'],'eta_range_s':[min(u['target_eta_s'] for u in assigned),max(u['target_eta_s'] for u in assigned)],'m_last_h_gap_s':max([u['target_eta_s'] for u in ms],default=0)-max([u['target_eta_s'] for u in hs],default=0),'eta_uncertainty_s':45,'survival_probability_is_design_hypothesis':True})
    # Search lanes cover full north/south span; arbitrary polygons handled by projection.
    midlat=(bounds['latMin']+bounds['latMax'])/2;sy=111132.;sx=111320.*math.cos(math.radians(midlat))
    height=(bounds['latMax']-bounds['latMin'])*sy
    nlanes=max(1,min(inv['L'],math.ceil(height/p['search_spacing'])))
    lon_front=max(v[0] for poly in lp for v in poly);lon_back=min(v[0] for poly in lp for v in poly)
    for i in range(inv['L']):
        lane=i%nlanes;wave=i//nlanes;lat=bounds['latMin']+(lane+.5)*(bounds['latMax']-bounds['latMin'])/nlanes
        # At least one rear-origin scout per lane; other members form forward search/attack reserve.
        rear=(wave==0 and i<math.ceil(inv['L']*p['search_rear_fraction']))
        desired_lon=lon_back+350/sx if rear else lon_front-(600+wave*1800)/sx
        dep=_nearest([desired_lon,lat,3000.],lp)
        east=bounds['lonMax']-400/sx;west=max(bounds['lonMin']+400/sx,dep[0])
        adjacent=(lane+2)%nlanes;lat2=bounds['latMin']+(adjacent+.5)*(bounds['latMax']-bounds['latMin'])/nlanes
        # Second leg is optional broad return, not credited as guaranteed coverage.
        wps=[[east,lat],[west,lat2]]
        key=f'L{i:03d}'
        task={'kind':'search','waypoints':wps,'waypoint_index':0,'binding':'search','lane':lane,'search_keeper':wave==0,'role':'L'}
        units.append({'unit_key':key,'role':'L','deployment':dep,'task':task,'binding':'search','target_eta_s':_eta(distance(dep,[east,lat,0]),'L')})
    lus=[u for u in units if u['role']=='L'];requesters=sorted(lus,key=lambda u:(not u['task']['search_keeper'],abs(u['deployment'][1]-midlat),u['unit_key']))
    sat={'planned_trigger_s':trigger,'requester_unit_key':requesters[0]['unit_key'],'backup_unit_keys':[u['unit_key'] for u in requesters[1:]],'backup_slot_seconds':3,'latest_trigger_s':min(1150,trigger+45),'duration_s':180,'window_guard_s':12,'adaptive_advance_seconds':18,'adaptive_h_arrival_threshold':.6,'quota':1}
    runtime=dict(p,public_replan_interval_s=8,public_fresh_s=5,hidden_fresh_s=6,hidden_confirm_grace_s=18,hidden_lease_s=40,proposal_retry_s=6,search_switch_m=8000,search_turn_loss_s=45,l_remaining_margin_s=90,search_keep_fraction=.10,public_binding_lock_distance_m=18000,public_reassign_hysteresis=1.25,minimum_searchers_per_group=1,damage_credit_requires_observed_self_terminal=False)
    return {'version':VERSION,'id':candidate_id,'name':p['name'],'units':units,'public_targets':targets,'satellite':sat,'runtime':runtime,'search':{'lane_count':nlanes,'spacing_m':height/nlanes,'effective_ground_radius_at_2000m':math.sqrt(3000**2-2000**2),'altitude_is_not_action_control':True,'initial_altitude_m':3000,'explicit_lifetime_s':1800,'guaranteed_hidden_count':None},'input_scope':'public opening and later current-component reports only','actual_score':None}


def _pos(x):
    p=x.get('position')
    if isinstance(p,dict):return [p.get('x',0),p.get('y',0),p.get('z',0)]
    return list(p or [0,0,0])

def _role(k,v):return v.get('role') or v.get('task',{}).get('role') or k[0]

def _remaining(k,v,step):return float(v.get('remaining_life_s',max(0,1800-step) if _role(k,v)=='L' else max(0,3000-step)))

def _estimated_remaining(k,v,t):
    nominal=SPEED[_role(k,v)];vel=v.get('velocity')
    measured=math.sqrt(sum(float(q)**2 for q in vel)) if vel and len(vel)>=3 else nominal
    speed=max(.45*nominal,min(1.8*nominal,measured))
    return distance(_pos(v),_pos(t))/speed*1.08+12

def step_group(plan,current,targets,step,memory):
    """Group-local proposals. Input memory MUST be scoped to this communication group.
    Actual tasks in current override proposals/memory; prior_t03_make remains authority.
    Target absence never produces destruction confirmation. No unobserved HP used.
    Orders are task nominations; adapter dispatches only own current members.
    """
    p=plan['runtime'];mem=copy.deepcopy(memory or {});orders={};satreq=[]
    active={k:v for k,v in current.items() if v.get('visible',True) and v.get('health',1)>0}
    originals={u['unit_key']:u for u in plan['units']}
    # Tasks remain per-unit; this ledger is rebuilt exclusively from current component.
    assigned={}
    for k,v in active.items():
        task=v.get('task') or originals[k]['task']
        if task.get('kind')=='attack':assigned.setdefault(task.get('target_ref'),[]).append(k)
    mem['current_bindings']={ref:sorted(ks) for ref,ks in assigned.items()}
    active_sat=any(v.get('satellite_active',False) for v in active.values())
    if active_sat:mem['satellite_seen_active']=True
    sat=plan['satellite'];responsible=[sat['requester_unit_key']]+sat['backup_unit_keys']
    if not mem.get('satellite_seen_active'):
        for rank,k in enumerate(responsible):
            if k not in active:continue
            slot=sat['planned_trigger_s']+min(rank,15)*sat['backup_slot_seconds']
            if rank==0:
                hs=[(kk,v) for kk,v in active.items() if _role(kk,v)=='H' and (v.get('task') or {}).get('target_ref') in targets]
                approaching=sum(_estimated_remaining(kk,v,targets[v['task']['target_ref']])<=22 for kk,v in hs)
                if hs and approaching>=max(2,math.ceil(len(hs)*sat['adaptive_h_arrival_threshold'])):slot=max(slot-sat['adaptive_advance_seconds'],step)
            if step>=slot and step<=sat['latest_trigger_s'] and k not in mem.get('satellite_requested',[]) and 'satellite_requested_step' not in (active[k].get('task') or {}):
                # Lowest static-ranked currently present eligible member requests once.
                satreq.append(k);mem.setdefault('satellite_requested',[]).append(k);break
    # Route continuation: switch before a waypoint can terminate the native model.
    for k,v in active.items():
        task=copy.deepcopy(v.get('task') or originals[k]['task'])
        if task.get('kind')!='search':continue
        wps=task.get('waypoints') or originals[k]['task'].get('waypoints',[]);idx=task.get('waypoint_index',0)
        if idx<len(wps)-1 and distance(_pos(v),[*wps[idx],0])<=p['search_switch_m']:
            nxt=[*wps[idx+1],0]
            if distance(_pos(v),nxt)/100+p['search_turn_loss_s']<_remaining(k,v,step)-p['l_remaining_margin_s']:
                task['waypoint_index']=idx+1;orders[k]=task
    # Hidden target waves use actual successful bindings, no damage/death assumptions.
    hidden=mem.setdefault('hidden',{});livehidden=[]
    for ref,t in targets.items():
        if int(t.get('type',0))!=9500:continue
        age=step-float(t.get('timestamp_s',-1e9))
        if age>p['hidden_fresh_s']:continue
        candidates=[]
        for k,v in active.items():
            if _role(k,v)!='L':continue
            task=v.get('task') or originals[k]['task']
            if task.get('kind')=='attack':continue # Never strip another hidden task.
            eta=_estimated_remaining(k,v,t)
            if eta+p['l_remaining_margin_s']<_remaining(k,v,step):candidates.append((eta,k))
        nearest=min((x[0] for x in candidates),default=1e9)
        slack=1800-step-nearest
        priority=(slack if p['loss_mode']=='deep' or p['window_mode']=='early' else nearest,ref)
        livehidden.append((priority,ref,t,candidates))
    for _,ref,t,candidates in sorted(livehidden):
        state=hidden.setdefault(ref,{'ever_bound':[],'wave':0,'last_proposal_s':-1e9,'next_review_s':step})
        bound=assigned.get(ref,[])
        newly=[k for k in bound if k not in state['ever_bound']]
        if newly:
            state['ever_bound']=sorted(set(state['ever_bound']+newly));state['wave']=max(1,state['wave'])
            state['next_review_s']=max(state.get('next_review_s',step),step+max(_estimated_remaining(k,active[k],t) for k in newly)+p['hidden_confirm_grace_s'])
        if step-state['last_proposal_s']<p['proposal_retry_s']:continue
        # Expired group membership does not prove death; wait existing ETA lease.
        if state['ever_bound'] and step<state.get('next_review_s',step):continue
        if state['ever_bound'] and float(t.get('timestamp_s',-1))<state['next_review_s']-p['hidden_confirm_grace_s']:continue
        # Freshly surviving target after previous arrival estimate authorizes a new wave.
        if state['ever_bound']:
            outstanding=[k for k in bound if _estimated_remaining(k,active[k],t)>p['hidden_confirm_grace_s']]
            if outstanding:continue
            needed=p['hidden_topup']
        else:needed=max(0,p['hidden_batch']-len(bound))
        available=[]
        searchers=[k for k,v in active.items() if _role(k,v)=='L' and (v.get('task') or originals[k]['task']).get('kind')=='search' and k not in orders]
        keep=max(p['minimum_searchers_per_group'],math.ceil(len(searchers)*p['search_keep_fraction']))
        keepers=set(sorted(searchers,key=lambda k:(not (active[k].get('task') or originals[k]['task']).get('search_keeper',False),k))[:keep])
        for eta,k in sorted(candidates):
            if k in keepers or k in orders:continue
            if available and eta-available[0][0]>p['hidden_spread']:continue
            available.append((eta,k))
            if len(available)>=needed:break
        if needed<=0 or not available:continue
        # A partial last wave is permitted only near L expiry; otherwise wait for a full budget.
        if len(available)<needed and step<1450:continue
        for eta,k in available:
            orders[k]={'kind':'attack','target_ref':ref,'binding':'hidden_primary' if not state['ever_bound'] else 'hidden_reinforcement','role':'L','planned_eta_s':step+eta,'previous_search':copy.deepcopy(active[k].get('task') or originals[k]['task']),'requires_prior_t03_make':True}
        state['last_proposal_s']=step;state['wave']+=1
        state['proposed_only']=[k for _,k in available]
        # Never add proposals to ever_bound. Adapter must feed actual returned bindings.
    # Public budgets are sticky. Additional reassignment only among already surplus
    # own-target members, fresh destination, same-group coverage, and outside terminal lock.
    if step>=sat['planned_trigger_s']+sat['duration_s'] and step-mem.get('last_public_replan_s',-1e9)>=p['public_replan_interval_s']:
        mem['last_public_replan_s']=step
        pubs={t['ref']:t for t in plan['public_targets']};deficits=[];surplus=[]
        for ref,t in pubs.items():
            ks=[k for k in assigned.get(ref,[]) if k in active];is_m=t['type']==9400
            budget=sum((20 if active_sat else (9.6 if is_m else 12)) if _role(k,active[k])=='H' else (2.4 if is_m else 3) for k in ks)
            # Initial HP is a conservative requirement; no invented previous damage credit.
            deficit=max(0,t['initial_health']-budget)
            fresh=ref in targets and step-float(targets[ref].get('timestamp_s',-1e9))<=p['public_fresh_s']
            if deficit and fresh:deficits.append((-(5 if is_m else 2),ref,deficit))
            if budget<=t['initial_health']:continue
            for k in ks:
                role=_role(k,active[k]);d=(20 if active_sat else (9.6 if is_m else 12)) if role=='H' else (2.4 if is_m else 3)
                if budget-d>=t['initial_health'] and distance(_pos(active[k]),t['position'])>p['public_binding_lock_distance_m']:
                    surplus.append((k,ref,d,budget))
        useddonors=set();remaining_budget={ref:sum((20 if active_sat else (9.6 if t['type']==9400 else 12)) if _role(k,active[k])=='H' else (2.4 if t['type']==9400 else 3) for k in assigned.get(ref,[]) if k in active) for ref,t in pubs.items()}
        for _,dest,deficit in sorted(deficits):
            choices=[]
            for k,src,d,budget in surplus:
                if k in useddonors or k in orders or src==dest:continue
                if remaining_budget[src]-d<pubs[src]['initial_health']:continue
                eta=_estimated_remaining(k,active[k],targets[dest]);oldeta=distance(_pos(active[k]),pubs[src]['position'])/SPEED[_role(k,active[k])]
                if eta>oldeta*p['public_reassign_hysteresis'] or eta>_remaining(k,active[k],step)-60:continue
                choices.append((eta,k,src,d))
            for eta,k,src,d in sorted(choices):
                if deficit<=0:break
                if remaining_budget[src]-d<pubs[src]['initial_health']:continue
                role=_role(k,active[k]);targetdamage=(20 if active_sat else (9.6 if pubs[dest]['type']==9400 else 12)) if role=='H' else (2.4 if pubs[dest]['type']==9400 else 3)
                orders[k]={'kind':'attack','target_ref':dest,'binding':'public_reinforcement','role':role,'planned_eta_s':step+eta};useddonors.add(k);remaining_budget[src]-=d;deficit-=targetdamage
    return {'orders':orders,'request_satellite':satreq,'memory':mem}
