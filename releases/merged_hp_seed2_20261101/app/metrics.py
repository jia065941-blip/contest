"""Compact evaluator-only event accounting. No policy decisions or simulator edits."""
from collections import Counter,defaultdict

ROLES={21000:'H',21001:'M',21002:'L'}
TARGETS={9400:'mother',9600:'sentry',9500:'hidden'}
INTERCEPTOR=24000


def ratio(n,d):return n/d if d else None


class Ledger:
    def __init__(self,entities):
        # Initial inventory, not inferred from the units that happen to survive.
        self.entities={int(e['id']):dict(e) for e in entities}
        self.launched=set();self.intercepted=set();self.interceptor_damaged=set()
        self.pairs=defaultdict(lambda:dict(events=0,actual_damage=0.,nominal_damage=0.,first_time=None,last_time=None))
        self.zero_effect=Counter();self.damage_source_unresolved=0;self.other_target_damage=Counter()

    def launch(self,unit):
        if self.entities.get(unit,{}).get('type') in ROLES:self.launched.add(unit)

    def damage(self,source,victim,before,after,nominal,time_s):
        src=self.entities.get(source);dst=self.entities.get(victim)
        if src is None or dst is None:self.damage_source_unresolved+=1;return
        delta=max(0.,float(before)-float(after))
        if src['type']==INTERCEPTOR and dst['type'] in ROLES:
            if delta>0:self.interceptor_damaged.add(victim)
            if before>0 and after<=0:self.intercepted.add(victim)
        if dst['type'] in TARGETS and src['type'] not in ROLES and delta>0:self.other_target_damage[victim]+=delta
        if src['type'] not in ROLES or dst['type'] not in TARGETS:return
        if delta<=0:self.zero_effect[source]+=1;return
        p=self.pairs[(source,victim)];p['events']+=1;p['actual_damage']+=delta;p['nominal_damage']+=nominal
        if p['first_time'] is None:p['first_time']=time_s
        p['last_time']=time_s

    def finalize(self,final_health):
        targets={u:e for u,e in self.entities.items() if e['type'] in TARGETS}
        if not set(targets)<=set(final_health):raise ValueError('All objectives need final health for score attribution')
        if any(u not in self.launched for u,_ in self.pairs):raise ValueError('Damage source lacks observed launch receipt')
        target_rows=[]
        for t,e in targets.items():
            pairs={u:v for (u,victim),v in self.pairs.items() if victim==t}
            target_rows.append(dict(target=t,type=TARGETS[e['type']],initial_health=e['health'],final_health=final_health[t],
                destroyed=final_health[t]<=0,weight=e.get('weight'),unique_contributors=len(pairs),other_source_damage=self.other_target_damage[t],
                damage_by_role={role:sum(v['actual_damage'] for u,v in pairs.items() if ROLES[self.entities[u]['type']]==role) for role in ROLES.values()}))
        roles={}
        for typ,role in ROLES.items():
            allocated={u for u,e in self.entities.items() if e['type']==typ};launched=allocated&self.launched
            hits={u for u,t in self.pairs if u in allocated}
            score={u for u,t in self.pairs if u in allocated and final_health[t]<=0}
            partial={u for u,t in self.pairs if u in allocated and final_health[t]>0}
            intercepted=allocated&self.intercepted;table={}
            for target_type,name in TARGETS.items():
                pairs={(u,t):v for (u,t),v in self.pairs.items() if u in allocated and self.entities[t]['type']==target_type}
                successful={u for u,t in pairs if final_health[t]<=0};stranded={u for u,t in pairs if final_health[t]>0}
                table[name]=dict(unique_hitting_units=len({u for u,t in pairs}),hit_events=sum(v['events'] for v in pairs.values()),
                    unique_targets_hit=len({t for u,t in pairs}),effective_damage=sum(v['actual_damage'] for v in pairs.values()),
                    nominal_damage=sum(v['nominal_damage'] for v in pairs.values()),scoring_target_contributors=len(successful),
                    surviving_target_contributors=len(stranded),damage_on_surviving_targets=sum(v['actual_damage'] for (u,t),v in pairs.items() if final_health[t]>0))
            roles[role]=dict(allocated=len(allocated),launched=len(launched),not_launched=len(allocated-launched),
                unique_hitting_units=len(hits),intercepted=len(intercepted),intercepted_after_launch=len(intercepted&launched),intercepted_before_launch=len(intercepted-launched),interceptor_damaged=len(allocated&self.interceptor_damaged),
                scoring_target_contributors=len(score),surviving_target_contributors=len(partial),only_surviving_target_contributors=len(partial-score),
                contributors_to_both=len(partial&score),no_objective_damage=len(launched-hits),
                no_objective_damage_and_intercepted=len((launched-hits)&intercepted),
                no_objective_damage_other_outcome=len((launched-hits)-intercepted),
                zero_effect_damage_commands=sum(v for u,v in self.zero_effect.items() if u in allocated),
                hit_fraction_of_launched=ratio(len(hits),len(launched)),intercepted_fraction_of_launched=ratio(len(intercepted&launched),len(launched)),
                scoring_contributor_fraction_of_launched=ratio(len(score),len(launched)),scoring_contributor_fraction_of_allocated=ratio(len(score),len(allocated)),
                surviving_target_contributor_fraction_of_launched=ratio(len(partial),len(launched)),by_target_type=table)
        return dict(roles=roles,objectives=target_rows,unresolved_damage_sources=self.damage_source_unresolved,
            note='Observed damage attribution, not causal necessity or the score that intercepted units would otherwise earn.')
