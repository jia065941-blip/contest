"""D2 pure policy/merge functions; no simulator access or external model calls.

Only H can fill A/B in this version. M remains visible for conservative budgets.
Quality guards select the full proposal; they NEVER gate factorial ablations.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
from functools import lru_cache
from pathlib import Path

HERE = Path(__file__).resolve().parent
ANCHOR_PATH = HERE / 'anchor.json'
ANCHOR_SHA256 = 'cd7d9b0b96501e4a5bab95b12c12c7fe7b4d4c3a7c2ade5f7d5cef36dd8f5779'
LEGACY_PATH = HERE / 'anchor_runtime.py'
NAME = '缺口改派—源任务显式接替'
ARMS = ('m0', 'm1', 'm2', 'm3')
MAX_PROPOSAL_TIME_S = 900


@lru_cache(maxsize=1)
def _legacy():
    spec = importlib.util.spec_from_file_location('astra_d2_frozen_anchor_runtime', LEGACY_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=1)
def load_anchor():
    raw = ANCHOR_PATH.read_bytes()
    if hashlib.sha256(raw).hexdigest() != ANCHOR_SHA256:
        raise ValueError('frozen anchor file hash mismatch')
    source = json.loads(raw)['code']
    return _legacy().compile_skill(source), {
        'file_sha256': ANCHOR_SHA256,
        'function_source_sha256': hashlib.sha256(source.encode()).hexdigest(),
        'code_author': 'preserved Harness compiler of historical single-H mechanism',
        'd2_author': 'GPT-6 Astra high Codex subagent',
    }


def project_context(ctx):
    """Allowlist the authorized context; unknown evaluator fields cannot flow in."""
    unit_fields = ('role', 'position', 'kind', 'target', 'remaining_s',
                   'satellite_active', 'search_keeper')
    target_fields = ('type', 'position', 'age_s', 'freshness_known', 'initial_public_health')
    return {
        'time_s': ctx['time_s'],
        'units': {u: {k: copy.deepcopy(v[k]) for k in unit_fields if k in v}
                  for u, v in ctx['units'].items() if v.get('role') in ('H', 'M')},
        'targets': {t: {k: copy.deepcopy(v[k]) for k in target_fields if k in v}
                    for t, v in ctx['targets'].items() if t in ctx['public_requirements']},
        'public_requirements': {t: {'type': v['type'], 'initial_health': v['initial_health']}
                                for t, v in ctx['public_requirements'].items()},
        'satellite_window': {k: ctx['satellite_window'][k] for k in ('start', 'end')},
        'baseline_orders': {u: {k: v[k] for k in ('kind', 'target_ref') if k in v}
                            for u, v in ctx.get('baseline_orders', {}).items()},
    }


def damage_budget(role, target_type):
    if role not in ('H', 'M') or target_type not in (9400, 9600):
        raise ValueError('unsupported damage budget type')
    return (9.6 if target_type == 9400 else 12.0) if role == 'H' else (2.4 if target_type == 9400 else 3.0)


def _eta(unit, target):
    return math.dist(unit['position'], target['position']) / 200.0 * 1.08 + 12.0


def _fresh(target):
    age = target.get('age_s')
    return bool(target.get('freshness_known')) and age is not None and 0 <= age <= 5


def _budget(ctx, orders):
    totals = {t: 0.0 for t in ctx['public_requirements']}
    for u, v in ctx['units'].items():
        target = orders.get(u, {}).get('target_ref', v.get('target'))
        req = ctx['public_requirements'].get(target)
        if req and req['type'] in (9400, 9600) and v['role'] in ('H', 'M'):
            totals[target] += damage_budget(v['role'], req['type'])
    return totals


def validate_roles(roles, ctx):
    """Command/authorized-group checks. No freshness/ETA/coverage quality gates.

    The engine remains final authority for native legality. Context contains only
    currently active authorized group members and legitimately known public tracks.
    Baseline same-step priority is an adapter contract, not an engine physics rule.
    """
    if not isinstance(roles, list) or len(roles) > 2:
        raise ValueError('D2 accepts at most two roles')
    seen = set()
    for r in roles:
        if set(r) != {'unit', 'target', 'duty'}:
            raise ValueError('invalid role fields')
        u, target = r['unit'], r['target']
        if u in seen or u not in ctx['units'] or target not in ctx['targets']:
            raise ValueError('nonlocal or duplicate binding')
        seen.add(u)
        if ctx['units'][u]['role'] != 'H':
            raise ValueError('this version implements H roles only')
        if ctx['targets'][target]['type'] not in (9400, 9600):
            raise ValueError('official attack role/target mismatch')
        if u in ctx.get('baseline_orders', {}):
            raise ValueError('baseline same-tick order has adapter priority')
        if ctx['units'][u].get('target') == target:
            raise ValueError('role must change its attack target')
        if not isinstance(r['duty'], str) or not r['duty']:
            raise ValueError('missing role duty')


def merge(response, roles, enabled, ctx):
    """Independent one-shot role commits; disabled entities keep all baseline orders."""
    if not isinstance(enabled, int) or enabled < 0 or enabled >= 1 << len(roles):
        raise ValueError('invalid role mask')
    chosen = [r for i, r in enumerate(roles) if enabled & (1 << i)]
    validate_roles(chosen, ctx)
    result = copy.deepcopy(response)
    orders = result['orders']
    for r in chosen:
        orders[r['unit']] = {
            'kind': 'attack', 'target_ref': r['target'], 'binding': 'astra_d2_source_handoff', 'role': 'H',
        }
    return result


def coverage_audit(response, roles, enabled, ctx):
    """Report tactical budget shortfalls without suppressing any legal action."""
    merged = merge(response, roles, enabled, ctx)
    totals = _budget(ctx, merged['orders'])
    sources = sorted({ctx['units'][r['unit']].get('target') for r in roles} - {None})
    rows = {t: {'estimated_damage': totals.get(t, 0.0),
                'initial_public_requirement': ctx['public_requirements'][t]['initial_health'],
                'quality_guard_violated': totals.get(t, 0.0) + 1e-8 < ctx['public_requirements'][t]['initial_health']}
            for t in sources if t in ctx['public_requirements']}
    return {'source_budgets': rows, 'quality_guard_violated': any(v['quality_guard_violated'] for v in rows.values()),
            'legal_action_rejection': False, 'used_as_ablation_gate': False}


def propose(ctx, anchor_fn=None):
    """Return frozen A/B proposal from legal data, or an explicit no-binding reason."""
    ctx = project_context(ctx)
    t = ctx['time_s']
    if t > MAX_PROPOSAL_TIME_S:
        return {'roles': [], 'reason': 'outside_frozen_runtime900_second_limit'}
    if t % 15:
        return {'roles': [], 'reason': 'outside15_second_grid'}
    if anchor_fn is None:
        anchor_fn, _ = load_anchor()
    anchor = anchor_fn(copy.deepcopy(ctx))
    if not anchor.get('roles'):
        return {'roles': [], 'reason': 'anchor_no_proposal', 'anchor_reason': anchor.get('reason')}
    if len(anchor['roles']) != 1:
        raise ValueError('single-H anchor changed role count')
    a = copy.deepcopy(anchor['roles'][0])
    validate_roles([a], ctx)
    A, D = a['unit'], a['target']
    S = ctx['units'][A].get('target')
    if S not in ctx['targets'] or not _fresh(ctx['targets'][S]):
        return {'roles': [], 'reason': 'source_not_fresh_for_B', 'anchor_roles': [a]}
    totals = _budget(ctx, ctx['baseline_orders'])
    choices = []
    for u, v in ctx['units'].items():
        Q = v.get('target')
        if u == A or v['role'] != 'H' or u in ctx['baseline_orders'] or Q in (S, D):
            continue
        req = ctx['public_requirements'].get(Q)
        if not req or req['type'] not in (9400, 9600):
            continue
        if totals.get(Q, 0.0) - damage_budget('H', req['type']) + 1e-8 < req['initial_health']:
            continue
        eta = _eta(v, ctx['targets'][S])
        if eta >= v['remaining_s'] - 60:
            continue
        choices.append((eta, u, Q))
    if not choices:
        return {'roles': [], 'reason': 'no_eligible_binding_at_tick', 'anchor_roles': [a]}
    eta, B, Q = min(choices)
    b = {'unit': B, 'target': S, 'duty': '接替A原目标；从独立盈余任务Q改派至S'}
    roles = [a, b]
    validate_roles(roles, ctx)
    # The full proposal alone must satisfy tactical source protection.
    quality = coverage_audit({'orders': ctx['baseline_orders']}, roles, 3, ctx)
    if quality['quality_guard_violated']:
        return {'roles': [], 'reason': 'full_proposal_source_quality_guard', 'quality': quality}
    return {'roles': roles, 'reason': 'first joint eligible H redirect and H source takeover',
            'D_S_Q_refs': {'D': D, 'S': S, 'Q': Q}, 'B_eta_s': eta,
            'B_eligible_count': len(choices), 'full_quality': quality}


def decomposition_proof(response, roles, ctx):
    """Certify disjoint assignment composition, without another simulation arm."""
    validate_roles(roles, ctx)
    if len(roles) != 2:
        raise ValueError('D2 proof requires two roles')
    independent = copy.deepcopy(response)
    for role in roles:
        independent = merge(independent, [role], 1, ctx)
    joint = merge(response, roles, 3, ctx)
    equal = independent == joint
    if not equal:
        raise ValueError('independent composition identity failed')
    return {'status': 'statically_decomposable', 'independent_A_plus_B_equals_joint': True,
            'basis': 'disjoint one-shot order writes, identical persistent task fields and unchanged other response fields',
            'duplicate_composition_arm_required': False, 'irreducible_cooperation_claim': False}


INTERVENTION_STATUS = {
    'peer_cut': {'status': 'not_applicable', 'reason': 'Do not remove group resources; D2 has no independent online message consumer'},
    'peer_eta_nominal': {'status': 'not_applicable', 'reason': 'Only nominal role speed is consumed; no measured peer velocity dependency'},
    'peer_target_public_prior': {'status': 'not_evaluated', 'reason': 'Would alter prebinding selection; v1 fixes roles and commits independent operations without postbinding coordinate consumer'},
    'delay_A_5': {'status': 'not_implemented_or_evaluated', 'reason': 'Current version only same-tick4-arm factorial'},
    'delay_B_5': {'status': 'not_implemented_or_evaluated', 'reason': 'Current version only same-tick4-arm factorial'},
}


class EventBinder:
    """First jointly eligible event across authorized observations; never rebind."""
    def __init__(self, anchor_fn=None):
        self.anchor_fn = anchor_fn
        self.event = None
        self.last_proposal = None
        self.attempts = set()

    def consider(self, ctx, response):
        if self.event is not None or ctx['time_s'] > MAX_PROPOSAL_TIME_S or ctx['time_s'] % 15:
            return None
        key = (ctx['time_s'], tuple(sorted(ctx['units'])))
        if key in self.attempts:
            return None
        self.attempts.add(key)
        c = project_context(ctx)
        self.last_proposal = propose(c, self.anchor_fn)
        if not self.last_proposal['roles']:
            return None
        roles = self.last_proposal['roles']
        audits = {}
        for mask in range(4):
            merge(response, roles, mask, c)
            audits['m' + str(mask)] = coverage_audit(response, roles, mask, c)
        self.event = {
            'time_s': c['time_s'], 'roles': copy.deepcopy(roles),
            'reason': self.last_proposal['reason'], 'legal_context': c,
            'D_S_Q_refs': self.last_proposal['D_S_Q_refs'],
            'binding': {'fixed_before_fork': True, 'B_eligible_count': self.last_proposal['B_eligible_count'],
                        'B_eta_s': self.last_proposal['B_eta_s'], 'M_roles_implemented': False},
            'subset_quality_audits': audits,
            'static_decomposition': decomposition_proof(response, roles, c),
            'information_and_timing_interventions': copy.deepcopy(INTERVENTION_STATUS),
        }
        return copy.deepcopy(self.event)
