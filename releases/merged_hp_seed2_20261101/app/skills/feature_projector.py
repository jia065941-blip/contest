"""Anonymous legal-state relations for D2. Pure, standard library, no outcome input.

Fixed semantic normalizations, not score-fitted thresholds. Numeric outputs are
approximate relations, never exact-match retrieval keys. This is not a legality
validator or a newly deployed selector. Native/compiler checks remain required.
"""
import math

CHOICES = {'KEEP': 0, 'A': 1, 'B': 2, 'AB': 3}


def _number(value):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError('non-finite legal observation')
    return value


def _r(value):
    return round(_number(value), 2) + 0.0


def _eta(unit, target):
    speed = {'H': 200.0, 'M': 120.0}[unit['role']]
    return math.dist(unit['position'], target['position']) / speed * 1.08 + 12.0


def _damage(role, kind):
    if role not in ('H', 'M') or kind not in (9400, 9600):
        return 0.0
    return {'H': {9400: 9.6, 9600: 12.0}, 'M': {9400: 2.4, 9600: 3.0}}[role][kind]


def _phase(x):
    if x < 0:
        return 'before'
    if x > 1:
        return 'after'
    return ('first_quarter', 'second_quarter', 'third_quarter', 'fourth_quarter')[min(3, int(x * 4))]


def _coverage_band(x):
    if x < 1 - 1e-8:
        return 'below_initial_requirement'
    if x <= 1.1 + 1e-8:
        return '0_to_10_percent_margin'
    if x <= 1.5 + 1e-8:
        return '10_to_50_percent_margin'
    return 'over_50_percent_margin'


def _fresh(target):
    age = target.get('age_s')
    return bool(target.get('freshness_known')) and age is not None and 0 <= age <= 5


def features(ctx, roles):
    """Project ONLY explicit legal fields and fixed roles; ignore all extra fields.

    Input keys/IDs are join keys internally, never output. No headings, velocity,
    hidden remaining HP, engine state, map name or future outcome is accessed.
    Same-task distributions describe nominal budget, not hits or communication.
    """
    if len(roles) != 2:
        raise ValueError('requires fixed A/B roles')
    units, targets, req = ctx['units'], ctx['targets'], ctx['public_requirements']
    a, b = (r['unit'] for r in roles)
    A, B = units[a], units[b]
    D, S, Q = roles[0]['target'], A['target'], B['target']
    orders = ctx.get('baseline_orders', {})
    if a == b or A['role'] != 'H' or B['role'] != 'H' or len({D, S, Q}) != 3 or roles[1]['target'] != S:
        raise ValueError('invalid two-H source-takeover binding')
    if a in orders or b in orders:
        raise ValueError('baseline order conflict')
    refs = {'D': D, 'S': S, 'Q': Q}
    if any(targets[t]['type'] not in (9400, 9600) for t in refs.values()):
        raise ValueError('unsupported target')
    start, end = (_number(ctx['satellite_window'][x]) for x in ('start', 'end'))
    duration = end - start
    if duration <= 0:
        raise ValueError('invalid planning window')
    now = _number(ctx['time_s'])
    AD, AS, BS, BQ = _eta(A, targets[D]), _eta(A, targets[S]), _eta(B, targets[S]), _eta(B, targets[Q])
    phase = (now + AD - start) / duration
    cover = {}
    for choice, mask in CHOICES.items():
        buckets = {x: [] for x in refs}
        for uid, u in units.items():
            if u.get('role') not in ('H', 'M'):
                continue
            t = orders.get(uid, {}).get('target_ref', u.get('target'))
            if uid == a and mask & 1:
                t = D
            if uid == b and mask & 2:
                t = S
            for label, ref in refs.items():
                if t == ref:
                    damage = _damage(u['role'], req[ref]['type'])
                    buckets[label].append((_eta(u, targets[ref]), damage, u['role']))
        cover[choice] = {}
        for label, arrivals in buckets.items():
            ref = refs[label]
            health = _number(req[ref]['initial_health'])
            if health <= 0:
                raise ValueError('invalid public initial health')
            total = math.fsum(x[1] for x in arrivals)
            ordered = sorted(arrivals)
            quantiles = []
            for quantile in (.25, .5, .75):
                accumulated, value = 0.0, None
                for eta, damage, _ in ordered:
                    accumulated += damage
                    if accumulated + 1e-8 >= total * quantile:
                        value = _r((now + eta - start) / duration)
                        break
                quantiles.append(value)
            cover[choice][label] = {
                'initial_hp_budget_band': _coverage_band(total / health),
                'nominal_budget_weighted_arrival_window_quantiles': quantiles,
                'nominal_budget_before_window_end_fraction': _r(math.fsum(d for eta, d, _ in arrivals if now + eta <= end) / total) if total else None,
                'nominal_budget_H_share': _r(math.fsum(d for _, d, role in arrivals if role == 'H') / total) if total else None,
            }
    return {
        'schema': 'anonymous_legal_relations_v2',
        'task_target_types': {label: {9400: 'mother', 9600: 'sentry'}[targets[t]['type']] for label, t in refs.items()},
        'observations': {label: 'fresh' if _fresh(targets[t]) else 'freshness_unknown_or_stale' for label, t in refs.items()},
        'satellite_observed': bool(A.get('satellite_active')),
        'trigger_window_phase': _phase((now - start) / duration),
        'A_destination_arrival_window_phase': _r(phase),
        'A_destination_arrival_window_quarter': _phase(phase),
        'window_end_slack_over_window_duration': {'A_to_D': _r(1 - phase), 'B_to_S': _r((end - now - BS) / duration)},
        'travel_time_ratios': {'A_new_over_original': _r(AD / AS), 'B_new_over_original': _r(BS / BQ), 'B_at_source_over_A_original': _r(BS / AS)},
        'lifetime_slack_fraction': {'A': _r((A['remaining_s'] - AD - 60) / A['remaining_s']), 'B': _r((B['remaining_s'] - BS - 60) / B['remaining_s'])},
        'coverage_and_arrivals_by_choice': cover,
        'limits': 'Nominal distance ETA and public initial-HP budget only; no remaining HP, hit certainty, heading, velocity or interception probability. Planned window is not observed satellite activation.',
    }


def semantic_signature(feature):
    """Deliberately coarse review view, NOT a policy or a uniqueness key.

    Equal-quarter timing uses fixed geometric meaning; no outcome-fitted bins.
    Omits continuous timing detail so coarse collisions remain visible in audits.
    """
    return {key: feature[key] for key in ('task_target_types', 'observations', 'satellite_observed', 'trigger_window_phase', 'A_destination_arrival_window_quarter')} | {
        'coverage_bands': {choice: {label: row['initial_hp_budget_band'] for label, row in rows.items()} for choice, rows in feature['coverage_and_arrivals_by_choice'].items()}}
