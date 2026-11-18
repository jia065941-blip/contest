"""Outcome-blind exploratory B binding, NOT a learned benefit selector.

Caller supplies only alternatives admitted by the unchanged native/compiler
guards, excludes the original B, and fixes A. This module performs no I/O and
cannot check actor identity, native legality, or the matched-branch protocol.
The same function must run on the actual new root's first eligible state.
"""
import math

VERSION = 'source_timing_contrast_v1'
BANDS = {'below_initial_requirement': 0, '0_to_10_percent_margin': 1,
         '10_to_50_percent_margin': 2, 'over_50_percent_margin': 3}


def _number(x):
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x):
        raise ValueError('required finite legal numeric feature unavailable')
    return float(x)


def _fraction(x):
    # Upstream None means zero assigned budget, not an unknown positive budget.
    x = 0.0 if x is None else _number(x)
    if not 0 <= x <= 1:
        raise ValueError('fraction outside [0, 1]')
    return x


def _fraction_bin(x):
    return sum(_fraction(x) > cut for cut in (0, .25, .5, .75))


def _phase(x):
    # Saturate outside the public planning window; extreme lateness is not
    # rewarded as greater exploratory contrast than any other late arrival.
    if x is None:
        return None
    x = _number(x)
    return -1 if x < 0 else 4 if x > 1 else min(3, int(x * 4))


def _distance(a, b):
    if a is None or b is None:
        return 0 if a is b else 1
    return abs(a - b)


def describe(features):
    """Small explicit allowlist; IDs, fingerprints and outcomes never enter."""
    try:
        if features['schema'] != 'anonymous_legal_relations_v2':
            raise ValueError('unsupported legal feature schema')
        cov = features['coverage_and_arrivals_by_choice']
        q = cov['AB']['Q']
        qtype = features['task_target_types']['Q']
        if qtype not in ('mother', 'sentry'):
            raise ValueError('unsupported donor target type')
        retained = BANDS[q['initial_hp_budget_band']]
        source = []
        for arm in ('B', 'AB'):
            row = cov[arm]['S']
            quantiles = row['nominal_budget_weighted_arrival_window_quantiles']
            if len(quantiles) != 3:
                raise ValueError('three nominal arrival quantiles required')
            source.extend(_phase(v) for v in quantiles)
            source.append(_fraction_bin(row['nominal_budget_before_window_end_fraction']))
        phase = 1 - _number(features['window_end_slack_over_window_duration']['B_to_S'])
        lifetime = _fraction(features['lifetime_slack_fraction']['B'])
        return {
            'donor_stratum': (qtype, retained,
                              _fraction_bin(q['nominal_budget_H_share']),
                              _fraction_bin(q['nominal_budget_before_window_end_fraction'])),
            'source_profile': (_phase(phase), *source),
            'B_arrival_phase': phase,
            'B_lifetime_slack': lifetime,
        }
    except (KeyError, TypeError) as exc:
        raise ValueError('missing or malformed authorized legal feature') from exc


def rank(original_features, candidate_features):
    """Lexicographic maximum: donor match, source contrast, retained cover.

Donor match is a coarse functional control, never a claim that Q or native
states are identical. The final earliest-arrival tie break avoids maximizing
unbounded lateness after the source-profile contrast has already saturated.
"""
    old, new = describe(original_features), describe(candidate_features)
    oldq, newq = old['donor_stratum'], new['donor_stratum']
    mismatch = tuple(int(a != b) for a, b in zip(oldq, newq))
    contrast = sum(_distance(a, b) for a, b in zip(old['source_profile'], new['source_profile']))
    score = (*(-x for x in mismatch), contrast,
             newq[1], newq[2], newq[3],
             -new['B_arrival_phase'], new['B_lifetime_slack'])
    return score, {'donor_mismatch_by_type_budget_Hshare_timely_share': list(mismatch),
                   'source_coarse_contrast': contrast,
                   'old_source_profile': list(old['source_profile']),
                   'selected_source_profile': list(new['source_profile']),
                   'selected_donor_stratum': list(newq)}


def choose(original_features, candidate_features):
    """Return one exploratory candidate and evidence of the outcome-blind rank.

candidate_features is a list of {'candidate_id': opaque token, 'features': ...}.
Tokens are checked for uniqueness and returned, never sorted or scored. Exact
rank ties use the supplied enumeration order, frozen by the evaluator before
any outcomes. Preserve this order in provenance. No malformed candidate is
silently skipped, and an empty pool is not a simulated KEEP observation.
"""
    describe(original_features)
    if not isinstance(candidate_features, list):
        raise ValueError('candidate list required')
    if not candidate_features:
        return {'candidate_id': None, 'reason': 'no_new_binding', 'version': VERSION,
                'candidate_count': 0, 'rank': None}
    seen, scored = set(), []
    for i, candidate in enumerate(candidate_features):
        token = candidate.get('candidate_id')
        if not isinstance(token, str) or not token or token in seen:
            raise ValueError('nonempty unique candidate tokens required')
        seen.add(token)
        score, audit = rank(original_features, candidate['features'])
        scored.append((score, i, token, audit))
    winner = max(scored, key=lambda item: item[0])
    score, index, token, audit = winner
    return {'candidate_id': token, 'reason': 'donor_matched_source_timing_contrast',
            'version': VERSION, 'candidate_count': len(scored), 'rank': list(score),
            'enumeration_index': index, 'rank_ties': sum(item[0] == score for item in scored),
            'source_contrast_available': audit['source_coarse_contrast'] > 0,
            'benefit_claim': False, **audit}
