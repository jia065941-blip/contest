"""Frozen development return tree; no data/model-file dependency."""
import math
ARMS = ('KEEP', 'A', 'B', 'AB')
TOLERANCE = 1e-10
FEATURE_NAMES = ('A_arrival_quarter', 'A_destination_budget', 'A_source_retained_budget', 'B_source_retained_budget', 'B_takeover_time_ratio', 'AB_source_timely_fraction', 'KEEP_destination_timely_fraction', 'B_source_is_sentry', 'B_arrival_window_quarter', 'A_source_timely_share', 'AB_source_arrival_spread', 'B_donor_retained_H_share')
BANDS = {'below_initial_requirement': 0, '0_to_10_percent_margin': 1, '10_to_50_percent_margin': 2, 'over_50_percent_margin': 3}
FORBIDDEN_KEYS = frozenset(('case_id', 'cluster_id', 'evaluator_state', 'feature_fingerprint', 'final_score', 'fingerprint', 'future_outcome', 'map', 'map_id', 'map_name', 'outcome', 'outcomes', 'remaining_hp', 'returns', 'seed', 'seed_id'))

def _number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError('required numeric legal feature unavailable')
    value = float(value)
    if not math.isfinite(value):
        raise ValueError('non-finite legal feature')
    return value

def _bin(value, cuts, low=None, high=None):
    value = _number(value)
    if (low is not None and value < low) or (high is not None and value > high):
        raise ValueError('legal feature outside its semantic domain')
    return sum(value > cut for cut in cuts)

def _phase(value):
    value = _number(value)
    return -1 if value < 0 else 4 if value > 1 else min(3, int(value * 4))

def _timely_fraction(value):
    # Preserve the old eight features exactly: no extra zero cut here.
    return _bin(0.0 if value is None else value, (.25, .5, .75), 0, 1)

def _share(value):
    # Upstream None denotes zero assigned nominal budget, not unknown HP.
    return _bin(0.0 if value is None else value, (0, .25, .5, .75), 0, 1)

def _spread(quantiles):
    if not isinstance(quantiles, (list, tuple)) or len(quantiles) != 3:
        raise ValueError('three nominal arrival quantiles required')
    if all(value is None for value in quantiles):
        return -1
    if any(value is None for value in quantiles):
        raise ValueError('partially missing arrival quantiles')
    low, middle, high = map(_number, quantiles)
    if not low <= middle <= high:
        raise ValueError('unordered arrival quantiles')
    return _bin(high - low, (.25, .5, 1), 0)

def _check_legal_container(value):
    if isinstance(value, dict):
        if FORBIDDEN_KEYS.intersection(value):
            raise ValueError('evaluator identity, outcome or hidden state in policy input')
        for child in value.values():
            _check_legal_container(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _check_legal_container(child)

def project(features):
    """Explicit allowlist; unused legal fields cannot enter model fitting."""
    _check_legal_container(features)
    try:
        if features['schema'] != 'anonymous_legal_relations_v2':
            raise ValueError('unsupported legal feature schema')
        cov = features['coverage_and_arrivals_by_choice']
        qtype = features['task_target_types']['Q']
        if qtype not in ('mother', 'sentry'):
            raise ValueError('unsupported public target type')
        return {
            'A_arrival_quarter': _phase(features['A_destination_arrival_window_phase']),
            'A_destination_budget': BANDS[cov['A']['D']['initial_hp_budget_band']],
            'A_source_retained_budget': BANDS[cov['A']['S']['initial_hp_budget_band']],
            'B_source_retained_budget': BANDS[cov['B']['Q']['initial_hp_budget_band']],
            'B_takeover_time_ratio': _bin(features['travel_time_ratios']['B_at_source_over_A_original'], (.5, 1, 2), 0),
            'AB_source_timely_fraction': _timely_fraction(cov['AB']['S']['nominal_budget_before_window_end_fraction']),
            'KEEP_destination_timely_fraction': _timely_fraction(cov['KEEP']['D']['nominal_budget_before_window_end_fraction']),
            'B_source_is_sentry': int(qtype == 'sentry'),
            'B_arrival_window_quarter': _phase(1 - _number(features['window_end_slack_over_window_duration']['B_to_S'])),
            'A_source_timely_share': _share(cov['A']['S']['nominal_budget_before_window_end_fraction']),
            'AB_source_arrival_spread': _spread(cov['AB']['S']['nominal_budget_weighted_arrival_window_quantiles']),
            'B_donor_retained_H_share': _share(cov['AB']['Q']['nominal_budget_H_share']),
        }
    except (KeyError, TypeError, IndexError) as exc:
        raise ValueError('missing or malformed authorized legal feature') from exc

def choose(advantages):
    values = {arm: _number(advantages[arm]) for arm in ARMS}
    maximum = max(values.values())
    return next(arm for arm in ARMS if maximum - values[arm] <= TOLERANCE)

def output(gains, support):
    advantages = dict(zip(ARMS, [0.0] + list(gains)))
    return {'choice': choose(advantages), 'predicted_advantages': advantages,
            'leaf_support': support}

def historical_base_predict(features):
    x = project(features)
    if x['B_takeover_time_ratio'] <= 2.5:
        return output([-1.2058200148937404, 0.0, -0.378071833648394], 11)
    else:
        return output([0.6182476943346498, 0.0, 0.6182476943346498], 12)

def _extra_number(v):
    if isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v):
        raise ValueError('finite legal numeric observation required')
    return float(v)

def _extra_share(v):
    v = 0.0 if v is None else _extra_number(v)
    if not 0 <= v <= 1:
        raise ValueError('share outside domain')
    return v

def _extra_quantiles(row):
    q = row['nominal_budget_weighted_arrival_window_quantiles']
    if not isinstance(q,(list,tuple)) or len(q)!=3:
        raise ValueError('complete quantile triple required')
    if all(v is None for v in q):
        if row['nominal_budget_H_share'] is not None or row['nominal_budget_before_window_end_fraction'] is not None:
            raise ValueError('inconsistent empty nominal budget')
        return None
    if any(v is None for v in q):
        raise ValueError('partially missing quantiles')
    q = tuple(_extra_number(v) for v in q)
    if not q[0] <= q[1] <= q[2]:
        raise ValueError('unordered arrival quantiles')
    if row['nominal_budget_H_share'] is None or row['nominal_budget_before_window_end_fraction'] is None:
        raise ValueError('missing shares for nonempty budget')
    return q

def _extra_bin(v,cuts):
    return sum(v>c for c in cuts)

def _extra_phase(v):
    return -1 if v<0 else 4 if v>1 else min(3,int(v*4))

def extra_project(features):
    _check_legal_container(features)
    if features['schema']!='anonymous_legal_relations_v2':
        raise ValueError('unsupported feature schema')
    source_type = features['task_target_types']['S']
    if source_type not in ('mother','sentry'):
        raise ValueError('invalid public source type')
    c = features['coverage_and_arrivals_by_choice']
    keep_s, a_s, keep_d = c['KEEP']['S'], c['A']['S'], c['KEEP']['D']
    sk, sa, dk = map(_extra_quantiles,(keep_s,a_s,keep_d))
    if sk is None:
        raise ValueError('fixed A requires assigned source coverage before withdrawal')
    cuts=(0,.25,.5,.75)
    h0=_extra_share(keep_s['nominal_budget_H_share'])
    h1=_extra_share(a_s['nominal_budget_H_share'])
    t0=_extra_share(keep_s['nominal_budget_before_window_end_fraction'])
    t1=_extra_share(a_s['nominal_budget_before_window_end_fraction'])
    ad=_extra_number(features['A_destination_arrival_window_phase'])
    return {
        'A_source_is_sentry':int(source_type=='sentry'),
        'A_source_H_share_loss':_extra_bin(round(h0-h1,2),cuts),
        'A_source_retained_H_share':_extra_bin(h1,cuts),
        'A_source_timely_fraction_loss':_extra_bin(round(max(0,t0-t1),2),cuts),
        'A_source_lower_quartile_delay':5 if sa is None else _extra_bin(round(max(0,sa[0]-sk[0]),2),(0,.25,.5,1)),
        'KEEP_destination_H_share':-1 if dk is None else _extra_bin(_extra_share(keep_d['nominal_budget_H_share']),cuts),
        'KEEP_destination_median_phase':5 if dk is None else _extra_phase(dk[1]),
        'destination_median_advance_by_A':7 if dk is None else _extra_bin(round(dk[1]-ad,2),(-.5,-.25,0,.25,.5,1)),
    }

def predict(features):
    x = project(features)
    x.update(extra_project(features))
    if x['B_takeover_time_ratio'] <= 2.5:
        return {'choice': 'KEEP'}
    else:
        if x['A_source_retained_budget'] <= 1.5:
            return {'choice': 'A'}
        else:
            return {'choice': 'KEEP'}
