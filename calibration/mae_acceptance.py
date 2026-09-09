"""Independent raw-trajectory MAE assessment; never imports the search score.

Macro numerical acceptance is not full taskbook or real-world acceptance.
Missing/failed requested paths are never dropped to obtain an average.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

AXES = ('inflation', 'real_gdp_growth', 'consumption_gdp', 'investment_gdp')


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def validate_contract(c):
    required = {'schema_version', 'target_kind', 'evidence_level', 'horizon',
                'expected_seeds', 'periods', 'thresholds', 'targets', 'units'}
    if not isinstance(c, dict) or set(c) != required:
        raise ValueError('invalid MAE contract fields')
    if type(c['schema_version']) is not int or c['schema_version'] != 1:
        raise ValueError('unsupported schema')
    if c['target_kind'] not in ('synthetic', 'development_observational'):
        raise ValueError('only explicitly developmental target kinds are supported')
    if c['evidence_level'] not in ('development_predeclared', 'post_hoc_development'):
        raise ValueError('explicit development evidence level required')
    if c['units'] != 'fraction':
        raise ValueError('all rates/shares and thresholds must be fractions, not percent')
    h, seeds, periods = c['horizon'], c['expected_seeds'], c['periods']
    if type(h) is not int or not 2 <= h <= 27:
        raise ValueError('invalid horizon')
    if (not isinstance(seeds, list) or not seeds or
            any(type(s) is not int or not 0 <= s < 2**32 for s in seeds) or len(set(seeds)) != len(seeds)):
        raise ValueError('unique expected uint32 seeds required')
    if (not isinstance(periods, list) or not periods or
            any(type(t) is not int or not 2 <= t <= h for t in periods) or periods != sorted(set(periods))):
        raise ValueError('ordered unique scoring periods within 2..horizon required; initialization excluded')
    for name in ('thresholds', 'targets'):
        if not isinstance(c[name], dict) or set(c[name]) != set(AXES):
            raise ValueError(f'{name} must explicitly specify all axes; missing targets use null')
    for axis in AXES:
        if not finite(c['thresholds'][axis]) or c['thresholds'][axis] < 0:
            raise ValueError('finite nonnegative MAE threshold required')
        values = c['targets'][axis]
        if values is not None and (not isinstance(values, dict) or set(values) != {str(t) for t in periods}
                                   or any(not finite(v) for v in values.values())):
            raise ValueError('target period coverage must be exact, or the whole axis must be null')


def assess_mae(result, contract):
    validate_contract(contract)
    c = contract
    report = {'schema_version': 1, 'target_kind': c['target_kind'], 'evidence_level': c['evidence_level'],
              'contract_sha256': hashlib.sha256(json.dumps(c, sort_keys=True, allow_nan=False).encode()).hexdigest(),
              'macro_mae_status': 'INVALID', 'axes': {}, 'path_integrity_errors': [],
              'expected_seed_count': len(c['expected_seeds']), 'failed_paths_excluded': False,
              'initialization_period_excluded': True, 'economic_validity_certified': False,
              'taskbook_acceptance': 'NOT_EVALUATED',
              'not_checked': ['weighted_micro_comparability', 'all_sector_equations',
                              'formal_30_seed_stability', 'historical_measurement_and_policy_alignment'],
              'uncertainty_note': 'Across-seed sample SD and min/max only; not a confidence interval. Target-sample uncertainty is not estimated.'}
    errors = report['path_integrity_errors']
    if not isinstance(result, dict) or result.get('valid') is not True:
        errors.append('backend_not_valid')
    paths = result.get('paths') if isinstance(result, dict) else None
    if not isinstance(paths, list):
        errors.append('missing_paths')
        return report
    seeds = [p.get('seed') if isinstance(p, dict) else None for p in paths]
    if any(type(s) is not int for s in seeds) or sorted(seeds) != sorted(c['expected_seeds']):
        errors.append('requested_seed_coverage_mismatch_or_duplicate')
    for i, path in enumerate(paths):
        if not isinstance(path, dict):
            errors.append(f'path_{i}:invalid_schema')
            continue
        rows = path.get('trajectory')
        if path.get('valid') is not True:
            errors.append(f"seed_{path.get('seed')}:path_invalid:{path.get('exception', 'no_exception_recorded')}")
        if (not isinstance(rows, list) or len(rows) != c['horizon'] or
                any(not isinstance(row, dict) or type(row.get('period')) is not int for row in rows) or
                [row['period'] for row in rows] != list(range(1, c['horizon']+1))):
            errors.append(f"seed_{path.get('seed')}:incomplete_or_misordered_trajectory")
            continue
        for row in rows:
            if any(not finite(row.get(axis)) for axis in AXES):
                errors.append(f"seed_{path.get('seed')}:missing_or_nonfinite_macro:t{row['period']}")
            if row.get('termination_reason') or row.get('accounting_failure'):
                errors.append(f"seed_{path.get('seed')}:reported_failure:t{row['period']}")
    if errors:
        return report
    paths = sorted(paths, key=lambda path: path['seed'])
    for axis in AXES:
        threshold, target = c['thresholds'][axis], c['targets'][axis]
        if target is None:
            report['axes'][axis] = {'status': 'NOT_EVALUATED', 'reason': 'missing_comparable_targets',
                                    'threshold': threshold}
            continue
        mean_values = {str(t): statistics.mean(path['trajectory'][t-1][axis] for path in paths) for t in c['periods']}
        mean_errors = {t: mean_values[t]-target[t] for t in target}
        mean_mae = statistics.mean(abs(e) for e in mean_errors.values())
        per_seed = {str(path['seed']): statistics.mean(abs(path['trajectory'][t-1][axis]-target[str(t)])
                    for t in c['periods']) for path in paths}
        values = list(per_seed.values())
        report['axes'][axis] = {
            'status': 'PASS' if mean_mae <= threshold else 'FAIL', 'threshold': threshold,
            'mean_path_mae': mean_mae, 'mean_of_seed_maes': statistics.mean(values),
            'seed_mae_sd': statistics.stdev(values) if len(values) > 1 else None,
            'seed_mae_min': min(values), 'seed_mae_max': max(values), 'per_seed_mae': per_seed,
            'seed_mae_pass_fraction': sum(v <= threshold for v in values)/len(values),
            'all_seeds_within_mae_threshold': all(v <= threshold for v in values),
            'mean_path': mean_values, 'signed_period_errors': mean_errors,
            'pointwise_within_same_threshold_diagnostic_only': all(abs(e) <= threshold for e in mean_errors.values())}
    states = [r['status'] for r in report['axes'].values()]
    report['macro_mae_status'] = 'FAIL' if 'FAIL' in states else 'NOT_EVALUATED' if 'NOT_EVALUATED' in states else 'PASS'
    report['all_axes_per_seed_pass_fraction'] = (sum(all(report['axes'][a]['per_seed_mae'][str(p['seed'])] <=
        c['thresholds'][a] for a in AXES) for p in paths)/len(paths) if 'NOT_EVALUATED' not in states else None)
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--result', required=True, help='backend response containing raw paths, not search aggregate scores')
    p.add_argument('--contract', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    result = json.loads(Path(args.result).read_text(encoding='utf-8'))
    contract = json.loads(Path(args.contract).read_text(encoding='utf-8'))
    with Path(args.output).open('x', encoding='utf-8') as f:
        json.dump(assess_mae(result, contract), f, ensure_ascii=False, indent=2, allow_nan=False)
