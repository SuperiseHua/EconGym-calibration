"""Bounded development runtime/recovery experiment, never formal economic validation."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--runtime-path', action='append', default=[])
    args = parser.parse_args()
    repo, out = Path(args.repo).resolve(strict=True), Path(args.output).resolve()
    sys.path[:0] = [str(repo/'calibration'), str(repo)]
    sys.path.extend(args.runtime_path)
    os.chdir(repo)
    from bridge import AgentEWMBackend, source_hashes, sha, write_json, temporal_mae, trajectory_moments
    from core import Parameter, Target, Budget, calibrate, score
    from data_access import CollectedData
    out.mkdir(parents=True, exist_ok=False)
    snapshot = out/'code_snapshot'
    snapshot.mkdir()
    for file in (repo/'calibration').glob('*.py'):
        shutil.copy2(file, snapshot/file.name)
    before = source_hashes(repo)
    write_json(out/'sources_before.json', before)
    protocol = {
        'evidence_level': 'opened_development_software_and_synthetic_recovery',
        'real_data_fit': False, 'economic_validity_certified': False,
        'small_smoke': {'N': 64, 'horizon': 4, 'seed': 908801, 'repeat_same_seed': True},
        'long_smoke': {'N': 1000, 'horizon': 27, 'seeds': [908811, 908812]},
        'foundation_smoke': {'N': 64, 'horizon': 4, 'seed': 908813,
                             'values': {'alpha': .36, 'depreciation_rate': .06, 'initial_capital_output_ratio': 2.5},
                             'provenance': 'arbitrary software-routing fixture, not an empirical estimate'},
        'recovery': {'N': 256, 'horizon': 4, 'target_seeds': [908821, 908822, 908823],
                     'search_seeds': [908831, 908832], 'validation_seeds': [908841, 908842, 908843],
                     'target_parameters': {'capital_adjustment_speed': .12, 'price_adjustment_speed': .30,
                                           'inflation_expectation_lambda': .65},
                     'methods': ['blockwise', 'random'], 'max_configurations_each': 12,
                     'optimizer_seed': 908851,
                     'search_bounds': {'capital_adjustment_speed': [.05, .4], 'price_adjustment_speed': [.05, .4],
                                       'inflation_expectation_lambda': [.1, .9]},
                     'scale_and_pointwise_tolerance': {'inflation': .01, 'real_gdp_growth': .02,
                                                       'consumption_gdp': .05, 'investment_gdp': .03},
                     'scored_periods': [2, 3, 4],
                     'acceptance_meaning': 'synthetic numerical contract only, not taskbook real-world acceptance'},
        'response_probes': {'capital_adjustment_speed': .1, 'price_adjustment_speed': .3,
                            'inflation_expectation_lambda': .7},
        'real_data_comparison': {'metrics': ['real_gdp_growth', 'consumption_gdp'],
            'period_year_candidate': {'2': 2023, '3': 2024, '4': 2025},
            'policy': 'fixed 2022, not actual historical policy; descriptive mismatch only',
            'never_used_to_optimize': True},
        'no_post_validation_tuning': True, 'upstream_modifications_allowed': False}
    write_json(out/'protocol.json', protocol)
    for a, b in [('target_seeds', 'search_seeds'), ('target_seeds', 'validation_seeds'),
                 ('search_seeds', 'validation_seeds')]:
        assert not set(protocol['recovery'][a]) & set(protocol['recovery'][b])
    totals = {'calls': 0, 'requested_paths': 0, 'returned_paths': 0, 'valid_paths': 0}

    class LoggedBackend(AgentEWMBackend):
        def evaluate(self, parameters, seeds):
            start = time.monotonic()
            totals['calls'] += 1
            totals['requested_paths'] += len(seeds)
            result = super().evaluate(parameters, seeds)
            totals['returned_paths'] += len(result['paths'])
            totals['valid_paths'] += sum(p['valid'] for p in result['paths'])
            print(json.dumps({'stage': self.output.name, 'call': self.calls, 'seeds': list(seeds),
                              'valid': result['valid'], 'seconds': round(time.monotonic()-start, 2)}), flush=True)
            return result

    def backend(name, n, horizon, foundation=None):
        return LoggedBackend(repo, out/name, n, horizon, args.runtime_path, foundation)

    summary = {'protocol_sha256': sha(out/'protocol.json'), 'evidence_level': protocol['evidence_level']}
    small = backend('small_smoke', 64, 4)
    first, repeat = small.evaluate({}, (908801,)), small.evaluate({}, (908801,))
    summary['small_smoke'] = {'valid': first['valid'] and repeat['valid'], 'same_seed_exact': first == repeat}
    if not summary['small_smoke']['valid'] or not summary['small_smoke']['same_seed_exact']:
        summary.update(stopped_at='small_smoke', totals=totals)
        write_json(out/'summary.json', summary)
        return 2
    large = backend('long_smoke', 1000, 27).evaluate({}, (908811, 908812))
    summary['long_smoke'] = {'valid': large['valid'], 'paths': [
        {'seed': p['seed'], 'valid': p['valid'], 'periods': len(p['trajectory']),
         'exception': p.get('exception'), 'last': p['trajectory'][-1] if p['trajectory'] else None}
        for p in large['paths']], 'not_formal_30_seed_stability': True}
    from taskbook_inputs import UNITS
    card = {'schema_version': 1, 'status': 'development_candidate',
            'parameters': protocol['foundation_smoke']['values'],
            'provenance': {k: {'source': 'synthetic runtime fixture; not an empirical estimate',
                               'definition': 'test explicit foundation route only', 'unit': UNITS[k], 'year': 2022}
                           for k in UNITS}}
    foundation = backend('foundation_smoke', 64, 4, card).evaluate({}, (908813,))
    summary['foundation_smoke'] = {'valid': foundation['valid'], 'not_empirical_candidate': True,
        'exceptions': [p.get('exception') for p in foundation['paths']]}
    recovery = protocol['recovery']
    target = backend('target_fixture', 256, 4).evaluate(recovery['target_parameters'], tuple(recovery['target_seeds']))
    if not target['valid']:
        summary.update(stopped_at='target_fixture', totals=totals)
        write_json(out/'summary.json', summary)
        return 2
    target_values = {k: v for k, v in target['moments'].items() if not k.endswith(':t1')}
    scales = recovery['scale_and_pointwise_tolerance']
    targets = {k: Target(v, scales[k.split(':')[0]], scales[k.split(':')[0]]) for k, v in target_values.items()}
    write_json(out/'synthetic_targets.json', target_values)
    params = (Parameter('capital_adjustment_speed', .05, .4, .2, 'investment_gdp:t2'),
              Parameter('price_adjustment_speed', .05, .4, .2, 'inflation:t2'),
              Parameter('inflation_expectation_lambda', .1, .9, .5, 'inflation:t3'))
    val_seeds = tuple(recovery['validation_seeds'])
    baseline = backend('uncalibrated_validation', 256, 4).evaluate({}, val_seeds)

    def metrics(result):
        if not result['valid']:
            return {'valid': False}
        return {'valid': True, 'normalized_rmse': score(result, targets)['loss'],
                'mean_path_mae': temporal_mae(result['moments'], target_values),
                'per_seed_mae': {str(p['seed']): temporal_mae(trajectory_moments([p], 4), target_values)
                                 for p in result['paths']}}

    summary['synthetic_recovery'] = {'uncalibrated': metrics(baseline), 'methods': {}}
    for method in recovery['methods']:
        result = calibrate(backend(method, 256, 4), params, targets,
            Budget(12, tuple(recovery['search_seeds']), val_seeds), method=method, optimizer_seed=908851)
        write_json(out/f'{method}_result.json', result)
        summary['synthetic_recovery']['methods'][method] = {
            'status': result['status'], 'selected': result['selected_before_validation'],
            'search_configurations': result['search_configurations'],
            'search_loss': result['search_score']['loss'] if result['search_score'] else None,
            'diagnostic_reasons': result['diagnostic_reasons'],
            'validation': metrics(result['validation']['result']) if result['validation'] else None}
    oracle = backend('known_parameter_noise_reference', 256, 4).evaluate(recovery['target_parameters'], val_seeds)
    summary['synthetic_recovery']['known_parameter_noise_reference'] = metrics(oracle)
    summary['synthetic_recovery']['note'] = 'Known-parameter response is a Monte Carlo noise reference, not a guaranteed loss lower bound.'
    probes = backend('parameter_response', 256, 4)
    summary['parameter_response'] = {}
    for key, value in protocol['response_probes'].items():
        result = probes.evaluate({key: value}, val_seeds)
        item = {'valid': result['valid']}
        if result['valid'] and baseline['valid']:
            item['max_absolute_moment_change'] = max(abs(v-baseline['moments'][k]) for k, v in result['moments'].items())
            item['mean_path_deltas'] = {k: v-baseline['moments'][k] for k, v in result['moments'].items()}
            item['max_expected_inflation_change'] = max(abs(r['expected_inflation_next']-b['expected_inflation_next'])
                for path, base_path in zip(result['paths'], baseline['paths'])
                for r, b in zip(path['trajectory'], base_path['trajectory']))
            item['post_initial_zero_investment_periods'] = sum(r['investment_gdp'] == 0
                for path in result['paths'] for r in path['trajectory'][1:])
        summary['parameter_response'][key] = item
    data = CollectedData(repo/'calibration/data/collected_v1')
    observed = data.observations([2023, 2024, 2025])
    if baseline['valid']:
        summary['real_data_descriptive_only'] = {
            'real_calibration_run': False, 'historical_validity_certified': False,
            'caveats': protocol['real_data_comparison'],
            'uncalibrated_mean_path': [
                {'year_candidate': row['year'], 'period': t,
                 **{metric: {'simulated': baseline['moments'][f'{metric}:t{t}'], 'observed': row[metric],
                              'difference': baseline['moments'][f'{metric}:t{t}']-row[metric]}
                    for metric in ('real_gdp_growth', 'consumption_gdp')}}
                for t, row in enumerate(observed, start=2)]}
    summary.update(totals=totals, tracked_sources_unchanged=before == source_hashes(repo),
        real_calibration_ready=False, taskbook_acceptance='NOT_EVALUATED',
        upstream_modified=False, no_post_validation_tuning=True)
    write_json(out/'summary.json', summary)
    print(json.dumps({'output': str(out), 'totals': totals, 'upstream_unchanged': summary['tracked_sources_unchanged']}), flush=True)
    return 0


if __name__ == '__main__':
    sys.dont_write_bytecode = True
    raise SystemExit(main())
