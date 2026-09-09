"""Development logging invariance, paired investment probes, independent MAE reports."""
import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--repo', required=True)
    p.add_argument('--runtime-path', action='append', default=[])
    p.add_argument('--output', required=True)
    args = p.parse_args()
    repo, out = Path(args.repo).resolve(strict=True), Path(args.output).resolve()
    sys.path[:0] = [str(repo/'calibration'), str(repo)]
    sys.path.extend(args.runtime_path)
    os.chdir(repo)
    from bridge import AgentEWMBackend, source_hashes, sha, write_json
    from mae_acceptance import assess_mae, AXES
    def read(path): return json.loads(path.read_text(encoding='utf-8'))
    old = repo/'calibration/runs/runtime_effect_v1'
    old_hashes = {str(f.relative_to(old)): sha(f) for f in old.rglob('*') if f.is_file()}
    before = source_hashes(repo)
    out.mkdir(parents=True, exist_ok=False)
    (out/'code_snapshot').mkdir()
    for file in (repo/'calibration').glob('*.py'):
        shutil.copy2(file, out/'code_snapshot'/file.name)
    targets = read(old/'synthetic_targets.json')
    protocol = {'evidence_level': 'development_diagnostics_no_optimization',
                'paired_seeds': [908901, 908902, 908903], 'N': 256, 'horizon': 4,
                'cases': {'baseline': {}, 'capital': {'capital_adjustment_speed': .1},
                          'price': {'price_adjustment_speed': .3}, 'expectation': {'inflation_expectation_lambda': .7}},
                'logging_invariance': {'opened_development_seed': 908841, 'N': 256, 'horizon': 4},
                'long_smoke': {'seed': 908911, 'N': 1000, 'horizon': 27},
                'old_mae_reports': 'post_hoc_development_new_metric_layer_does_not_replace_old_decisions',
                'frozen_old_results': old_hashes, 'upstream_sources': before,
                'optimization_runs': 0, 'no_threshold_changes': True}
    write_json(out/'protocol.json', protocol)
    def contract(seeds, evidence):
        return {'schema_version': 1, 'target_kind': 'synthetic', 'evidence_level': evidence,
                'horizon': 4, 'expected_seeds': list(seeds), 'periods': [2, 3, 4], 'units': 'fraction',
                'thresholds': {'inflation': .01, 'real_gdp_growth': .02, 'consumption_gdp': .05, 'investment_gdp': .03},
                'targets': {a: {str(t): targets[f'{a}:t{t}'] for t in (2, 3, 4)} for a in AXES}}
    old_contract = contract((908841, 908842, 908843), 'post_hoc_development')
    new_contract = contract(protocol['paired_seeds'], 'development_predeclared')
    write_json(out/'old_mae_contract.json', old_contract)
    write_json(out/'new_mae_contract.json', new_contract)
    summary = {'evidence_level': protocol['evidence_level'], 'old_mae_diagnostic_reports': {}, 'cases': {}}
    for label, result in (
        ('uncalibrated', read(old/'uncalibrated_validation/eval_0000/response.json')),
        ('blockwise', read(old/'blockwise_result.json')['validation']['result']),
        ('random', read(old/'random_result.json')['validation']['result']),
        ('known_parameter', read(old/'known_parameter_noise_reference/eval_0000/response.json'))):
        assessment = assess_mae(result, old_contract)
        write_json(out/f'old_{label}_mae.json', assessment)
        summary['old_mae_diagnostic_reports'][label] = {
            'mean_path_macro_mae_status': assessment['macro_mae_status'],
            'per_seed_all_axes_pass_fraction': assessment['all_axes_per_seed_pass_fraction'],
            'original_search_decision': read(old/f'{label}_result.json')['status'] if label in ('blockwise', 'random') else 'not_a_search',
            'taskbook_acceptance': assessment['taskbook_acceptance']}
    all_results = []
    def run(label, n, horizon, seeds, params):
        b = AgentEWMBackend(repo, out/label, n, horizon, args.runtime_path)
        result = b.evaluate(params, tuple(seeds))
        all_results.append(result)
        print(json.dumps({'case': label, 'valid': result['valid'], 'paths': len(result['paths'])}), flush=True)
        return result
    replay = run('logging_invariance', 256, 4, [908841], {})
    replay_copy = copy.deepcopy(replay['paths'][0])
    for row in replay_copy['trajectory']: row.pop('investment_diagnostic', None)
    old_path = read(old/'uncalibrated_validation/eval_0000/path_seed_908841.json')
    summary['logging_invariance_exact'] = replay_copy == old_path
    if not replay['valid'] or not summary['logging_invariance_exact']:
        write_json(out/'summary.json', {**summary, 'stopped_at': 'logging_invariance'})
        return 2
    cases = {}
    for label, params in protocol['cases'].items():
        result = run(label, 256, 4, protocol['paired_seeds'], params)
        cases[label] = result
        assessment = assess_mae(result, new_contract)
        write_json(out/f'{label}_mae.json', assessment)
        rows = [r for path in result['paths'] for r in path['trajectory']]
        diagnostics = [r['investment_diagnostic'] for r in rows]
        flags = ('target_cap_active', 'investment_zero_floor_active', 'investment_growth_cap_active',
                 'credit_rationing_material', 'goods_rationing_material')
        summary['cases'][label] = {'valid': result['valid'], 'diagnostic_rows': len(rows),
            'diagnostic_status_counts': {state: sum(d['status'] == state for d in diagnostics) for state in ('PASS', 'FAIL', 'UNAVAILABLE')},
            'binding_counts_all_periods': {flag: sum(d.get('binding_flags', {}).get(flag, False) for d in diagnostics) for flag in flags},
            'macro_mae_status_vs_old_synthetic_target': assessment['macro_mae_status']}
    long = run('long_smoke', 1000, 27, [908911], {})
    summary['long_smoke'] = {'valid': long['valid'], 'periods': len(long['paths'][0]['trajectory'])}
    summary['paired_transmission'] = {}
    for label in ('capital', 'price', 'expectation'):
        pairs = []
        if cases['baseline']['valid'] and cases[label]['valid']:
            for base_path, path in zip(cases['baseline']['paths'], cases[label]['paths']):
                assert base_path['seed'] == path['seed']
                for base_row, row in zip(base_path['trajectory'], path['trajectory']):
                    d0, d1 = base_row['investment_diagnostic'], row['investment_diagnostic']
                    if d0['status'] == 'UNAVAILABLE' or d1['status'] == 'UNAVAILABLE': continue
                    fields = ('real_lending_rate', 'capital_user_cost', 'target_capital',
                              'desired_investment', 'financed_investment', 'actual_investment')
                    pairs.append({'seed': path['seed'], 'period': row['period'],
                        'deltas': {k: d1['inputs'][k]-d0['inputs'][k] for k in fields},
                        'baseline_flags': d0['binding_flags'], 'changed_flags': d1['binding_flags']})
        summary['paired_transmission'][label] = pairs
    diagnostic_rows = [r['investment_diagnostic'] for result in all_results for path in result['paths'] for r in path['trajectory']]
    summary.update(simulator_paths=sum(len(r['paths']) for r in all_results),
        valid_paths=sum(p['valid'] for r in all_results for p in r['paths']),
        investment_diagnostic_rows=len(diagnostic_rows),
        investment_diagnostic_all_passed=all(d['status'] == 'PASS' for d in diagnostic_rows),
        diagnostic_failures=[d for d in diagnostic_rows if d['status'] != 'PASS'],
        old_outputs_unchanged=old_hashes == {str(f.relative_to(old)): sha(f) for f in old.rglob('*') if f.is_file()},
        tracked_sources_unchanged=before == source_hashes(repo),
        optimizer_runs=0, taskbook_acceptance='NOT_EVALUATED')
    write_json(out/'summary.json', summary)
    print(json.dumps({k: v for k, v in summary.items() if k not in ('paired_transmission', 'diagnostic_failures')}, indent=2), flush=True)
    return 0 if summary['investment_diagnostic_all_passed'] and all(r['valid'] for r in all_results) else 2


if __name__ == '__main__':
    sys.dont_write_bytecode = True
    raise SystemExit(main())
