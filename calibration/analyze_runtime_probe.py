"""Read retained development outputs only; no simulator, optimizer, or scoring rewrite."""
import argparse
import json
import math
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root = Path(args.run)
    summary, protocol = read(root/'summary.json'), read(root/'protocol.json')
    baseline = summary['synthetic_recovery']['uncalibrated']['normalized_rmse']
    analysis = {'evidence_level': 'post_hoc_diagnosis_of_opened_development',
                'no_threshold_or_candidate_changes': True, 'methods': {}}
    for method in ('blockwise', 'random'):
        result = read(root/f'{method}_result.json')
        validation = result['validation']['score']
        failed = {k: {'error': error, 'tolerance': result['targets'][k]['tolerance']}
                  for k, error in validation['errors'].items()
                  if abs(error) > result['targets'][k]['tolerance']}
        analysis['methods'][method] = {
            'validation_loss_reduction_fraction': 1-validation['loss']/baseline,
            'failed_validation_moments': failed,
            'training_passed': result['search_score']['passed'],
            'validation_passed': validation['passed'],
            'actual_search_configurations': result['search_configurations'],
            'actual_search_paths': result['search_seed_paths_requested'],
            'validation_paths': result['validation_seed_paths_requested']}
        if method == 'blockwise':
            first_pair = result['search_trace'][1:3]
            analysis['blockwise_first_capital_bracket'] = [
                {'parameters': row['parameters'], 'investment_t2_error': row['score']['errors']['investment_gdp:t2'],
                 'investment_t2_simulated': row['result']['moments']['investment_gdp:t2']}
                for row in first_pair]
    scales = protocol['recovery']['scale_and_pointwise_tolerance']
    keys = [k for k in summary['parameter_response']['capital_adjustment_speed']['mean_path_deltas']
            if not k.endswith(':t1')]
    directions = {}
    for key, row in summary['parameter_response'].items():
        delta_parameter = protocol['response_probes'][key] - (.5 if key == 'inflation_expectation_lambda' else .2)
        directions[key] = [row['mean_path_deltas'][k]/scales[k.split(':')[0]]/delta_parameter for k in keys]
    a, b = directions['capital_adjustment_speed'], directions['inflation_expectation_lambda']
    analysis['capital_expectation_direction_cosine'] = sum(x*y for x,y in zip(a,b))/math.sqrt(
        sum(x*x for x in a)*sum(y*y for y in b))
    analysis['direction_note'] = 'Finite one-sided finite-difference probes of different sizes; suggestive collinearity, not a local-rank or global-identification proof.'
    paths = read(root/'uncalibrated_validation/eval_0000/response.json')['paths']
    rows = [r for p in paths for r in p['trajectory'][1:]]
    analysis['baseline_post_initial_constraints'] = {
        'rows': len(rows), 'target_at_capacity': sum(r['target_at_capacity'] for r in rows),
        'goods_constraint_binding': sum(r['goods_constraint_binding'] for r in rows),
        'max_abs_C_plus_I_minus_0829': max(abs(r['consumption_gdp']+r['investment_gdp']-.829) for r in rows)}
    for method in ('blockwise', 'random'):
        rows_actual = read(root/f'{method}_result.json')['validation']['result']['paths']
        paired = []
        targets = read(root/'synthetic_targets.json')
        for base, selected in zip(paths, rows_actual):
            assert base['seed'] == selected['seed']
            def loss(path):
                errors = []
                for row in path['trajectory'][1:]:
                    for metric in scales:
                        errors.append((row[metric]-targets[f"{metric}:t{row['period']}"])/scales[metric])
                return math.sqrt(sum(e*e for e in errors)/len(errors))
            before, after = loss(base), loss(selected)
            paired.append({'seed': base['seed'], 'before': before, 'after': after, 'improved': after < before})
        analysis['methods'][method]['paired_seed_losses'] = paired
    with Path(args.output).open('x', encoding='utf-8') as f:
        json.dump(analysis, f, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(analysis, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
