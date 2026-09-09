"""Additive AgentEWM taskbook adapter; original economics remain unchanged."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from taskbook_inputs import FOUNDATION, check_foundation, load_foundation, preparation_status

HERE = Path(__file__).resolve().parent
PARAMETERS = {'capital_adjustment_speed': 'bank', 'price_adjustment_speed': 'env_core',
              'inflation_expectation_lambda': 'env_core'}
METRICS = ('inflation', 'real_gdp_growth', 'consumption_gdp', 'investment_gdp')


def write_json(path, data):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hashes(repo):
    names = subprocess.check_output(['git', '-c', f'safe.directory={repo.as_posix()}',
        '-C', str(repo), 'ls-files', '-z'], text=True).split('\0')
    return {name: sha(repo/name) for name in names if name}


def check_parameters(parameters):
    if not isinstance(parameters, dict):
        raise ValueError('parameters must be a dictionary')
    if set(parameters) - PARAMETERS.keys():
        raise ValueError('Only the three taskbook transmission parameters are permitted')
    for key, value in parameters.items():
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f'{key}: development adapter domain is [0,1], not an estimated prior')
        if key == 'capital_adjustment_speed' and value == 0:
            raise ValueError('upstream capital_adjustment_speed must be strictly positive')


def check_run(n, horizon, seeds):
    if type(n) is not int or not 16 <= n <= 10000:
        raise ValueError('integer N in [16,10000] required')
    if type(horizon) is not int or not 2 <= horizon <= 27:
        raise ValueError('integer horizon in [2,27] required')
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError('nonempty unique seeds required')
    if any(type(s) is not int or not 0 <= s < 2**32 for s in seeds):
        raise ValueError('uint32 seeds required')


def controlled_config(parameters, n, horizon, seed, foundation=None):
    from utils.config import load_config
    check_parameters(parameters)
    check_run(n, horizon, [seed])
    foundation = check_foundation(foundation)
    cfg = load_config('real_society')
    entities = {e.entity_name: e.entity_args.params for e in cfg.Environment.Entities}
    cfg.Environment.env_core.episode_length = horizon
    entities['households'].households_n = n
    entities['households'].freeze_labor_efficiency = True
    entities['market'].sigma_z = 0.0
    for key, value in dict(house_alg='rule_based', household_rule='fixed_consumption',
                           household_consumption_share=.95, household_labor_rule='initial',
                           seed=seed, cuda=False, test=True, epoch_length=horizon).items():
        cfg.Trainer[key] = value
    for key, value in parameters.items():
        target = cfg.Environment.env_core if PARAMETERS[key] == 'env_core' else entities[PARAMETERS[key]]
        target[key] = value
    if foundation is not None:
        for key, value in foundation['parameters'].items():
            entities[FOUNDATION[key]][key] = value
    if (entities['households'].type != 'ramsey' or entities['market'].type != 'perfect'
            or not entities['bank'].firm_accounting):
        raise ValueError('Only Ramsey perfect-market shared-ledger configuration is supported')
    return cfg


def trajectory_moments(paths, horizon):
    return {f'{metric}:t{t}': sum(p['trajectory'][t-1][metric] for p in paths)/len(paths)
            for metric in METRICS for t in range(1, horizon+1)
            if not (metric == 'inflation' and t == 1)}


def temporal_mae(moments, targets):
    result = {}
    for metric in METRICS:
        keys = [k for k in targets if k.startswith(metric+':t')]
        if keys:
            result[metric] = sum(abs(moments[k]-targets[k]) for k in keys)/len(keys)
    return result


def run_path(cfg, seed):
    import numpy as np
    import torch
    from env.env_core import EconomicSociety
    from agents.rule_based.rules_core import rule_agent
    from utils.seeds import set_seeds
    set_seeds(seed, cuda=False)
    torch.set_num_threads(1)
    env = EconomicSociety(cfg.Environment)
    if env.market.firm_n != 1 or set(env.government) != {'tax'}:
        raise ValueError('Only single-good fiscal tax route supported')
    env.set_tax_type('us_federal')
    policies = {name: rule_agent(env, cfg.Trainer, agent_name=name, type=entity.type)
        for name, entity in [('households', env.households), ('market', env.market), ('bank', env.bank)]}
    policies['government'] = {'tax': rule_agent(env, cfg.Trainer, agent_name='government', type='tax')}
    obs = env.reset(seed=seed)
    previous, rows, valid = float(env.main_gov.GDP), [], True
    for t in range(1, int(cfg.Environment.env_core.episode_length)+1):
        # The next step promotes Kt_next to Kt before production. Pre-step Kt is stale.
        old_k = env.market.Kt_next.copy()
        expected_used = float(env.expected_inflation)
        from sector_accounting import opening_snapshot, collect_sectors
        sector_opening = opening_snapshot(env)
        def action(policy, values):
            if isinstance(policy, dict):
                return {k: action(v, values[k]) for k, v in policy.items()}
            return policy.get_action(torch.as_tensor(values, dtype=torch.float32))
        actions = {name: action(policy, obs[name]) for name, policy in policies.items()}
        obs, rewards, done = env.step(actions, t-1)
        m, b, h, g = env.market, env.bank, env.households, env.main_gov
        nominal, real = float(g.nominal_GDP), float(g.GDP)
        if nominal <= 0 or previous <= 0:
            raise ValueError('nonpositive scoring denominator')
        # market.price has already advanced. consumer_prices records actual transaction prices.
        price = env.consumer_prices/(1+env.consumption_tax_rate)
        c = float(np.sum(h.final_consumption * env.consumer_prices.reshape(1, -1)))
        investment = float(np.sum(b.actual_fixed_investment * price))
        unused = float(np.sum(m.goods_supply-env.real_deals))
        capital_residual = float(np.max(np.abs(m.Kt_next-(1-b.depreciation_rate)*old_k-b.actual_fixed_investment)))
        bank_scale = sum(abs(float(x)) for x in (b.current_account, b.capital_loan, b.household_loans,
            b.government_bonds, b.total_deposits, b.firm_deposits, b.equity))
        bank_pass = abs(float(b.balance_sheet_residual)) <= 1e-5+1e-10*bank_scale
        row = {'period': t, 'inflation': float(env.inflation_rate), 'real_gdp_growth': real/previous-1,
            'consumption_gdp': c/nominal, 'investment_gdp': investment/nominal,
            'real_gdp': real, 'nominal_gdp': nominal, 'transaction_price': float(price.item()),
            'unused_real_output': unused, 'bank_balance_sheet_residual': float(b.balance_sheet_residual),
            'bank_balance_sheet_scale': bank_scale, 'bank_check': bank_pass, 'capital_residual': capital_residual,
            'firm_cash_residual': float(np.max(np.abs(m.cash_residual))),
            'firm_loan_residual': float(np.max(np.abs(m.loan_residual))),
            'expected_inflation_used': expected_used, 'expected_inflation_next': float(env.expected_inflation),
            'target_at_capacity': bool(np.any(b.target_at_capacity)),
            'goods_constraint_binding': bool(np.any(b.actual_fixed_investment < b.financed_fixed_investment-1e-8)),
            'real_lending_rate': float(b.real_lending_rate),
            'termination_reason': env.termination_reason, 'accounting_failure': b.accounting_failure}
        # Observation only: a diagnostic issue cannot change the simulator's actions or validity gate.
        from investment_diagnostics import collect_investment
        try:
            row['investment_diagnostic'] = collect_investment(env, expected_used, price)
        except (AttributeError, ValueError, TypeError, OverflowError, ZeroDivisionError) as error:
            row['investment_diagnostic'] = {'status': 'UNAVAILABLE', 'error': repr(error)}
        try:
            row['sector_accounting'] = collect_sectors(env, sector_opening, price)
        except (AttributeError, ValueError, TypeError, OverflowError, ZeroDivisionError) as error:
            row['sector_accounting'] = {'status': 'UNAVAILABLE', 'error': repr(error)}
        if any(not math.isfinite(v) for v in row.values() if isinstance(v, float)):
            raise ValueError('nonfinite trajectory')
        rows.append(row)
        valid &= (bank_pass and capital_residual <= 1e-5+1e-10*float(np.sum(abs(old_k)))
                  and unused >= -(1e-5+1e-10*abs(real)) and not b.accounting_failure
                  and not env.termination_reason)
        previous = real
        if done:
            valid &= t == int(cfg.Environment.env_core.episode_length)
            break
    return {'seed': seed, 'valid': bool(valid and len(rows) == int(cfg.Environment.env_core.episode_length)),
        'trajectory': rows, 'full_taskbook_accounting': 'not_verified',
        'micro_validation': 'blocked_on_comparable_weighted_definition'}


def worker(request, output):
    from omegaconf import OmegaConf
    if (not isinstance(request, dict) or set(request) - {'parameters', 'seeds', 'n', 'horizon', 'foundation'}
            or not {'parameters', 'seeds', 'n', 'horizon'} <= set(request)):
        raise ValueError('invalid worker request schema')
    check_parameters(request['parameters'])
    check_run(request['n'], request['horizon'], request['seeds'])
    check_foundation(request.get('foundation'))
    paths = []
    for seed in request['seeds']:
        cfg = controlled_config(request['parameters'], request['n'], request['horizon'], seed,
                                request.get('foundation'))
        with (output/f'config_seed_{seed}.yaml').open('x', encoding='utf-8') as f:
            f.write(OmegaConf.to_yaml(cfg, resolve=True))
        try:
            path = run_path(cfg, seed)
        except Exception as error:
            path = {'seed': seed, 'valid': False, 'exception': repr(error), 'trajectory': []}
        paths.append(path)
        write_json(output/f'path_seed_{seed}.json', path)
    valid = all(p['valid'] for p in paths)
    return {'valid': valid, 'moments': trajectory_moments(paths, request['horizon']) if valid else {},
        'paths': paths, 'evidence_level': 'development_integration_only', 'economic_validity_certified': False,
        'taskbook_acceptance': 'blocked_pending_measurement_and_full_checks'}


class AgentEWMBackend:
    def __init__(self, repo, output, n=64, horizon=3, runtime_paths=(), foundation=None):
        check_run(n, horizon, [1])
        self.foundation = check_foundation(foundation)
        self.repo, self.output = Path(repo).resolve(strict=True), Path(output).resolve()
        self.output.mkdir(parents=True, exist_ok=False)
        self.n, self.horizon, self.calls = n, horizon, 0
        self.runtime_paths = [str(Path(p).resolve(strict=True)) for p in runtime_paths]
        self.sources = source_hashes(self.repo)
        self.extension = {p.name: sha(p) for p in HERE.glob('*.py')}
        write_json(self.output/'sources_before.json', self.sources)
        write_json(self.output/'foundation_input.json', self.foundation)

    def describe(self):
        return {'adapter': 'agentewm_taskbook_interfaces_v2', 'repo': str(self.repo), 'N': self.n, 'horizon': self.horizon,
            'source_digest': hashlib.sha256(json.dumps(self.sources, sort_keys=True).encode()).hexdigest(),
            'calibration_code': self.extension, 'policy': 'upstream_fixed_consumption_initial_labor',
            'full_taskbook_acceptance': False, 'base_scene': 'real_society',
            'foundation': self.foundation, 'foundation_empirically_approved': False}

    def evaluate(self, parameters, seeds):
        check_parameters(parameters)
        check_run(self.n, self.horizon, seeds)
        if source_hashes(self.repo) != self.sources or {p.name: sha(p) for p in HERE.glob('*.py')} != self.extension:
            raise RuntimeError('source changed during integration run')
        job = self.output/f'eval_{self.calls:04d}'
        job.mkdir(exist_ok=False)
        self.calls += 1
        write_json(job/'request.json', {'parameters': parameters, 'seeds': list(seeds), 'n': self.n,
                                      'horizon': self.horizon, 'foundation': self.foundation})
        command = [sys.executable, '-B', str(HERE/'launch.py'), '--repo', str(self.repo)]
        for path in self.runtime_paths:
            command.extend(['--runtime-path', path])
        command.extend(['worker', '--request', str(job/'request.json'), '--output', str(job)])
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
        p = subprocess.run(command, cwd=self.repo, env=env, capture_output=True, text=True,
                           encoding='utf-8', errors='replace', timeout=300)
        write_json(job/'process.json', {'returncode': p.returncode, 'stdout': p.stdout, 'stderr': p.stderr})
        if (source_hashes(self.repo) != self.sources
                or {p.name: sha(p) for p in HERE.glob('*.py')} != self.extension):
            raise RuntimeError('upstream or adapter bytes changed')
        if p.returncode:
            raise RuntimeError(f'worker failed; retained {job/"process.json"}')
        return json.loads((job/'response.json').read_text(encoding='utf-8'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', required=True)
    parser.add_argument('--runtime-path', action='append', default=[])
    parser.add_argument('mode', choices=['prepare', 'evaluate', 'worker', 'synthetic-calibrate'])
    parser.add_argument('--output', required=True)
    parser.add_argument('--request')
    parser.add_argument('--foundation-card', help='explicit external development candidates; not optimizer inputs')
    parser.add_argument('--data-bundle', help='prepare only: verify and attach data readiness, without fitting')
    parser.add_argument('--n', type=int, default=64)
    parser.add_argument('--horizon', type=int, default=3)
    parser.add_argument('--seeds', type=int, nargs='+', default=[908101])
    args = parser.parse_args()
    repo, out = Path(args.repo).resolve(strict=True), Path(args.output).resolve()
    foundation = load_foundation(args.foundation_card)
    req = json.loads(Path(args.request).read_text(encoding='utf-8')) if args.request else {}
    # Resolve caller-relative inputs before the upstream config loader requires repo cwd.
    data_bundle = Path(args.data_bundle).resolve(strict=True) if args.data_bundle else None
    if data_bundle is not None and args.mode != 'prepare':
        parser.error('--data-bundle is prepare-only; no real-data fitting entry point is enabled')
    if args.mode == 'worker' and args.foundation_card:
        parser.error('worker foundation must be in the preserved request, not a CLI override')
    if args.mode == 'synthetic-calibrate' and args.request:
        parser.error('synthetic-calibrate does not accept external targets or requests')
    os.chdir(repo)
    if args.mode == 'worker':
        if not args.request:
            parser.error('worker requires --request')
        write_json(out/'response.json', worker(req, out))
        return
    if not isinstance(req, dict) or set(req)-{'parameters'}:
        raise ValueError('request only accepts parameters')
    check_parameters(req.get('parameters', {}))
    check_run(args.n, args.horizon, args.seeds)
    if args.mode == 'prepare':
        from omegaconf import OmegaConf
        cfg = controlled_config(req.get('parameters', {}), args.n, args.horizon, args.seeds[0], foundation)
        status = preparation_status(foundation, data_bundle)
        status.update(sources=source_hashes(repo),
                      calibration_code={p.name: sha(p) for p in HERE.glob('*.py')},
                      requested_seeds=args.seeds, config_seed=args.seeds[0],
                      configuration_only=True, optimizer_parameters=req.get('parameters', {}))
        out.mkdir(parents=True, exist_ok=False)
        with (out/'controlled_config.yaml').open('x', encoding='utf-8') as f:
            f.write(OmegaConf.to_yaml(cfg, resolve=True))
        write_json(out/'readiness.json', status)
        return
    backend = AgentEWMBackend(repo, out, args.n, args.horizon, args.runtime_path, foundation)
    if args.mode == 'evaluate':
        result = backend.evaluate(req.get('parameters', {}), tuple(args.seeds))
    else:
        from core import Parameter, Target, Budget, calibrate
        if args.horizon < 3:
            raise ValueError('synthetic fixture needs three periods')
        reference = backend.evaluate({}, (908201,))
        if not reference['valid']:
            raise RuntimeError('synthetic target fixture failed')
        targets = {k: Target(v, 1., 1e-8) for k, v in reference['moments'].items()}
        params = (Parameter('capital_adjustment_speed', .05, .4, .2, 'investment_gdp:t2'),
                  Parameter('price_adjustment_speed', .05, .4, .2, 'inflation:t2'),
                  Parameter('inflation_expectation_lambda', .1, .9, .5, 'inflation:t3'))
        result = calibrate(backend, params, targets, Budget(2, (908201,), (908202,)), method='random')
        validation_result = result['validation']['result'] if result.get('validation') else None
        if validation_result and validation_result['valid']:
            fixture_values = {k: v.value for k, v in targets.items()}
            result['synthetic_validation_mean_path_mae'] = temporal_mae(validation_result['moments'], fixture_values)
            result['synthetic_validation_per_seed_mae'] = {
                str(path['seed']): temporal_mae(trajectory_moments([path], args.horizon), fixture_values)
                for path in validation_result['paths']}
        result.update(target_provenance='synthetic_self_target_search_seed_overlap_disclosed',
                      taskbook_acceptance=False, real_data_fit=False)
    write_json(out/'result.json', result)
    write_json(out/'verification.json', {'tracked_sources_unchanged': source_hashes(repo) == backend.sources,
                                        'backend': backend.describe()})
    print(json.dumps({'output': str(out), 'valid': result.get('valid'), 'status': result.get('status')}))
