"""Non-simulator integration checks. This is NOT audit.py (which runs simulations)."""
import argparse
import ast
import io
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--repo', required=True)
    p.add_argument('--runtime-path', action='append', default=[])
    p.add_argument('--output', required=True)
    args = p.parse_args()
    repo, out = Path(args.repo).resolve(strict=True), Path(args.output).resolve()
    sys.dont_write_bytecode = True
    sys.path[:0] = [str(repo/'calibration'), str(repo)]
    sys.path.extend(args.runtime_path)
    os.chdir(repo)
    import bridge
    out.mkdir(parents=True, exist_ok=False)
    before = bridge.source_hashes(repo)
    code_before = {f.name: bridge.sha(f) for f in (repo/'calibration').glob('*.py')}
    for name in code_before:
        ast.parse((repo/'calibration'/name).read_text(encoding='utf-8'), filename=name)
    suite = unittest.TestSuite()
    for module in ('test_bridge', 'test_taskbook_inputs', 'test_mae_investment', 'test_sector_accounting'):
        suite.addTests(unittest.defaultTestLoader.loadTestsFromName(module))
    log = io.StringIO()
    result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    with (out/'unit_tests.txt').open('x', encoding='utf-8') as f:
        f.write(log.getvalue())
    prepare_args = ['launch.py', '--repo', str(repo), 'prepare', '--output', str(out/'prepared'),
                    '--data-bundle', str(repo/'calibration/data/collected_v1'),
                    '--n', '1000', '--horizon', '27', '--seeds', '908701']
    with patch('sys.argv', prepare_args), \
            patch.object(bridge, 'run_path', side_effect=AssertionError('no simulator in static checks')), \
            patch.object(bridge, 'AgentEWMBackend', side_effect=AssertionError('no simulation backend')):
        bridge.main()
    readiness = json.loads((out/'prepared/readiness.json').read_text(encoding='utf-8'))
    source_unchanged = before == bridge.source_hashes(repo)
    original_core = repo.parents[1]/'extensions/econgym-calibration/src/econgym_calibration/core.py'
    checks = {'schema_version': 1, 'scope': 'configuration_data_and_unit_interfaces_only',
              'tests': {'run': result.testsRun, 'errors': len(result.errors),
                        'failures': len(result.failures), 'skipped': len(result.skipped),
                        'passed': result.wasSuccessful()},
              'tracked_file_count': len(before), 'tracked_sources_unchanged': source_unchanged,
              'adapter_sources_unchanged_during_check': code_before == {
                  f.name: bridge.sha(f) for f in (repo/'calibration').glob('*.py')},
              'calibration_code_sha256': code_before,
              'core_identical_to_preserved_extension': original_core.is_file() and
                  bridge.sha(original_core) == bridge.sha(repo/'calibration/core.py'),
              'collected_data_files_verified': readiness['data']['source_files_verified'],
              'preparation_without_auto_applying_data': readiness['foundation_card'] is None and
                  readiness['data']['auto_apply_parameters'] is False,
              'simulator_module_imported': 'env.env_core' in sys.modules,
              'physical_simulator_paths': 0, 'optimization_runs': 0,
              'real_calibration_ready': False, 'taskbook_acceptance': 'NOT_EVALUATED'}
    checks['interface_checks_passed'] = (result.wasSuccessful() and source_unchanged and
        checks['adapter_sources_unchanged_during_check'] and checks['core_identical_to_preserved_extension'] and
        checks['preparation_without_auto_applying_data'] and not checks['simulator_module_imported'])
    bridge.write_json(out/'checks.json', checks)
    bridge.write_json(out/'sources.json', before)
    print(json.dumps(checks, ensure_ascii=False, indent=2))
    return 0 if checks['interface_checks_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
