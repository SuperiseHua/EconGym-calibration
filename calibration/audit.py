"""Read-only upstream/local alignment and isolated validation, never merges engines."""
import argparse
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import unittest
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bridge import source_hashes, sha, write_json, AgentEWMBackend


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--repo', required=True)
    p.add_argument('--runtime-path', action='append', default=[])
    p.add_argument('--workspace', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    repo, workspace, out = Path(args.repo), Path(args.workspace), Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    os.chdir(repo)
    before = source_hashes(repo)
    import subprocess
    changes = subprocess.check_output(['git', '-c', f'safe.directory={repo.as_posix()}', '-C', str(repo),
        'diff', '--name-status', 'f0ccbb8', 'HEAD'], text=True).splitlines()
    mapping = {}
    for name in ('EconGym', 'EconGym-paper-accounting-fixed-30a1f1f'):
        root = workspace/name
        mapping[name] = {path: ('missing_locally' if not (root/path).is_file() else
                               'identical' if sha(root/path) == digest else 'different')
                         for path, digest in before.items()
                         if Path(path).suffix in ('.py', '.yaml', '.yml', '.md')}
    write_json(out/'alignment.json', {'remote_commit': subprocess.check_output(
        ['git', '-c', f'safe.directory={repo.as_posix()}', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip(),
        'changes_vs_f0ccbb8': changes, 'local_relative_to_AgentEWM': mapping,
        'new_core_identical_to_existing_extension': sha(repo/'calibration/core.py') == sha(
            workspace/'extensions/econgym-calibration/src/econgym_calibration/core.py'),
        'taskbook_sha256': sha(repo/'calibration/TASKBOOK_RECEIVED_20260908.md')})
    checks = {}
    for label, start, pattern in [('adapter', str(repo/'calibration'), 'test_bridge.py'),
                                  ('upstream', str(repo/'tests'), 'test*.py')]:
        log = io.StringIO()
        suite = unittest.TestLoader().discover(start, pattern=pattern, top_level_dir=start)
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
        with (out/f'{label}_tests.txt').open('x', encoding='utf-8') as f:
            f.write(log.getvalue())
        checks[label] = {'run': result.testsRun, 'failures': len(result.failures),
                         'errors': len(result.errors), 'skipped': len(result.skipped), 'success': result.wasSuccessful()}
    backend = AgentEWMBackend(repo, out/'paired', n=64, horizon=4, runtime_paths=args.runtime_path)
    first = backend.evaluate({}, (908301,))
    second = backend.evaluate({}, (908301,))
    checks['same_seed_repeat_exact'] = first == second
    checks['base_valid'] = first['valid'] and second['valid']
    checks['parameter_responses'] = {}
    for key, value in [('capital_adjustment_speed', .1), ('price_adjustment_speed', .3),
                       ('inflation_expectation_lambda', .7)]:
        response = backend.evaluate({key: value}, (908301,))
        changes = {k: v-first['moments'][k] for k, v in response['moments'].items()} if first['valid'] else {}
        checks['parameter_responses'][key] = {'valid': response['valid'], 'moment_deltas': changes,
                                             'any_numerical_response': any(abs(x)>1e-12 for x in changes.values())}
    checks['tracked_sources_unchanged'] = before == source_hashes(repo)
    write_json(out/'checks.json', checks)
    print(json.dumps(checks, ensure_ascii=False))


if __name__ == '__main__':
    # Explicit runtime flags work in the embedded interpreter as well as normal Python.
    for i, token in enumerate(sys.argv):
        if token == '--runtime-path': sys.path.append(sys.argv[i+1])
    sys.path.insert(0, sys.argv[sys.argv.index('--repo')+1])
    main()
