"""Data-independent publication checks; does not run a simulator or optimizer."""
import argparse
import os
from pathlib import Path
import sys
import unittest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime-path', action='append', default=[],
                        help='Optional explicit dependency directory for embedded Python')
    args = parser.parse_args()
    sys.dont_write_bytecode = True
    cal = Path(__file__).resolve().parent
    repo = cal.parent
    sys.path[:0] = [str(cal), str(repo), str(cal / 'experiments/price_stability_v1')]
    sys.path.extend(str(Path(p).resolve(strict=True)) for p in args.runtime_path)
    os.chdir(repo)
    suite = unittest.TestSuite()
    for name in ('test_bridge', 'test_taskbook_inputs', 'test_mae_investment',
                 'test_sector_accounting', 'test_stability'):
        suite.addTests(unittest.defaultTestLoader.loadTestsFromName(name))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    simulator_imported = 'env.env_core' in sys.modules
    print(f'Publication checks: tests={result.testsRun}; '
          f'simulator_module_imported={simulator_imported}; '
          'real_calibration_ready=False; scientific_acceptance=NOT_EVALUATED')
    return 0 if (result.wasSuccessful() and result.testsRun == 77
                 and not simulator_imported and not result.skipped) else 1


if __name__ == '__main__':
    raise SystemExit(main())
