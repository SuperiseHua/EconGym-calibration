"""Independent arithmetic and tamper checks; no simulator imports."""
import json
from pathlib import Path
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_access import CollectedData, sha
import csv
import io
import tempfile
import shutil
import statistics


def main():
    bundle = Path(sys.argv[1]).resolve()
    if '--seal-prepared' in sys.argv:
        prepared = ['data_contract.json','macro_observations_2021_2025.json','capital_candidates_2022.json',
                    'scf_schema_check.json','psid_schema_check.json']
        with (bundle/'prepared_manifest.json').open('x', encoding='utf-8') as f:
            json.dump({'files':[{'path':name,'sha256':sha(bundle/name)} for name in prepared]},f,indent=2)
    data = CollectedData(bundle)
    with data.source('macro/quarterly.csv').open(encoding='utf-8-sig') as f: q = list(csv.DictReader(f))
    with data.source('macro/monthly.csv').open(encoding='utf-8-sig') as f: m = list(csv.DictReader(f))
    target = data.observations([2022])[0]
    gdp = statistics.mean(float(r['GDP']) for r in q if r['observation_date'].startswith('2022-'))
    pce = statistics.mean(float(r['PCE']) for r in m if r['observation_date'].startswith('2022-'))
    checks = {'raw_and_prepared_manifest_valid':True,
        'GDP_2022_independently_recomputed':abs(gdp-target['annual_levels']['GDP']) < 1e-8,
        'CY_2022_independently_recomputed':abs(pce/gdp-target['consumption_gdp']) < 1e-12,
        'headline_PCE_stays_missing': target['headline_pce_inflation'] is None,
        'no_auto_apply':data.contract['auto_apply_parameters'] is False,
        'no_fit_or_holdout_assigned':data.contract['fit_years'] is None and data.contract['holdout_years'] is None}
    try: data.calibration_targets()
    except RuntimeError: checks['unapproved_fit_refused'] = True
    else: checks['unapproved_fit_refused'] = False
    try: data.observations([1991])
    except ValueError: checks['unprepared_year_refused'] = True
    else: checks['unprepared_year_refused'] = False
    # Alter a tiny throwaway prepared copy, never the original bundle or scientific records.
    with tempfile.TemporaryDirectory(prefix='agentewm_data_test_') as temp:
        root = Path(temp)
        (root/'data_contract.json').write_text(json.dumps(data.contract), encoding='utf-8')
        (root/'manifest.json').write_text(json.dumps({'files':[]}), encoding='utf-8')
        (root/'prepared_manifest.json').write_text(json.dumps({'files':[{'path':'observation.json','sha256':'0'*64}]}), encoding='utf-8')
        (root/'observation.json').write_text('{}', encoding='utf-8')
        try: CollectedData(root)
        except ValueError: checks['tampered_prepared_data_refused'] = True
        else: checks['tampered_prepared_data_refused'] = False
        # Check the path resolver against a real file outside the bundle.
        try: data.resolve(__file__)
        except ValueError: checks['path_escape_refused'] = True
        else: checks['path_escape_refused'] = False
    with (bundle/'interface_checks.json').open('x',encoding='utf-8') as f:
        json.dump(checks,f,indent=2)
    print(json.dumps(checks))
    if not all(checks.values()): raise SystemExit(1)


if __name__ == '__main__': main()
