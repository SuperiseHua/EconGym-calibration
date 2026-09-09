"""Read-only inventory and independent mean-loss arithmetic, no simulator imports."""
import argparse
import hashlib
import json
import math
from pathlib import Path

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--repo',required=True);parser.add_argument('--output',required=True)
    a=parser.parse_args();repo=Path(a.repo);run=repo/'calibration/runs/price_stability_v1';out=Path(a.output)
    p=read(run/'protocol.json');s=read(run/'summary.json');f=read(run/'freeze.json');sel=read(run/'selections_before_validation.json')
    files=list(run.glob('*/*/eval_*/path_seed_*.json'));nrows=0;bad=[];mean_checks=[];hashes={};last_selection=(run/'selections_before_validation.json').stat().st_mtime_ns
    validation_after=True
    for path in files:
        hashes[str(path.relative_to(run))]=sha(path);r=read(path)
        if not r['valid']:bad.append(str(path))
        for row in r['trajectory']:
            nrows+=1
            if row['sector_accounting']['status']!='PASS' or row['investment_diagnostic']['status']!='PASS':bad.append(str(path))
        if path.parent.parent.name.startswith('validation_'):
            validation_after &= path.stat().st_mtime_ns>=last_selection
    def mean(label,seeds):
        rows=[read(label/'eval_0000'/f'path_seed_{seed}.json')['trajectory'] for seed in seeds]
        return [sum(r[t-1][k] for r in rows)/len(rows)/scale for t in range(2,9) for k,scale in p['scales'].items()]
    def rmse(x,y):return math.sqrt(sum((a-b)**2 for a,b in zip(x,y))/len(x))
    for prof in p['profiles']:
        name=prof['name'];root=run/name
        ys=[mean(root/f'target_{ti}',prof['target_seeds'][str(truth)]) for ti,truth in enumerate(p['synthetic_truth_prices'])]
        vals=[mean(root/f'validation_grid_{gi}',prof['validation_seeds']) for gi in range(5)]
        for ti,truth in enumerate(p['synthetic_truth_prices']):
            key=name+'_truth_'+str(truth);cell=s['cells'][key]
            errors=[abs(rmse(x,ys[ti])-cell['validation_grid_losses'][gi]) for gi,x in enumerate(vals)]
            for size in p['search_sizes']:
                prices=[rep[str(truth)][str(size)]['selected_price'] for rep in sel[name]]
                assert prices==cell['by_search_size'][str(size)]['selected_prices']
            mean_checks.append({'cell':key,'max_absolute_RMSE_difference':max(errors),'passed':max(errors)<1e-10})
    checked={str(path):sha(path)==expected for path,expected in f['files'].items()}
    tracked=all(sha(repo/name)==expected for name,expected in f['tracked_sources'].items())
    result={'status':'READ_ONLY_ARITHMETIC_AND_PRESERVATION_CHECK_NOT_NEW_SCIENTIFIC_PASS',
        'paths':len(files),'rows':nrows,'accounting_failures':bad,'validation_after_selection_file':validation_after,
        'selection_hash_unchanged':sha(run/'selections_before_validation.json')==s['selections_sha256'],
        'contract_hash_unchanged':sha(run/'contract.json')==f['contract_sha256'],
        'protocol_hash_unchanged':sha(run/'protocol.json')==f['protocol_sha256'],
        'frozen_files_unchanged':all(checked.values()),'tracked_sources_unchanged':tracked,
        'independent_arithmetic':mean_checks,'result_file_hashes':hashes,'physical_simulator_paths':0}
    result['verification_passed']=len(files)==1856 and nrows==14848 and not bad and validation_after and all(
        result[k] for k in ('selection_hash_unchanged','contract_hash_unchanged','protocol_hash_unchanged','frozen_files_unchanged','tracked_sources_unchanged')) and all(c['passed'] for c in mean_checks)
    out.mkdir(parents=True,exist_ok=False);(out/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='result_file_hashes'},indent=2))
    return 0 if result['verification_passed'] else 1

if __name__=='__main__':raise SystemExit(main())
