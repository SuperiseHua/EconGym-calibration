"""Prepare a hash-bound contract, then run repeated price-only development search."""
import argparse
import datetime
import io
import json
import os
from pathlib import Path
import shutil
import sys
import time
import unittest

HERE=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run'])
    p.add_argument('--repo',required=True);p.add_argument('--output',required=True)
    p.add_argument('--runtime-path',action='append',default=[]);a=p.parse_args()
    repo=Path(a.repo).resolve(strict=True);cal=repo/'calibration';out=Path(a.output).resolve()
    sys.path[:0]=[str(HERE),str(cal),str(repo)];sys.path.extend(a.runtime_path);os.chdir(repo)
    from bridge import AgentEWMBackend,sha,source_hashes,write_json
    from stability_stats import losses,choose,bootstrap_losses,ambiguity_set,rank_agreement,check_seed_roles,valid_rows
    from contract import make_contract,make_protocol
    import numpy as np
    from mae_acceptance import assess_mae
    from taskbook_inputs import UNITS
    if a.mode=='prepare':
        out.mkdir(parents=True,exist_ok=False)
        log=io.StringIO();suite=unittest.defaultTestLoader.loadTestsFromName('test_stability')
        result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
        (out/'static_tests.txt').write_text(log.getvalue(),encoding='utf-8')
        if not result.wasSuccessful():raise RuntimeError('static tests failed before freeze')
        c=make_contract(cal,sha);protocol=make_protocol(c)
        write_json(out/'contract.json',c);write_json(out/'protocol.json',protocol)
        snapshot=out/'code_snapshot';snapshot.mkdir()
        files=list(HERE.glob('*.py'))+list(cal.glob('*.py'))
        for f in files:
            group='executor' if f.parent==HERE else 'calibration';(snapshot/group).mkdir(exist_ok=True)
            shutil.copy2(f,snapshot/group/f.name)
        files+=list((cal/'data/collected_supplement_v2').glob('*'))
        files+=[cal/'data/collected_v1/inputs/capital/BEA_FixedAssets_Section1.xlsx',
                cal/'runs/foundation_noise_v1/protocol.json',cal/'runs/foundation_noise_v1/summary.json']
        freeze={'files':{str(f):sha(f) for f in files if f.is_file()},'tracked_sources':source_hashes(repo),
            'contract_sha256':sha(out/'contract.json'),'protocol_sha256':sha(out/'protocol.json'),
            'tests':{'run':result.testsRun,'passed':True},'simulator_called':False}
        write_json(out/'freeze.json',freeze)
        print(json.dumps({'prepared':str(out),'tests':freeze['tests'],'contract_sha256':freeze['contract_sha256'],
            'calls':protocol['expected_simulator_calls'],'paths':protocol['expected_physical_paths']}));return 0
    freeze=json.loads((out/'freeze.json').read_text(encoding='utf-8'))
    def check_frozen():
        if any(sha(Path(f))!=v for f,v in freeze['files'].items()) or source_hashes(repo)!=freeze['tracked_sources']:
            raise RuntimeError('source/data drift since freeze')
        if sha(out/'contract.json')!=freeze['contract_sha256'] or sha(out/'protocol.json')!=freeze['protocol_sha256']:
            raise RuntimeError('contract/protocol drift')
    check_frozen();protocol=json.loads((out/'protocol.json').read_text(encoding='utf-8'));check_seed_roles(protocol)
    write_json(out/'started.json',{'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'freeze_sha256':sha(out/'freeze.json')})
    totals={'calls':0,'paths':0,'rows':0};cache={};selections={}
    def theta(price):return {**protocol['fixed_parameters'],'price_adjustment_speed':price}
    def evaluate(profile,label,price,seeds):
        check_frozen();f=profile['foundation'];card=None if f is None else {'schema_version':1,'status':'development_candidate',
            'parameters':f,'provenance':{k:{'source':'frozen secondary opened-development proxy',
                'definition':'unchanged prior screen proxy, not approved real-data foundation','unit':UNITS[k],'year':2022} for k in UNITS}}
        start=time.monotonic();backend=AgentEWMBackend(repo,out/profile['name']/label,256,8,a.runtime_path,card)
        r=backend.evaluate(theta(price),tuple(seeds));totals['calls']+=1;totals['paths']+=len(r['paths'])
        valid_rows(r,seeds,8);totals['rows']+=sum(len(x['trajectory']) for x in r['paths'])
        print(json.dumps({'profile':profile['name'],'stage':label,'paths':len(seeds),'seconds':round(time.monotonic()-start,1)}),flush=True)
        x=np.array([[row[k]/scale for row in path['trajectory'][1:] for k,scale in protocol['scales'].items()] for path in r['paths']])
        if not np.isfinite(x).all():raise ValueError('nonfinite metric')
        return r,x
    for pi,profile in enumerate(protocol['profiles']):
        name=profile['name'];targets={}
        for ti,trueprice in enumerate(protocol['synthetic_truth_prices']):
            r,y=evaluate(profile,f'target_{ti}',trueprice,profile['target_seeds'][str(trueprice)])
            targets[str(trueprice)]=(r,y)
        reps=[]
        for rep,seeds in enumerate(profile['search_seed_pools']):
            xs=[]
            for gi,price in enumerate(protocol['grid']):
                _,x=evaluate(profile,f'search_{rep}_grid_{gi}',price,seeds);xs.append(x)
            xs=np.stack(xs);report={}
            for ti,trueprice in enumerate(protocol['synthetic_truth_prices']):
                y=targets[str(trueprice)][1];by_size={}
                for size in protocol['search_sizes']:
                    v=losses(xs[:,:size,:],y);best=choose(v,protocol['grid'])
                    boot=bootstrap_losses(xs[:,:size,:],y,400,9140000+pi*10000+rep*100+ti*30+size)
                    delta=boot-boot[:,[best]]
                    by_size[str(size)]={'losses':v.tolist(),'selected_index':best,'selected_price':protocol['grid'][best],
                        'ambiguity_indices':ambiguity_set(boot,best),
                        'pointwise_delta_to_selected_95pct':np.quantile(delta,[.025,.975],axis=0).T.tolist()}
                report[str(trueprice)]=by_size
            reps.append(report)
        selections[name]=reps;cache[name]={'targets':targets}
    write_json(out/'selections_before_validation.json',selections)
    selection_hash=sha(out/'selections_before_validation.json')
    summary={'evidence_level':protocol['evidence_level'],'cells':{},'totals':totals,
        'selections_sha256':selection_hash,'real_data_fit':False,'full_taskbook_acceptance':False}
    for pi,profile in enumerate(protocol['profiles']):
        name=profile['name'];raw=[];xs=[]
        for gi,price in enumerate(protocol['grid']):
            r,x=evaluate(profile,f'validation_grid_{gi}',price,profile['validation_seeds']);raw.append(r);xs.append(x)
        xs=np.stack(xs)
        for ti,trueprice in enumerate(protocol['synthetic_truth_prices']):
            target_raw,y=cache[name]['targets'][str(trueprice)];v=losses(xs,y)
            boot=bootstrap_losses(xs,y,1000,9160000+pi*100+ti)
            gridbest=choose(v,protocol['grid']);paired=boot-boot[:,[2]]
            intervals=np.quantile(paired,[.025,.975],axis=0).T;mae=[]
            for gi,r in enumerate(raw):
                c={'schema_version':1,'target_kind':'synthetic','evidence_level':'development_predeclared',
                    'horizon':8,'expected_seeds':profile['validation_seeds'],'periods':list(range(2,9)),
                    'units':'fraction','thresholds':protocol['scales'],
                    'targets':{k:{str(t):target_raw['moments'][f'{k}:t{t}'] for t in range(2,9)} for k in protocol['scales']}}
                m=assess_mae(r,c);write_json(out/f'{name}_truth_{ti}_grid_{gi}_mae.json',m)
                mae.append({'mean_path_MAE':m['macro_mae_status'],'seed_all_axes_pass_fraction':m['all_axes_per_seed_pass_fraction']})
            cell={'truth_price':trueprice,'validation_grid_losses':v.tolist(),'validation_grid_best_index_diagnostic':gridbest,
                'pointwise_validation_delta_default_95pct':intervals.tolist(),'validation_MAE':mae,
                'target_split_half_normalized_RMS':float(np.sqrt(np.mean((y[:32].mean(0)-y[32:].mean(0))**2))),
                'by_search_size':{}}
            for size in protocol['search_sizes']:
                rows=[rep[str(trueprice)][str(size)] for rep in selections[name]];indices=[r['selected_index'] for r in rows]
                counts=[indices.count(i) for i in range(5)]
                kendall=[rank_agreement(rows[i]['losses'],rows[j]['losses']) for i in range(6) for j in range(i)]
                cell['by_search_size'][str(size)]={'selected_prices':[protocol['grid'][i] for i in indices],
                    'selection_counts':counts,'modal_fraction':max(counts)/6,'mean_pairwise_rank_agreement':float(np.mean(kendall)),
                    'mean_abs_parameter_error':float(np.mean([abs(protocol['grid'][i]-trueprice) for i in indices])),
                    'ambiguity_set_sizes':[len(r['ambiguity_indices']) for r in rows],
                    'validation_losses':[float(v[i]) for i in indices],
                    'mean_validation_loss':float(np.mean(v[indices])),
                    'mean_finite_grid_regret':float(np.mean(v[indices]-v[gridbest])),
                    'default_improvement_pointwise_interval_upper_below_zero_count':int(sum(intervals[i,1]<0 for i in indices)),
                    'mean_path_MAE_pass_count':sum(mae[i]['mean_path_MAE']=='PASS' for i in indices)}
            summary['cells'][name+'_truth_'+str(trueprice)]=cell
    check_frozen()
    if sha(out/'selections_before_validation.json')!=selection_hash:raise RuntimeError('selection changed')
    if totals['calls']!=74 or totals['paths']!=1856:raise RuntimeError('budget mismatch')
    summary['tracked_sources_unchanged']=True;summary['frozen_files_unchanged']=True
    write_json(out/'summary.json',summary)
    print(json.dumps({'completed':str(out),'totals':totals}));return 0

if __name__=='__main__':
    sys.dont_write_bytecode=True
    raise SystemExit(main())
