"""Frozen bounded foundation scan and nested-sample noise experiment."""
import argparse
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--repo', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--runtime-path', action='append', default=[])
    args = p.parse_args()
    repo, out = Path(args.repo).resolve(strict=True), Path(args.output).resolve()
    sys.path[:0] = [str(repo/'calibration'), str(repo)]
    sys.path.extend(args.runtime_path)
    os.chdir(repo)
    import numpy as np
    from bridge import AgentEWMBackend, sha, source_hashes, write_json, trajectory_moments
    from core import Target, score
    from mae_acceptance import assess_mae
    from taskbook_inputs import UNITS
    out.mkdir(parents=True, exist_ok=False)
    snapshot=out/'code_snapshot'; snapshot.mkdir()
    for f in (repo/'calibration').glob('*.py'): shutil.copy2(f, snapshot/f.name)
    capitals=json.loads((repo/'calibration/data/collected_v1/capital_candidates_2022.json').read_text(encoding='utf-8'))
    profiles={'upstream_default':None}
    for row in capitals:
        profiles[row['sector']]={'alpha':.36, 'depreciation_rate':row['depreciation_over_yearend_capital_candidate'],
                                'initial_capital_output_ratio':row['capital_to_total_gdp_candidate']}
    for key,value,label in [('alpha',.32,'private_alpha32'),('alpha',.40,'private_alpha40'),
                             ('depreciation_rate',.04,'private_delta04'),('depreciation_rate',.09,'private_delta09')]:
        profiles[label]={**profiles['private_total'],key:value}
    params={'capital_adjustment_speed':.2,'price_adjustment_speed':.2,'inflation_expectation_lambda':.5}
    steps={'capital_adjustment_speed':.05,'price_adjustment_speed':.05,'inflation_expectation_lambda':.1}
    scales={'inflation':.01,'real_gdp_growth':.02,'consumption_gdp':.05,'investment_gdp':.03}
    rng=random.Random(909510)
    candidates=[params.copy()]+[{'capital_adjustment_speed':rng.uniform(.05,.4),
        'price_adjustment_speed':rng.uniform(.05,.4),'inflation_expectation_lambda':rng.uniform(.1,.9)} for _ in range(11)]
    protocol={'evidence_level':'development_predeclared_screen_not_confirmatory',
        'profiles':profiles,'range_interpretation':'data-anchored sensitivity candidates, not approved priors or equivalent sector mappings',
        'alpha_range_basis':'analyst sensitivity around upstream .36, not an estimated confidence interval',
        'scan':{'N':256,'horizon':8,'seeds':list(range(909001,909007)),'center':params,'steps':steps,
                'scales':scales,'minimum_half_step_RMS':.05,'minimum_unique_component_fraction':.2,
                'minimum_bootstrap_pass_fraction':.8,'bootstrap_repeats':200,
                'minimum_singular_ratio':.05,'note':'heuristic development screens, not calibrated statistical tests'},
        'noise':{'N':[256,1000],'horizon':8,'pool_seeds':list(range(909101,909165)),
                 'batch_mean_sizes':[2,8,16,32],'bootstrap_repeats':500,'random_seed':909600},
        'search':{'N':1000,'horizon':8,'target_seeds':list(range(909201,909233)),
                  'search_seeds':list(range(909301,909317)),'nested_search_sizes':[2,16],
                  'validation_seeds':list(range(909401,909417)),'candidates':candidates,
                  'target_parameters':{'capital_adjustment_speed':.12,'price_adjustment_speed':.3,'inflation_expectation_lambda':.65},
                  'selection':'minimum standardized RMSE over same fixed candidate bank; not method superiority test'},
        'original_sources':source_hashes(repo),'optimization_after_validation':False,
        'formal_seeds_used':False,'real_data_fit':False}
    write_json(out/'protocol.json',protocol)
    totals={'calls':0,'requested_paths':0,'returned_paths':0,'valid_paths':0}
    def evaluate(label,n,h,seeds,theta,foundation=None):
        card=None if foundation is None else {'schema_version':1,'status':'development_candidate','parameters':foundation,
            'provenance':{k:{'source':'collected_v1 BEA candidates plus explicitly assumed sensitivity deviations',
                            'definition':'development sensitivity only; sector/yearend mapping not approved', 'unit':UNITS[k],'year':2022} for k in UNITS}}
        b=AgentEWMBackend(repo,out/label,n,h,args.runtime_path,card)
        totals['calls']+=1;totals['requested_paths']+=len(seeds)
        start=time.monotonic(); result=b.evaluate(theta,tuple(seeds))
        totals['returned_paths']+=len(result['paths']);totals['valid_paths']+=sum(p['valid'] for p in result['paths'])
        print(json.dumps({'stage':label,'valid':result['valid'],'paths':len(seeds),'seconds':round(time.monotonic()-start,1)}),flush=True)
        return result
    def accounting(result):
        rows=[row for p in result['paths'] for row in p['trajectory']]
        return {'rows':len(rows),'all_passed':bool(rows) and result['valid'] and all(
            row.get('sector_accounting',{}).get('status')=='PASS' and row['investment_diagnostic']['status']=='PASS' for row in rows),
            'failures':[{'seed':p['seed'],'period':r['period'],'sector':r.get('sector_accounting'),
                         'investment_status':r['investment_diagnostic']['status']} for p in result['paths'] for r in p['trajectory']
                        if r.get('sector_accounting',{}).get('status')!='PASS' or r['investment_diagnostic']['status']!='PASS']}
    def matrix(result):
        return np.array([[row[a]/scales[a] for row in path['trajectory'][1:] for a in scales] for path in result['paths']])
    def gate(J):
        ratios=[];rms=[]
        for j in range(3):
            others=np.delete(J,j,axis=1);col=J[:,j]
            remainder=col-others@np.linalg.lstsq(others,col,rcond=None)[0]
            rms.append(float(np.sqrt(np.mean(col**2))))
            ratios.append(float(np.linalg.norm(remainder)/max(np.linalg.norm(col),1e-15)))
        sv=np.linalg.svd(J,compute_uv=False)
        return np.array(rms),np.array(ratios),float(sv[-1]/sv[0]) if sv[0]>1e-15 else 0.
    summary={'profiles':{},'noise':{},'search':{},'evidence_level':protocol['evidence_level']}
    # First verify all-sector instrumentation on one opened development route.
    replay=evaluate('accounting_replay',256,4,[908841],{})
    import copy
    replay_path=copy.deepcopy(replay['paths'][0])
    for r in replay_path['trajectory']:r.pop('sector_accounting',None)
    prior=json.loads((repo/'calibration/runs/mae_investment_v2/logging_invariance/eval_0000/path_seed_908841.json').read_text(encoding='utf-8'))
    summary['accounting_logging_invariant']=replay_path==prior
    summary['accounting_replay']=accounting(replay)
    if not summary['accounting_logging_invariant'] or not summary['accounting_replay']['all_passed']:
        write_json(out/'summary.json',{**summary,'stopped_at':'accounting_replay','totals':totals});return 2
    for label,foundation in profiles.items():
        base=evaluate('scan_'+label+'_base',256,8,protocol['scan']['seeds'],params,foundation)
        report={'base_valid':base['valid'],'base_accounting':accounting(base)}
        if not base['valid'] or not report['base_accounting']['all_passed']:
            report['status']='INVALID_FOUNDATION_OR_ACCOUNTING'
            report['exceptions']=[p.get('exception') for p in base['paths']]
            summary['profiles'][label]=report;continue
        contrasts=[];complete=True
        for key,step in steps.items():
            minus=evaluate('scan_'+label+'_'+key+'_minus',256,8,protocol['scan']['seeds'],{**params,key:params[key]-step},foundation)
            plus=evaluate('scan_'+label+'_'+key+'_plus',256,8,protocol['scan']['seeds'],{**params,key:params[key]+step},foundation)
            if not minus['valid'] or not plus['valid'] or not accounting(minus)['all_passed'] or not accounting(plus)['all_passed']:
                complete=False;break
            contrasts.append((matrix(plus)-matrix(minus))/2)
        if not complete: report['status']='INVALID_PERTURBATION_OR_ACCOUNTING'
        else:
            cube=np.stack(contrasts,axis=2);J=cube.mean(axis=0);rms,unique,singular=gate(J)
            boot=np.random.default_rng(909700)
            fractions=np.zeros(3)
            for _ in range(200):
                bR,bU,_=gate(cube[boot.integers(0,len(cube),len(cube))].mean(axis=0))
                fractions+=(bR>=.05)&(bU>=.2)
            fractions/=200
            report.update(status='DEVELOPMENT_SCREEN_PASS' if singular>=.05 and all(fractions>=.8) and all(unique>=.2) and all(rms>=.05) else 'WEAK_IDENTIFICATION',
                singular_ratio=singular,parameters={k:{'half_step_RMS':float(rms[j]),'unique_fraction':float(unique[j]),
                    'bootstrap_screen_pass_fraction':float(fractions[j]),'recommendation':'candidate_free_parameter' if rms[j]>=.05 and unique[j]>=.2 and fractions[j]>=.8 else 'temporarily_fix_or_report_weak'} for j,k in enumerate(steps)})
            rows=[r for p in base['paths'] for r in p['trajectory'][1:]]
            report['base_binding_fractions']={k:sum(r['investment_diagnostic']['binding_flags'][k] for r in rows)/len(rows)
                for k in ('target_cap_active','goods_rationing_material','credit_rationing_material','investment_zero_floor_active')}
        summary['profiles'][label]=report
        write_json(out/('profile_'+label+'.json'),report)
    for n in (256,1000):
        result=evaluate('noise_N'+str(n),n,8,protocol['noise']['pool_seeds'],protocol['search']['target_parameters'])
        if not result['valid'] or not accounting(result)['all_passed']:
            summary['noise'][str(n)]={'status':'INVALID','accounting':accounting(result)};continue
        values=matrix(result);boot=np.random.default_rng(909600)
        report={'per_axis_seed_SD':{axis:float(np.sqrt(np.mean(np.var(values[:,j::4]*scales[axis],axis=0,ddof=1)))) for j,axis in enumerate(scales)},
                'mean_estimator_bootstrap_RMS':{},'note':'empirical bootstrap conditional on 64-seed pool, not real-world uncertainty','accounting':accounting(result)}
        for size in (2,8,16,32):
            means=values[boot.integers(0,len(values),(500,size))].mean(axis=1)
            report['mean_estimator_bootstrap_RMS'][str(size)]=float(np.sqrt(np.mean((means-values.mean(axis=0))**2)))
        summary['noise'][str(n)]=report
    target=evaluate('larger_target',1000,8,protocol['search']['target_seeds'],protocol['search']['target_parameters'])
    if not target['valid'] or not accounting(target)['all_passed']:
        summary['search']={'status':'TARGET_INVALID'}
    else:
        target_values={k:v for k,v in target['moments'].items() if not k.endswith(':t1')}
        targets={k:Target(v,scales[k.split(':')[0]],scales[k.split(':')[0]]) for k,v in target_values.items()}
        write_json(out/'larger_targets.json',target_values)
        bank=[]
        for i,theta in enumerate(candidates):
            result=evaluate('candidate_'+str(i).zfill(2),1000,8,protocol['search']['search_seeds'],theta)
            row={'index':i,'parameters':theta,'valid':result['valid'] and accounting(result)['all_passed'],'losses':{}}
            if row['valid']:
                for size in (2,16):
                    subset={'valid':True,'moments':trajectory_moments(result['paths'][:size],8)}
                    row['losses'][str(size)]=score(subset,targets)['loss']
            bank.append(row)
        write_json(out/'candidate_bank.json',bank)
        selected={str(size):min((r for r in bank if r['valid']),key=lambda r:r['losses'][str(size)])['index'] for size in (2,16)} if any(r['valid'] for r in bank) else {}
        write_json(out/'selected_before_validation.json',selected)
        validations={}
        for i in sorted(set([0]+list(selected.values()))):
            result=evaluate('validation_'+str(i).zfill(2),1000,8,protocol['search']['validation_seeds'],candidates[i])
            c={'schema_version':1,'target_kind':'synthetic','evidence_level':'development_predeclared','horizon':8,
                'expected_seeds':protocol['search']['validation_seeds'],'periods':list(range(2,9)),'units':'fraction',
                'thresholds':scales,'targets':{a:{str(t):target_values[f'{a}:t{t}'] for t in range(2,9)} for a in scales}}
            a=assess_mae(result,c);write_json(out/f'validation_{i:02d}_mae.json',a)
            validations[str(i)]={'normalized_RMSE':score(result,targets)['loss'],'macro_MAE':a['macro_mae_status'],
                'per_seed_all_axes_pass_fraction':a.get('all_axes_per_seed_pass_fraction'),'accounting':accounting(result)}
        summary['search']={'selected':selected,'candidates':bank,'validations':validations,
            'selection_bank_shared':True,'larger_target_seeds':32,'search_sizes':[2,16],'validation_seeds':16,
            'old_effect_run_not_replaced':True}
    summary.update(totals=totals,tracked_sources_unchanged=protocol['original_sources']==source_hashes(repo),
                   full_taskbook_acceptance=False,real_data_fit=False)
    write_json(out/'summary.json',summary)
    print(json.dumps({'done':str(out),'totals':totals,'profiles':{k:v['status'] for k,v in summary['profiles'].items()}}),flush=True)
    return 0


if __name__=='__main__':
    sys.dont_write_bytecode=True
    raise SystemExit(main())
