"""Read-only evidence inventory and opening/yearend stock alignment, no scoring."""
import argparse
import hashlib
import json
from pathlib import Path
import pandas as pd


def main():
    p=argparse.ArgumentParser();p.add_argument('--calibration',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();cal=Path(a.calibration);out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    run=cal/'runs/foundation_noise_v1'
    paths=list(run.glob('*/eval_*/path_seed_*.json'))
    totals={'paths':len(paths),'rows':0,'sector_pass_rows':0,'investment_pass_rows':0,'equation_family_uses':0,'scalar_equations_checked':0}
    errors=[];hashes={};initial=[]
    for f in paths:
        raw=f.read_bytes();hashes[str(f.relative_to(run))]=hashlib.sha256(raw).hexdigest()
        data=json.loads(raw)
        for row in data['trajectory']:
            totals['rows']+=1
            s=row.get('sector_accounting',{})
            totals['sector_pass_rows']+=s.get('status')=='PASS'
            totals['investment_pass_rows']+=row['investment_diagnostic']['status']=='PASS'
            totals['equation_family_uses']+=len(s.get('checks',{}))
            totals['scalar_equations_checked']+=sum(v.get('equations_checked',0) for v in s.get('checks',{}).values())
            if s.get('status')!='PASS':errors.append({'file':str(f),'period':row['period']})
        if f.parent.parent.name.startswith('scan_') and f.parent.parent.name.endswith('_base') and data['seed']==909001:
            diag=data['trajectory'][0]['investment_diagnostic']
            initial.append({'profile':f.parent.parent.name,'user_cost':diag['inputs']['capital_user_cost'],
                'upper_MPK':diag['derived']['marginal_product_at_upper'], 'target_cap_active':diag['binding_flags']['target_cap_active']})
    workbook=cal/'data/collected_v1/inputs/capital/BEA_FixedAssets_Section1.xlsx'
    before=hashlib.sha256(workbook.read_bytes()).hexdigest()
    tables={n:pd.read_excel(workbook,sheet_name=n,header=None) for n in ('FAAt101-A','FAAt103-A')}
    def cell(sheet,code,year):
        table=tables[sheet];header=table[table.iloc[:,0].astype(str).eq('Line')]
        if len(header)!=1:raise ValueError('ambiguous header')
        cols=[c for c,v in header.iloc[0].items() if str(v) in (str(year),str(year)+'.0')]
        rows=table[table.iloc[:,2].astype(str).str.lower().eq(code)]
        if len(cols)!=1 or rows.empty or rows.iloc[:,cols[0]].nunique()!=1:raise ValueError('ambiguous cell')
        return float(rows.iloc[0,cols[0]])
    capital=[]
    for sector,kcode,dcode in [('private_nonresidential','k1ntotl1es00','m1ntotl1es00'),('private_total','k1ptotl1es00','m1ptotl1es00'),('all_fixed_assets','k1ttotl1es00','m1ttotl1es00')]:
        opening=cell('FAAt101-A',kcode,2021);closing=cell('FAAt101-A',kcode,2022);dep=cell('FAAt103-A',dcode,2022)
        capital.append({'sector':sector,'2021_yearend_current_cost_million_USD':opening,
            '2022_yearend_current_cost_million_USD':closing,'2022_depreciation_million_USD':dep,
            'depreciation_over_opening_stock_candidate':dep/opening,
            'depreciation_over_closing_stock_candidate':dep/closing,
            'approved_model_delta':False,'warning':'different valuation dates; ratio not automatically physical depreciation rate'})
    supplement=cal/'data/collected_supplement_v2'
    manifest=json.loads((supplement/'manifest.json').read_text(encoding='utf-8'))
    verification={k:hashlib.sha256((supplement/(k+'.csv')).read_bytes()).hexdigest()==v['sha256'] and
        all(x['observations']==x['expected'] for x in v['annual'].values())
        for k,v in manifest['series'].items() if v['status']=='DOWNLOADED'}
    result={'status':'READ_ONLY_INVENTORY_NOT_NEW_EMPIRICAL_PASS','accounting':totals,'failures':errors,
        'profile_initial_costs':initial,'capital_timing_candidates':capital,
        'workbook_unchanged':before==hashlib.sha256(workbook.read_bytes()).hexdigest(),
        'supplement_hash_and_year_completeness':verification,'source_hashes':hashes,
        'no_simulator_calls':True,'no_rescoring':True}
    (out/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='source_hashes'},indent=2))


if __name__=='__main__':main()
