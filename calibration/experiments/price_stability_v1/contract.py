"""Frozen interpretation choices for the bounded development task."""
import datetime
import json

def make_contract(cal,sha):
    supplement=cal/'data/collected_supplement_v2/manifest.json'
    data=json.loads(supplement.read_text(encoding='utf-8'))
    for name,item in data['series'].items():
        if item['status']!='DOWNLOADED' or sha(supplement.parent/(name+'.csv'))!=item['sha256']:
            raise ValueError('raw observation hash mismatch')
    prior=cal/'runs/foundation_noise_v1/protocol.json'
    foundation=json.loads(prior.read_text(encoding='utf-8'))['profiles']['private_nonresidential']
    opened=json.loads((prior.parent/'summary.json').read_text(encoding='utf-8'))
    for name in ('upstream_default','private_nonresidential'):
        if opened['profiles'][name]['parameters']['price_adjustment_speed']['recommendation']!='candidate_free_parameter':
            raise ValueError('profile not supported by prior price screen')
    return {'schema_version':1,'status':'FROZEN_DEVELOPMENT_DEFINITIONS_NOT_REAL_FIT_APPROVAL',
        'frozen_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'reference_observations':{
            'country':'US','vintage':'exact local collected_supplement_v2 files, no refresh',
            'annualization':'mean of all 12 monthly or 4 quarterly SAAR observations; ratios of annual flows',
            'inflation':{'series':'PCEPI','formula':'annual_mean_index[y]/annual_mean_index[y-1]-1',
                'excluded':['core PCE','CPI','December-over-December']},
            'real_gdp_growth':{'series':'GDPC1','formula':'annual_mean[y]/annual_mean[y-1]-1'},
            'consumption_gdp':{'numerator':'PCE','denominator':'GDP','price_basis':'both current dollars'},
            'investment_gdp':{'numerator':'PNFI','denominator':'GDP','scope':'private nonresidential productive fixed investment',
                'excluded':['residential capital','government fixed investment','inventory investment']},
            'units':'fractions; report errors as percentage points',
            'reference_base_year':2022,'retrospective_reference_years':[2023,2024,2025],
            'fit_years':[],'holdout_years':[],'forecast_origin_vintage_available':False,
            'series_sha256':{k:v['sha256'] for k,v in data['series'].items()},'manifest_sha256':sha(supplement)},
        'non_comparability_frozen':{
            'GDP':'model produced output with unsold goods is not certified as BEA expenditure GDP',
            'inflation':'one-good transaction price is not a validated PCE consumption basket',
            'capital':'nonresidential reference scope does not certify the model wealth-financing mapping',
            'CEX_PSID_SCF':'micro acceptance and asset-to-bank mapping excluded from this macro stability contract'},
        'foundation_policy':{
            'primary':'unchanged upstream defaults; alpha=.36, delta=.06, initial_KY=null (wealth-minus-debt initialization)',
            'secondary_private_nonresidential':foundation,
            'secondary_status':'unchanged opened-development proxy, not measured physical depreciation or approved prior',
            'capital_stock_timing':'2022 year-end is a reference opening stock for 2023, not opening stock for 2022',
            'secondary_delta_definition':'2022 nominal depreciation / 2022 current-cost year-end net capital; proxy only',
            'GDP_initialization':'retain upstream 2022 reference 25474.6 billion USD; do not replace with revised GDP',
            'alpha':'.36 remains a structural reference, not an estimate from labor-share data',
            'policy_path':'unchanged real_society tax/rate rules, not a replay of realized 2023-2025 policies'},
        'optimization':{'fixed_reference_not_estimated':{'capital_adjustment_speed':.2,'inflation_expectation_lambda':.5},
            'only_free_parameter':'price_adjustment_speed','domain':[.15,.25],
            'basis':'previous local screen at N256, horizon8, no extension to untested domains'},
        'simulation_calendar':'synthetic t1..t8, t1 excluded from scoring; no assignment to real forecast years',
        'real_calibration_ready':False,'mechanism_changes_authorized':False,'prior_protocol_sha256':sha(prior)}

def make_protocol(contract):
    from stability_stats import check_seed_roles
    profiles=[]
    for i,name in enumerate(('upstream_default','private_nonresidential')):
        base=9120000+i*10000
        profiles.append({'name':name,'foundation':None if i==0 else contract['foundation_policy']['secondary_private_nonresidential'],
            'target_seeds':{str(t):list(range(base+j*100,base+j*100+64)) for j,t in enumerate((.175,.225))},
            'search_seed_pools':[list(range(base+1000+r*100,base+1000+r*100+16)) for r in range(6)],
            'validation_seeds':list(range(base+3000,base+3064))})
    p={'schema_version':1,'evidence_level':'development_predeclared_conditional_on_prior_opened_screen',
        'profiles':profiles,'N':256,'horizon':8,'periods':list(range(2,9)),
        'grid':[.15,.175,.2,.225,.25],'synthetic_truth_prices':[.175,.225],
        'fixed_parameters':{'capital_adjustment_speed':.2,'inflation_expectation_lambda':.5},
        'scales':{'inflation':.01,'real_gdp_growth':.02,'consumption_gdp':.05,'investment_gdp':.03},
        'search_sizes':[2,8,16],'search_repeats':6,'target_seed_count':64,'validation_seed_count':64,
        'selection':'minimum normalized mean-path RMSE; 1e-12 ties prefer .2, then smaller price',
        'search_bootstrap':400,'validation_bootstrap':1000,'bootstrap_seed':9140000,
        'bootstrap_unit':'whole seed trajectory, paired candidates, independent target resampling',
        'interval_status':'pointwise exploratory percentile intervals, not simultaneous or selection-adjusted guarantees',
        'target_reused_across_repeats':True,'validation_reused_across_repeats':True,
        'validation_rule':'save all search selections before any validation simulation',
        'validation_grid':'all five prices predeclared; grid-best only a noisy finite-grid regret diagnostic',
        'reported_metrics':['selection frequencies','parameter error to synthetic truth','pairwise rank agreement',
            'bootstrap candidate ambiguity sets','independent MAE','paired validation loss difference to default',
            'finite-grid validation regret','target split-half discrepancy','per-seed all-axis pass fraction'],
        'no_binary_new_identification_claim':True,'post_validation_tuning':False,
        'failure_policy':'stop on missing/invalid path, sector failure or source drift; retain artifacts',
        'expected_simulator_calls':74,'expected_physical_paths':1856,'equal_budget_optimizer_comparison':False}
    p['unique_seed_count']=check_seed_roles(p)
    return p
