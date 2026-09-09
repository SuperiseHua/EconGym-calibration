"""Add a portable, allow-listed data bundle. Never rewrite source data or engine.
Run with the bundled data-analysis Python (pandas/openpyxl read support).
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile
import pandas as pd


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()


def write(path, value):
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8')) != value:
            raise ValueError(f'refusing different existing output: {path}')
        return
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)


def annual(frame, name, count):
    date_col = 'observation_date' if 'observation_date' in frame else 'date'
    x = frame[[date_col, name]].copy()
    x[name] = pd.to_numeric(x[name], errors='coerce')
    x[date_col] = pd.to_datetime(x[date_col], errors='raise')
    if x[date_col].duplicated().any(): raise ValueError(f'duplicate time key: {name}')
    x['year'] = x[date_col].dt.year
    out = {}
    for y, group in x.groupby('year'):
        if len(group) == count and group[name].notna().all():
            out[int(y)] = float(group[name].mean())
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--workspace', required=True)
    p.add_argument('--repo', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--resume-identical', action='store_true')
    args = p.parse_args()
    workspace, repo, out = Path(args.workspace).resolve(), Path(args.repo).resolve(), Path(args.output).resolve()
    if not out.is_relative_to(repo/'calibration/data'):
        raise ValueError('only new calibration/data bundle directories may be created')
    out.mkdir(parents=True, exist_ok=args.resume_identical)
    sources = []
    def add(relative, category, role, use, url=None):
        source = (workspace/relative).resolve(strict=True)
        # This static allow-list never traverses custody, formal results or old origin cards.
        if not source.is_relative_to(workspace): raise ValueError('outside source workspace')
        dest = out/'inputs'/category/source.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        before = digest(source)
        if dest.exists():
            if not args.resume_identical or digest(dest) != before: raise ValueError('existing nonidentical source copy')
        else:
            shutil.copyfile(source, dest)
        if digest(dest) != before or digest(source) != before: raise ValueError('copy changed bytes')
        sources.append(dict(id=f'{category}/{source.name}', source_relative_path=relative,
                            bundled_path=dest.relative_to(out).as_posix(), bytes=source.stat().st_size,
                            sha256=before, role=role, permitted_use=use, official_url=url))
        return dest
    us = '数据/美国/'
    q = add(us+'fredgraph/quarterly.csv', 'macro', 'raw quarterly national accounts', 'retrospective_observations')
    m = add(us+'fredgraph/monthly.csv', 'macro', 'raw monthly personal accounts', 'retrospective_observations')
    add(us+'fredgraph/README.txt', 'macro', 'units and download vintage', 'metadata')
    nipa = pd.read_csv(q)
    monthly = pd.read_csv(m)
    pnfi = add(us+'核心货币投资数据/raw/FRED_PNFI_nominal_quarterly.csv', 'investment',
               'private nonresidential fixed investment only', 'sector_candidate_not_total_fixed_investment',
               'https://fred.stlouisfed.org/series/PNFI')
    corepce = add(us+'核心货币投资数据/raw/FRED_PCEPILFE_monthly.csv', 'prices',
                  'core PCE, excludes food/energy', 'diagnostic_not_headline_PCE', 'https://fred.stlouisfed.org/series/PCEPILFE')
    add(us+'核心货币投资数据/raw/FRED_FEDFUNDS_monthly.csv', 'policy', 'historical policy rate', 'historical_route_preparation')
    add(us+'核心货币投资数据/processed/US_core_data_manifest_M1_M3_v1.csv', 'metadata', 'source catalog', 'metadata')
    add(us+'核心货币投资数据/processed/SPF_one_year_inflation_expectations_2000_2025_v1.csv', 'expectations',
        'survey expectations, not realized inflation', 'horizon_and_definition_review_required')
    add(us+'规范化原始数据/fred/02_fred_labor_prices.csv', 'labor', 'employment/wages/CPI', 'auxiliary_not_PCE')
    add(us+'规范化原始数据/fred/03_fred_monetary_fiscal.csv', 'policy', 'fiscal and monetary accounts', 'unit_vintage_review')
    prod = us+'后续实验核心数据/raw/production_capital/'
    bea = add(prod+'BEA_FixedAssets_Section1.xlsx', 'capital', 'official stock/depreciation/investment tables', 'sector_candidate')
    add(prod+'BEA_FixedAssets_Section2.xlsx', 'capital', 'detailed private capital tables', 'sector_candidate')
    for f in ['FRED_MPU4910141.csv', 'FRED_PRS85006173.csv', 'FRED_HOANBS.csv', 'FRED_MFPPBS.csv']:
        add(prod+f, 'labor', 'labor share/index/hours/productivity', 'unit_sector_review_required')
    add(us+'processed_v1/derived/e4_bea_stock_flow_anchors_2022.csv', 'legacy_reference',
        'old E4 derived anchors; includes residual investment', 'reference_only_never_auto_apply')
    scf = add(us+'scfp2022excel/SCFP2022.csv', 'scf', 'official public extract with household and implicate keys',
              'micro_mapping_preparation', 'https://www.federalreserve.gov/econres/scfindex.htm')
    rw = add(us+'规范化原始数据/scf/scf2022rw1s.zip', 'scf', 'official replicate weight archive', 'join_and_survey_variance_pending')
    add(us+'规范化原始数据/scf/scfp2022excel.zip', 'scf', 'official original public extract archive', 'source_reproduction')
    add('integrations/AgentEWM-calibration-20260908/agents/data/advanced_scfp2022_1110.csv',
        'existing_model_input', 'collaborator current initialized SCF file', 'comparison_only_original_unchanged')
    behavior = us+'后续实验核心数据/'
    for f in ['intrvw22.zip', 'diary22.zip', 'ce-pumd-interview-diary-dictionary.xlsx']:
        add(behavior+'raw/behavior_bc/'+f, 'cex_raw', '2022 release archive/dictionary', 'rebuild_reference_period_targets')
    for f in ['cex2022_release_cu_targets_v1.csv', 'cex2022_release_weighted_moments_v1.csv']:
        add(behavior+'processed/'+f, 'cex_legacy', 'old release-based processing', 'quarantined_definition_review')
    for f in ['atus2022_labor_targets_v1.csv', 'atus2022_weighted_moments_v1.csv']:
        add(behavior+'processed/'+f, 'atus', '2022 labor targets', 'auxiliary_not_labor_rule_replacement')
    add(behavior+'processed/scf2022_bc_augmented_targets_v1.csv', 'legacy_fusion',
        'statistical matching; not record-linked surveys', 'quarantined_do_not_replace_household_policy')
    add(behavior+'processed/supplemental_data_quality_report_v1.json', 'metadata', 'historical processing caveats', 'metadata')
    for f in ['psid_reference_person_family_panel_2011_2023.csv.gz', 'psid_family_key_panel_2011_2023.csv.gz',
              'psid_key_variable_map.csv', 'psid_individual_link_variable_map_2011_2023.csv',
              'psid_individual_link_quality_summary.csv', 'psid_variable_availability.csv',
              'psid_calibration_moments.csv', 'psid_wave_quality_summary.csv']:
        add(us+'processed_v1/psid/'+f, 'psid', 'longitudinal panel or mapping metadata', 'later_income_process_estimation')
    add('experiments/process_supplemental_behavior_demography_market.py', 'processing_provenance',
        'prior CEX/ATUS/fusion processor', 'inspect_only_not_executed')
    add(us+'scripts/process_all_us_data.py', 'processing_provenance', 'prior PSID and data conversion', 'inspect_only_not_executed')
    add(us+'processed_v1/美国数据全量处理说明与正式实验准备_v2_20260803.md', 'metadata',
        'historical preparation caveats', 'metadata')

    # Only complete years; SAAR annual levels are means, not sums.
    series = {k: annual(nipa, k, 4) for k in ['GDP','GDPC1','GPDI','GCE','NETEXP','GDPDEF']}
    series['PCE'] = annual(monthly, 'PCE', 12)
    series['PNFI'] = annual(pd.read_csv(pnfi), 'PNFI', 4)
    series['PCEPILFE'] = annual(pd.read_csv(corepce), 'PCEPILFE', 12)
    annual_rows = []
    for year in range(2021, 2026):
        values = {key: value.get(year) for key, value in series.items()}
        gdp = values['GDP']
        ratio = lambda key: values[key]/gdp if values[key] is not None and gdp else None
        row = {'year': year, 'annual_levels': values, 'real_gdp_growth':
               values['GDPC1']/series['GDPC1'][year-1]-1 if values['GDPC1'] and year-1 in series['GDPC1'] else None,
               'consumption_gdp': ratio('PCE'), 'private_nonresidential_investment_gdp_candidate': ratio('PNFI'),
               'gross_private_domestic_investment_gdp_reference': ratio('GPDI'),
               'government_gdp_reference': ratio('GCE'), 'net_exports_gdp_reference': ratio('NETEXP'),
               'headline_pce_inflation': None, 'core_pce_inflation_diagnostic':
               values['PCEPILFE']/series['PCEPILFE'][year-1]-1 if values['PCEPILFE'] and year-1 in series['PCEPILFE'] else None,
               'use': 'retrospective_data_preparation_not_frozen_fit_or_holdout'}
        annual_rows.append(row)
    write(out/'macro_observations_2021_2025.json', {'series_units':
          {'GDP/PCE/GPDI/GCE/NETEXP/PNFI': 'billion current USD annual flow', 'GDPC1': 'billion chained 2017 USD',
           'GDPDEF/PCEPILFE': 'source price index'},
          'vintage': 'local downloaded vintages; not origin-real-time', 'annualization': 'four quarterly SAAR / twelve monthly SAAR means',
          'rows': annual_rows, 'PNFI_scope': 'private nonresidential, not total fixed investment'})

    # Parse only the official 2022 column and matched sector codes. Do not choose a sector for the user.
    sectors = {'private_nonresidential': ('k1ntotl1es00','m1ntotl1es00'),
               'private_total': ('k1ptotl1es00','m1ptotl1es00'),
               'all_fixed_assets': ('k1ttotl1es00','m1ttotl1es00')}
    tables = {name: pd.read_excel(bea, sheet_name=name, header=None) for name in ['FAAt101-A','FAAt103-A']}
    anchors = []
    for sector, codes in sectors.items():
        extracted = []
        for table, code in zip(tables.values(), codes):
            headers = table.iloc[:, 0].astype(str).eq('Line')
            if headers.sum() != 1: raise ValueError('ambiguous BEA header')
            header = table.loc[headers].iloc[0]
            columns = [c for c,v in header.items() if str(v) in ('2022','2022.0')]
            rows = table[table.iloc[:,2].astype(str).str.lower().eq(code)]
            # BEA repeats total fixed assets in an addendum. Accept only identical values.
            if len(columns) != 1 or rows.empty or rows.iloc[:,columns[0]].nunique() != 1:
                raise ValueError('ambiguous BEA cell')
            extracted.append(float(rows.iloc[0,columns[0]]))
        k, depreciation = extracted
        anchors.append({'sector': sector, 'year': 2022, 'capital_yearend_million_usd': k,
             'annual_depreciation_million_usd': depreciation, 'capital_to_total_gdp_candidate': k/(series['GDP'][2022]*1000),
             'depreciation_over_yearend_capital_candidate': depreciation/k,
             'automatically_apply': False, 'issues': ['yearend stock is not automatically opening capital',
                 'sector differs from total GDP except all_fixed_assets; mapping must be approved',
                 'capital valuation and model initial finance must be checked']})
    write(out/'capital_candidates_2022.json', anchors)

    scf_data = pd.read_csv(scf)
    groups = scf_data.groupby('YY1').size()
    key_check = {'rows': len(scf_data), 'households': int(scf_data.YY1.nunique()),
                 'Y1_unique': bool(scf_data.Y1.is_unique), 'five_rows_per_household': bool(groups.eq(5).all()),
                 'implicate_digits': sorted(int(v) for v in scf_data.Y1.mod(10).unique()),
                 'raw_weight_sum': float(scf_data.WGT.sum()),
                 'weight_rule': 'preserve WGT exactly; do not divide by five again without source justification',
                 'fields_available': [c for c in ['YY1','Y1','WGT','INCOME','ASSET','NETWORTH','WAGEINC','EDUC','LF'] if c in scf_data],
                 'replicate_archive_members': zipfile.ZipFile(rw).namelist(),
                 'replicate_link_and_variance': 'not_yet_validated', 'existing_model_input_replaced': False}
    write(out/'scf_schema_check.json', key_check)
    panel = pd.read_csv(out/'inputs/psid/psid_reference_person_family_panel_2011_2023.csv.gz')
    write(out/'psid_schema_check.json', {'rows':len(panel), 'interview_years': sorted(int(x) for x in panel.interview_year.unique()),
         'income_hours_years': sorted(int(x) for x in panel.income_hours_year.dropna().unique()),
         'duplicate_person_wave': int(panel.duplicated(['permanent_person_id','interview_year']).sum()),
         'not_estimated_here': ['rho_e','sigma_e','tail_transition'],
         'caveats': ['deflate nominal earnings', 'reference-person changes and attrition', 'longitudinal weights',
                     'wave versus income year', 'survey missing codes and measurement error']})
    contract = {'schema_version':1, 'model_base_year':2022, 'data_bundle_available':True,
        'real_calibration_ready':False, 'auto_apply_parameters':False, 'fit_years':None, 'holdout_years':None,
        'macro_file':'macro_observations_2021_2025.json', 'capital_candidates_file':'capital_candidates_2022.json',
        'taskbook_status': {'real_gdp_growth':'observations_prepared', 'consumption_gdp':'observations_prepared_mapping_pending',
           'inflation':'missing_headline_PCEPI_do_not_substitute_core_or_CPI',
           'investment_gdp':'PNFI_candidate_only_sector_contract_pending',
           'alpha':'labor_share_file_available_units_and_sector_not_approved',
           'depreciation_and_KY':'BEA_candidates_prepared_not_applied',
           'SCF':'official_IDs_and_weight_archive_present_link_and_estimand_pending',
           'CEX':'raw_release_present_old_moments_quarantined', 'PSID':'longitudinal_panel_present_estimation_pending'},
        'cex_warning':'Legacy processor annualizes 4*TOTEX4CQ; reference-period coverage must be reconstructed with prior/current quarter fields. Do not calibrate consumption_share to old clipped moment.',
        'alpha_warning':'MPU4910141 source labels Percent but local values are around 0.6; resolve unit conventions and private-nonfarm scope before deriving alpha. PRS85006173 is an index, not a level share.',
        'protected_scope':'no old custody or sealed origin bundles opened; no simulator run; no original data overwritten'}
    write(out/'data_contract.json', contract)
    write(out/'manifest.json', {'workspace':str(workspace), 'files':sources, 'build_code_sha256':digest(__file__),
        'copied_file_count':len(sources), 'copied_bytes':sum(s['bytes'] for s in sources),
        'uses_runtime_absolute_paths':False, 'redistribution':'local integration only; survey licenses need review before publication'})
    checks = {'all_copies_match': all(digest(out/s['bundled_path'])==s['sha256'] for s in sources),
              'sources_unchanged': all(digest(workspace/s['source_relative_path'])==s['sha256'] for s in sources),
              'SCF_keys_valid':key_check['Y1_unique'] and key_check['five_rows_per_household'],
              'PSID_person_wave_unique': bool(not panel.duplicated(['permanent_person_id','interview_year']).any()),
              'headline_inflation_never_filled_with_proxy': all(r['headline_pce_inflation'] is None for r in annual_rows)}
    write(out/'verification.json', checks)
    print(json.dumps({'copied_files':len(sources),'bytes':sum(s['bytes'] for s in sources),'checks':checks}, default=bool))


if __name__ == '__main__': main()
