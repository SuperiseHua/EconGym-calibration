"""New immutable current-vintage observations, never automatic calibration inputs."""
import argparse
import concurrent.futures
import csv
import datetime
import hashlib
import io
import json
from pathlib import Path
import urllib.request


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True)
    a = p.parse_args()
    out = Path(a.output); out.mkdir(parents=True, exist_ok=False)
    series = {'PCEPI': (12, 'index_2017_100'), 'FPI': (4, 'billion_USD_SAAR'),
              'PNFI': (4, 'billion_USD_SAAR'), 'GDP': (4, 'billion_USD_SAAR'),
              'GDPC1': (4, 'billion_chained_2017_USD_SAAR'),
              'PCE': (12, 'billion_USD_SAAR'), 'GCE': (4, 'billion_USD_SAAR'),
              'NETEXP': (4, 'billion_USD_SAAR'), 'GPDI': (4, 'billion_USD_SAAR')}
    def fetch(item):
        name, (freq, unit) = item
        url = f'https://fred.stlouisfed.org/graph/fredgraph.csv?id={name}&cosd=2020-01-01&coed=2025-12-31'
        stamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
        try:
            with urllib.request.urlopen(url, timeout=40) as response:
                raw = response.read(2_000_001)
            if len(raw)>2_000_000: raise ValueError('unexpected response size')
            reader = csv.DictReader(io.StringIO(raw.decode('utf-8-sig')))
            if name not in (reader.fieldnames or []): raise ValueError('missing series column')
            datecol = reader.fieldnames[0]
            rows = [(r[datecol], float(r[name])) for r in reader if r[name] not in ('', '.')]
            if not rows: raise ValueError('empty series')
            (out/(name+'.csv')).write_bytes(raw)
            annual = {}
            for year in range(2020,2026):
                vals = [v for d,v in rows if d.startswith(str(year))]
                annual[str(year)] = {'observations':len(vals),'expected':freq,
                    'mean':sum(vals)/len(vals) if len(vals)==freq else None}
            return name, {'status':'DOWNLOADED', 'url':url, 'metadata_url':f'https://fred.stlouisfed.org/series/{name}',
                'retrieved_utc':stamp,'sha256':hashlib.sha256(raw).hexdigest(), 'bytes':len(raw),
                'unit':unit, 'annual':annual,
                'december':{d[:4]:v for d,v in rows if d[5:7]=='12'}}
        except Exception as e:
            return name, {'status':'UNAVAILABLE','url':url,'retrieved_utc':stamp,'error':str(e)}
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        data = dict(pool.map(fetch,series.items()))
    def value(name, year):
        return data[name].get('annual',{}).get(str(year),{}).get('mean')
    def ratio(x,y): return x/y if x is not None and y not in (None,0) else None
    derived=[]
    for y in range(2021,2026):
        g=value('GDP',y)
        price_ratio=ratio(value('PCEPI',y), value('PCEPI',y-1))
        dec_ratio=ratio(data['PCEPI'].get('december',{}).get(str(y)),data['PCEPI'].get('december',{}).get(str(y-1)))
        growth=ratio(value('GDPC1',y),value('GDPC1',y-1))
        parts=[value(k,y) for k in ('PCE','GPDI','GCE','NETEXP')]
        derived.append({'year':y,'GDP_billion_USD':g,
            'headline_PCE_annual_average_inflation':None if price_ratio is None else price_ratio-1,
            'headline_PCE_December_over_December':None if dec_ratio is None else dec_ratio-1,
            'real_GDP_annual_growth':None if growth is None else growth-1,
            'PCE_over_GDP':ratio(value('PCE',y),g),'private_fixed_investment_over_GDP':ratio(value('FPI',y),g),
            'private_nonresidential_fixed_investment_over_GDP':ratio(value('PNFI',y),g),
            'GPDI_over_GDP':ratio(value('GPDI',y),g),
            'published_expenditure_identity_residual_billion':g-sum(parts) if g is not None and all(x is not None for x in parts) else None})
    result={'schema_version':1,'purpose':'current_vintage_mapping_audit_only',
        'auto_apply_parameters':False,'real_calibration_ready':False,'point_in_time_forecast_data':False,
        'series':data,'derived':derived,
        'cautions':['Same retrieval batch, not guaranteed same underlying release vintage.',
                    'FPI includes residential; PNFI excludes residential; GPDI also includes inventories.',
                    'Annual average inflation and December-over-December are distinct conventions.',
                    'Never replace original collected_v1 or silently alter model foundations.']}
    (out/'manifest.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'series':{k:v['status'] for k,v in data.items()},'derived':derived},indent=2))


if __name__=='__main__': main()
