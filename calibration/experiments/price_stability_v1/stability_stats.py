"""Paired seed-cluster diagnostics; intervals are exploratory, not simultaneous."""
import numpy as np

def losses(candidates,target):
    x=np.asarray(candidates,dtype=float);y=np.asarray(target,dtype=float)
    if x.ndim!=3 or y.ndim!=2 or x.shape[2]!=y.shape[1] or not x.shape[1] or not y.shape[0]:
        raise ValueError('complete candidate and target arrays required')
    if not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError('nonfinite observation')
    return np.sqrt(np.mean((x.mean(axis=1)-y.mean(axis=0))**2,axis=1))

def choose(values,grid):
    v=np.asarray(values,dtype=float)
    if v.shape!=(len(grid),) or not np.isfinite(v).all():raise ValueError('invalid losses')
    return int(min(np.flatnonzero(v<=v.min()+1e-12),key=lambda i:(abs(grid[i]-.2),grid[i])))

def bootstrap_losses(candidates,target,repeats,seed):
    x=np.asarray(candidates);y=np.asarray(target);losses(x,y)
    rng=np.random.default_rng(seed);result=[]
    for _ in range(repeats):
        xi=rng.integers(0,x.shape[1],x.shape[1]);yi=rng.integers(0,len(y),len(y))
        result.append(losses(x[:,xi,:],y[yi,:]))
    return np.asarray(result)

def ambiguity_set(boot,chosen):
    return np.flatnonzero(np.quantile(boot-boot[:,[chosen]],.025,axis=0)<=0).tolist()

def rank_agreement(a,b):
    n=len(a)
    return float(sum(np.sign(a[i]-a[j])*np.sign(b[i]-b[j]) for i in range(n) for j in range(i))/(n*(n-1)/2))

def check_seed_roles(protocol):
    seen=set()
    for profile in protocol['profiles']:
        groups=list(profile['target_seeds'].values())+profile['search_seed_pools']+[profile['validation_seeds']]
        for group in groups:
            if not group or len(group)!=len(set(group)) or seen.intersection(group) or min(group)<=29:
                raise ValueError('seed roles overlap or use reserved seeds')
            seen.update(group)
    return len(seen)

def valid_rows(result,seeds,horizon):
    if not result.get('valid') or [p['seed'] for p in result['paths']]!=seeds:
        raise ValueError('invalid, missing or reordered requested seed paths')
    for path in result['paths']:
        if not path.get('valid') or [r['period'] for r in path['trajectory']]!=list(range(1,horizon+1)):
            raise ValueError('incomplete trajectory')
        for row in path['trajectory']:
            if row.get('sector_accounting',{}).get('status')!='PASS' or row.get('investment_diagnostic',{}).get('status')!='PASS':
                raise ValueError('sector or investment check failed')
