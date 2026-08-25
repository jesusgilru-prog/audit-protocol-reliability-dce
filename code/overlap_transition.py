"""Targeted batch in the low-overlap transition region.

The fine sweep (overlap_fine_sweep.py) has few scenarios with minimum fold
overlap in 0.005-0.010, which is exactly where the companion corpus sits,
so this script over-samples that window to put the manuscript's claim about
that corpus on an adequate sample rather than on ~46 scenarios.

Writes results/overlap_transition.json.
"""
import sys, json
import numpy as np, pandas as pd
sys.path.insert(0,'/home/jesus/paper3_universal_scaling/code')
from synth_audit_protocol_v3 import (gen_scenario_overlap, min_fold_overlap,
                                     quantile_strata, perm_pvalue, ALPHA, wilson)
rng=np.random.default_rng(20260827)
rows=[]; tries=0
while len([r for r in rows if 0.003<r['mf']<=0.020])<520 and tries<20000:
    tries+=1
    K=int(rng.integers(3,9)); n=int(rng.integers(15,60))
    sigma=float(rng.uniform(0.05,0.6)); het=float(rng.uniform(0.5,1.5))
    isu=bool(rng.integers(0,2)); ov=float(rng.uniform(0.005,0.045))
    df=gen_scenario_overlap(K,n,sigma,het,isu,ov,rng)
    mf=min_fold_overlap(df)
    if not (0.0<mf<=0.030): continue
    X=np.column_stack([np.ones(len(df)),df.log_re.values,df.log_pg.values])
    y=df.log_cp.values; g=df.facility.values
    r2,pn=perm_pvalue(X,y,g,rng,nperm=99)
    st=quantile_strata(df.log_re.values,nbins=max(2,K))
    _,ps=perm_pvalue(X,y,g,rng,nperm=99,strata=st)
    rows.append(dict(mf=mf,isu=isu,
        thr=(bool(np.isfinite(r2) and r2>0))==isu,
        perm=(bool(np.isfinite(pn) and pn>ALPHA))==isu,
        strat=(bool(np.isfinite(ps) and ps>ALPHA))==isu))
d=pd.DataFrame(rows)
d['bin']=pd.cut(d.mf,bins=[0,0.003,0.006,0.010,0.020,0.030],
                labels=['0-0.003','0.003-0.006','0.006-0.010','0.010-0.020','0.020-0.030'],
                include_lowest=True)
u,c=d[d.isu],d[~d.isu]
print('%14s %5s %5s %10s %20s %12s'%('min-fold','n_u','n_c','strat FA','strat power','power CI95'))
out=[]
for b in d.bin.cat.categories:
    su,sc=u[u.bin==b],c[c.bin==b]
    if len(su)<5 or len(sc)<5: continue
    fa=1-su.strat.mean(); pw=sc.strat.mean(); lo,hi=wilson(int(sc.strat.sum()),len(sc))
    print('%14s %5d %5d %10.3f %20.3f   [%.3f, %.3f]'%(b,len(su),len(sc),fa,pw,lo,hi))
    out.append(dict(bin=str(b),n_universal=int(len(su)),n_confounded=int(len(sc)),
                    stratified_false_alarm=float(fa),stratified_power=float(pw),
                    stratified_power_ci95=[lo,hi],
                    naive_false_alarm=float(1-su.perm.mean()),
                    threshold_power=float(sc.thr.mean())))
json.dump({'rng_seed':20260827,'nperm':99,'alpha':ALPHA,
           'binning_statistic':'obs_min_fold_overlap','bins':out},
          open('/home/jesus/paper3_universal_scaling/results/overlap_transition.json','w'),indent=2)
print('\nguardado results/overlap_transition.json  (n total %d)'%len(d))
