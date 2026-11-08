"""extended_experiments.py -- reproduces the stability, conditional-coverage, learning-curve, leakage and latency experiments (paper Sections 7.6-7.13).
Run after run_pipeline.py:  python extended_experiments.py   (about 3-5 minutes). Writes outputs/extended_results.json."""
import sys, json, time, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0,'.')
import numpy as np, pandas as pd, xgboost as xgb
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
import config, preprocess, common
from valuation import ValuationModel, feature_group
R={}
def m(a,p): return dict(r2=r2_score(a,p), mape=float(np.mean(abs(a-p)/a)*100), mae=float(mean_absolute_error(a,p)))
P=dict(n_estimators=3000,max_depth=6,learning_rate=.03,subsample=.8,colsample_bytree=.8,reg_alpha=.5,reg_lambda=2.0)
def fit_xgb(Xtr,ytr,Xva,yva,**kw):
    mod=xgb.XGBRegressor(early_stopping_rounds=50,random_state=42,n_jobs=-1,**{**P,**kw}); mod.fit(Xtr,ytr,eval_set=[(Xva,yva)],verbose=False); return mod
def fit_ridge(Xtr,ytr):
    s=StandardScaler().fit(Xtr); r=Ridge(alpha=10).fit(s.transform(Xtr),ytr); return lambda X:r.predict(s.transform(X))
def cq(res,a): n=len(res);k=min(int(np.ceil((n+1)*(1-a))),n);return np.sort(res)[k-1]
raw=pd.read_csv(config.RAW_PATH)

# ---- 1) five independent seeds, fully leak-free per seed
orig=preprocess.assign_splits; rows=[]
for seed in range(1,6):
    preprocess.assign_splits=lambda n,seed_=seed:orig(n,seed_)
    df,_=preprocess.preprocess(raw); f=common.feature_columns(df)
    S={k:df[df.split==k] for k in ('train','valid','calib','test')}
    mx=fit_xgb(S['train'][f],S['train'].price,S['valid'][f],S['valid'].price)
    pr=fit_ridge(S['train'][f],S['train'].price)
    a=np.expm1(S['test'].price.values); px=np.expm1(mx.predict(S['test'][f])); pri=np.expm1(pr(S['test'][f]))
    q=cq(np.abs(S['calib'].price.values-mx.predict(S['calib'][f])),.1)
    cov=float(np.mean(np.abs(S['test'].price.values-mx.predict(S['test'][f]))<=q))
    rows.append(dict(seed=seed,**{'xgb_'+k:v for k,v in m(a,px).items()},**{'ridge_'+k:v for k,v in m(a,pri).items()},cov90=cov))
preprocess.assign_splits=orig
d=pd.DataFrame(rows); R['seeds']=dict(mean=d.mean().to_dict(),std=d.std().to_dict(),rows=rows)
print(d.round(4).to_string()); print(d.mean().round(4).to_dict(), d.std().round(4).to_dict())

# ---- production split objects
feats,sp=common.load_splits(); Xtr,ytr=sp['train'];Xva,yva=sp['valid'];Xca,yca=sp['calib'];Xte,yte=sp['test']
vm=ValuationModel(); mx=vm.model; pt=mx.predict(Xte[vm.features]); a=np.expm1(yte.values); p=np.expm1(pt)
rid=fit_ridge(Xtr,ytr); pr=rid(Xte)

# ---- 2) bootstrap CIs
rng=np.random.RandomState(0); bs=[]
for _ in range(1000):
    i=rng.randint(0,len(a),len(a)); bs.append((m(a[i],p[i]),m(a[i],np.expm1(pr)[i])))
for tag,j in (('xgb',0),('ridge',1)):
    R['boot_'+tag]={k:[float(np.percentile([b[j][k] for b in bs],2.5)),float(np.percentile([b[j][k] for b in bs],97.5))] for k in ('r2','mape','mae')}
print('boot',R['boot_xgb'],R['boot_ridge'])

# ---- 3) blend
bl=np.expm1((pt+pr)/2); R['blend']=m(a,bl); R['xgb']=m(a,p); R['ridge']=m(a,np.expm1(pr)); print('blend',R['blend'])

# ---- 4) conditional coverage
q90=vm.conformal['global']['90']; hit=np.abs(yte.values-pt)<=q90
T=Xte.copy(); T['hit']=hit
T['ptype']=T[[c for c in T if c.startswith('property_type_')]].idxmax(axis=1).str.replace('property_type_','')
T['tier']=np.where(T.is_metro==1,'Metro','Other'); T['band']=pd.qcut(a,3,labels=['low','mid','high'])
R['cov_cond']={k:T.groupby(k).hit.agg(['mean','count']).round(3).to_dict('index') for k in ('ptype','tier','band')}; print(R['cov_cond'])
# subsample sensitivity of calibration set size
res_c=np.abs(yca.values-mx.predict(Xca[vm.features])); sens={}
for n in (100,200,500,1176):
    cv=[]
    for _ in range(300):
        s=rng.choice(len(res_c),n,replace=False); cv.append(np.mean(np.abs(yte.values-pt)<=cq(res_c[s],.1)))
    sens[n]=[float(np.mean(cv)),float(np.std(cv))]
R['calib_size']=sens; print(sens)

# ---- 5) residual shape
rl=yte.values-pt; from scipy import stats
R['resid']=dict(mean=float(rl.mean()),std=float(rl.std()),skew=float(stats.skew(rl)),kurt=float(stats.kurtosis(rl)),
  het_corr=float(np.corrcoef(np.abs(rl),pt)[0,1]))
print(R['resid'])

# ---- 6) global SHAP importance by group
c=mx.get_booster().predict(xgb.DMatrix(Xte[vm.features]),pred_contribs=True)[:,:-1]
g=pd.Series(np.abs(c).mean(0),index=vm.features).groupby(lambda n:feature_group(n)).sum(); R['shap_groups']=(g/g.sum()*100).round(2).sort_values(ascending=False).to_dict(); print(R['shap_groups'])

# ---- 7) learning curve
lc={}
for frac in (.1,.25,.5,1.0):
    idx=np.random.RandomState(1).choice(len(Xtr),int(len(Xtr)*frac),replace=False)
    mm=fit_xgb(Xtr.iloc[idx],ytr.iloc[idx],Xva,yva); rr=fit_ridge(Xtr.iloc[idx],ytr.iloc[idx])
    lc[frac]=dict(n=len(idx),xgb_mape=m(a,np.expm1(mm.predict(Xte)))['mape'],ridge_mape=m(a,np.expm1(rr(Xte)))['mape'])
R['learning_curve']=lc; print(lc)

# ---- 8) leakage experiments
fr=raw[(raw.market_price>=raw.market_price.quantile(.01))&(raw.market_price<=raw.market_price.quantile(.99))].reset_index(drop=True)
full=pd.read_csv(config.CLEAN_PATH); assert np.allclose(np.log1p(fr.market_price.values),full.price.values)
L=full.copy(); L['circle_rate_value']=fr.circle_rate_value.values; L['govt_valuation']=fr.govt_valuation.values
Lf=feats+['circle_rate_value','govt_valuation']; Ltr,Lva,Lte=[L[L.split==k] for k in ('train','valid','test')]
ml=fit_xgb(Ltr[Lf],Ltr.price,Lva[Lf],Lva.price); pl=np.expm1(ml.predict(Lte[Lf]))
imp=pd.Series(ml.get_booster().get_score(importance_type='gain')); share=float(imp[['circle_rate_value','govt_valuation']].sum()/imp.sum()*100)
R['leak_circle']=dict(**m(a,pl),importance_share_pct=share); print('circle leak',R['leak_circle'])
# target-encoding leakage: encode with ALL rows
fr['_c']=fr.city; cm=fr.groupby('city').market_price.mean(); sm=fr.groupby('state').market_price.mean()
E=full.copy(); E['city_encoded']=fr.city.map(cm).values; E['state_encoded']=fr.state.map(sm).values; E['log_city_encoded']=np.log(E.city_encoded)
Etr,Eva,Ete=[E[E.split==k] for k in ('train','valid','test')]
me=fit_xgb(Etr[feats],Etr.price,Eva[feats],Eva.price); R['leak_te']=m(a,np.expm1(me.predict(Ete[feats]))); print('TE leak',R['leak_te'])
# early stopping on test
mt=fit_xgb(Xtr,ytr,Xte,yte); R['es_on_test']=dict(**m(a,np.expm1(mt.predict(Xte))),best_iter=int(mt.best_iteration)); print('ES on test',R['es_on_test'])

# ---- 9) latency
t=time.time(); inp=dict(city='Delhi',state='Delhi',built_up_area_sqft=1200)
for _ in range(50): vm.predict(inp)
R['lat_predict_ms']=(time.time()-t)/50*1000; t=time.time()
for _ in range(20): vm.explain(inp)
R['lat_explain_ms']=(time.time()-t)/20*1000; print(R['lat_predict_ms'],R['lat_explain_ms'])
json.dump(R,open(__import__('os').path.join(config.OUTPUTS_DIR,'extended_results.json'),'w'),indent=1,default=float)
