"""
Step 3 (R4 parts 3-4): regression with dNBR_lc
Layer 1: all burned-over core cells, dNBR_lc ~ pre-fire spectra + terrain (VIF <= 7.5, 120 m thinning, offset test cells)
Layer 2: reference cells, + FuelClass dummies (field variable); nested F test; robustness by 300 m thinning (200 draws)
Check  : all cells with LCDB 2012 fuel class, same design as layer 1
"""
import numpy as np, pandas as pd
from scipy import stats
from scipy.spatial.distance import cdist
from A3_R4_regression_io import read_stack
from pathlib import Path
DATA = Path(__file__).resolve().parents[2] / "data"
a,n=read_stack(str(DATA / "PortHills2017_stack.tif")); a[a==-9999]=np.nan; S={k:a[:,:,i] for i,k in enumerate(n)}
S["y"]=np.load("dNBR_lc.npy"); lc=np.load("lcdb_stackgrid.npy")
r=S["grid_row"]-0.5; c=S["grid_col"]-0.5; core=(S["inside_aoi"]==1)&np.isfinite(S["y"])
CAND=["pre_B2","pre_B3","pre_B4","pre_B5","pre_B6","pre_B7","NDVI","NDMI","BSI","elev","slope","northness"]
def ols(X,y):
    Z=np.column_stack([np.ones(len(y)),X]); b,*_=np.linalg.lstsq(Z,y,rcond=None); e=y-Z@b; rss=e@e; nn,p=Z.shape
    se=np.sqrt(np.diag(rss/(nn-p)*np.linalg.inv(Z.T@Z))); tss=((y-y.mean())**2).sum()
    return dict(b=b,se=se,e=e,rss=rss,R2=1-rss/tss,adj=1-(rss/(nn-p))/(tss/(nn-1)),n=nn,p=p)
def vif(X): return np.array([1/(1-ols(np.delete(X,j,1),X[:,j])["R2"]) for j in range(X.shape[1])])
def moran_grid(z,rr,cc,step):
    idx={(i,j):k for k,(i,j) in enumerate(zip(rr,cc))}; z=z-z.mean(); num=W=0
    for k,(i,j) in enumerate(zip(rr,cc)):
        for di in(-step,0,step):
            for dj in(-step,0,step):
                if di==dj==0: continue
                m=idx.get((i+di,j+dj))
                if m is not None: num+=z[k]*z[m]; W+=1
    return len(z)/W*num/(z@z)
def test_r2(cols,tr,te,extra=None):
    X=lambda s:np.column_stack([S[k][s] for k in cols]); m=ols(X(tr),S["y"][tr])
    Z=np.column_stack([np.ones(te.sum()),X(te)]); p=Z@m["b"]; yt=S["y"][te]
    return m,1-((yt-p)**2).sum()/((yt-yt.mean())**2).sum(),np.sqrt(((yt-p)**2).mean())
out={}
# ---------- Layer 1
tr=core&(r%4==3)&(c%4==2); te=core&(r%4==1)&(c%4==0)
X=np.column_stack([S[k][tr] for k in CAND]); keep=list(CAND)
while True:
    v=vif(X); j=int(np.argmax(v))
    if v[j]<=7.5: break
    X=np.delete(X,j,1); keep.pop(j)
m1,t1,rm1=test_r2(keep,tr,te); v1=vif(np.column_stack([S[k][tr] for k in keep]))
sd_y=S["y"][tr].std()
L1=pd.DataFrame([dict(variable=k,coef=m1["b"][i+1],std_beta=m1["b"][i+1]*S[k][tr].std()/sd_y,SE=m1["se"][i+1],
    p=2*stats.t.sf(abs(m1["b"][i+1]/m1["se"][i+1]),m1["n"]-m1["p"]),VIF=v1[i]) for i,k in enumerate(keep)])
mor={s*30:round(moran_grid(ols(np.column_stack([S[k][core&(r%s==3%s)&(c%s==2%s)] for k in keep]),S["y"][core&(r%s==3%s)&(c%s==2%s)])["e"],
      r[core&(r%s==3%s)&(c%s==2%s)],c[core&(r%s==3%s)&(c%s==2%s)],s),3) for s in (4,8)}
out["L1"]=dict(n=m1["n"],R2=m1["R2"],adj=m1["adj"],n_test=int(te.sum()),test_R2=t1,RMSE=rm1,moran120=mor[120],moran240=mor[240],vars=",".join(keep))
L1.round(4).to_csv("tables/reg_L1_coefficients.csv",index=False)
# ---------- LCDB all-cell check (fuel type for every cell)
f4=core&np.isin(lc,[40,51,54,71]); trf=f4&(r%4==3)&(c%4==2); tef=f4&(r%4==1)&(c%4==0)
for k in (51,54,71): S[f"L{k}"]=(lc==k)*1.
LD=["L51","L54","L71"]
mi,ti,_=test_r2(keep,trf,tef); mf,tf,_=test_r2(LD,trf,tef); mb,tb,_=test_r2(keep+LD,trf,tef)
F=((mi["rss"]-mb["rss"])/3)/(mb["rss"]/(mb["n"]-mb["p"]))
out["LCDB"]=dict(n=mb["n"],n_test=int(tef.sum()),R2_image=mi["R2"],R2_fuel=mf["R2"],R2_both=mb["R2"],test_image=ti,test_fuel=tf,test_both=tb,
                 dR2=mb["R2"]-mi["R2"],F=F,p=stats.f.sf(F,3,mb["n"]-mb["p"]))
lcoef=[dict(data="LCDB all cells",fuel=nm,coef=mb["b"][len(keep)+1+i],SE=mb["se"][len(keep)+1+i]) for i,nm in enumerate(["gorse_broom","broadleaf_scrub","exotic_pine"])]
# ---------- Layer 2 reference cells
d=pd.read_csv(DATA / "A3_reference_cells.csv"); d=d[d.use_A3==True].copy(); d["y"]=S["y"][d.array_row,d.array_col]; d=d[np.isfinite(d.y)].reset_index(drop=True)
CL=["gorse_broom","broadleaf_scrub","exotic_pine"]; D=[]
for k in CL: d["D_"+k]=(d.FuelClass==k)*1.; D.append("D_"+k)
def nested(dd):
    y=dd.y.values; a0=ols(dd[keep].values,y); a1=ols(dd[keep+D].values,y); af=ols(dd[D].values,y)
    F=((a0["rss"]-a1["rss"])/3)/(a1["rss"]/(a1["n"]-a1["p"])); return a0,a1,af,F,stats.f.sf(F,3,a1["n"]-a1["p"])
a0,a1,af,F2,p2=nested(d)
out["L2"]=dict(n=a1["n"],R2_image=a0["R2"],adj_image=a0["adj"],R2_both=a1["R2"],adj_both=a1["adj"],R2_fuel=af["R2"],dR2=a1["R2"]-a0["R2"],F=F2,df2=a1["n"]-a1["p"],p=p2,
               resid_skew=stats.skew(a1["e"]),resid_shapiro_p=stats.shapiro(a1["e"]).pvalue)
lcoef+= [dict(data="Reference cells",fuel=k,coef=a1["b"][len(keep)+1+i],SE=a1["se"][len(keep)+1+i]) for i,k in enumerate(CL)]
nd=a1["b"][keep.index("NDMI")+1]; out["L2"]["NDMI_beta_with_fuel"]=nd*d.NDMI.std()/d.y.std(); out["L2"]["NDMI_p_with_fuel"]=2*stats.t.sf(abs(nd/a1["se"][keep.index("NDMI")+1]),a1["n"]-a1["p"])
xy=d[["NZTM_X","NZTM_Y"]].values; Dm=cdist(xy,xy); z=a1["e"]-a1["e"].mean(); Wm=((Dm>0)&(Dm<=300)).astype(float)
out["L2"]["moran300"]=len(z)/Wm.sum()*(z@Wm@z)/(z@z)
dr=[]
for s in range(200):
    g=np.random.default_rng(s); kp=[]
    for i in g.permutation(len(d)):
        if all(Dm[i,j]>=300 for j in kp): kp.append(i)
    b0,b1,_,_,ps=nested(d.iloc[kp]); dr.append((len(kp),b1["R2"]-b0["R2"],ps))
dr=np.array(dr); out["L2"].update(thin_n=np.median(dr[:,0]),thin_dR2_med=np.median(dr[:,1]),thin_dR2_q1=np.percentile(dr[:,1],25),thin_dR2_q3=np.percentile(dr[:,1],75),thin_share_sig=(dr[:,2]<.05).mean())
pd.DataFrame(lcoef).assign(lo=lambda x:x.coef-1.96*x.SE,hi=lambda x:x.coef+1.96*x.SE).round(1).to_csv("tables/reg_fuel_coefficients.csv",index=False)
summ=pd.DataFrame([dict(model=k,**v) for k,v in out.items()]); summ.round(4).to_csv("tables/reg_summary.csv",index=False)
np.save("L2_resid.npy",a1["e"]); np.save("L2_fitted.npy",d.y.values-a1["e"])
for k,v in out.items(): print(k,{kk:(round(vv,3) if isinstance(vv,float) else vv) for kk,vv in v.items()})
print(L1.round(3).to_string()); print(pd.DataFrame(lcoef).round(1).to_string())
