"""Figures 4.3 (scatterplots) and 4.4 (regression) for R4, using dNBR_lc"""
import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from A3_R4_regression_io import read_stack
from pathlib import Path
DATA = Path(__file__).resolve().parents[2] / "data"
plt.rcParams.update({"font.size":9,"axes.spines.top":False,"axes.spines.right":False})
CL=["pasture","gorse_broom","broadleaf_scrub","exotic_pine"]; LAB=["Pasture","Gorse/broom","Broadleaf","Exotic pine"]
COL=dict(zip(CL,["#C9A227","#8C6D1F","#3E6B5A","#4A6A8A"])); T=float(np.load("T_lc.npy"))
a,n=read_stack(str(DATA / "PortHills2017_stack.tif")); a[a==-9999]=np.nan; S={k:a[:,:,i] for i,k in enumerate(n)}; S["y"]=np.load("dNBR_lc.npy")
d=pd.read_csv(DATA / "A3_reference_cells.csv"); d=d[d.use_A3==True].copy(); d["y"]=S["y"][d.array_row,d.array_col]; d=d[np.isfinite(d.y)].reset_index(drop=True)
# ---- Fig 4.3
f,ax=plt.subplots(1,3,figsize=(11.5,3.7))
for cl,l in zip(CL,LAB):
    m=d.FuelClass==cl; ax[0].scatter(d.loc[m,"pre_B4"],d.loc[m,"pre_B5"],s=14,c=COL[cl],label=l,alpha=.85,edgecolor="none")
    for k,x in ((1,"pre_B5"),(2,"NDMI")):
        ax[k].scatter(d.loc[m,x],d.loc[m,"y"],s=12,c=COL[cl],alpha=.75,edgecolor="none")
        if m.sum()>10:
            b=np.polyfit(d.loc[m,x],d.loc[m,"y"],1); xx=np.linspace(d.loc[m,x].min(),d.loc[m,x].max(),5); ax[k].plot(xx,np.polyval(b,xx),color=COL[cl],lw=1.2)
for k,x in ((1,"pre_B5"),(2,"NDMI")):
    rho,p=stats.spearmanr(d[x],d.y); b=np.polyfit(d[x],d.y,1); xx=np.linspace(d[x].min(),d[x].max(),5)
    ax[k].plot(xx,np.polyval(b,xx),"k--",lw=1.3,label="all cells"); ax[k].axhline(T,color="grey",ls=":",lw=.8)
    ax[k].text(.03,.97,f"pooled ρ = {rho:.2f} (p = {p:.3f})",transform=ax[k].transAxes,va="top",fontsize=8.5)
ax[0].set(xlabel="Pre-fire red reflectance (B4)",ylabel="Pre-fire NIR reflectance (B5)",title="(a) Fuel classes in red–NIR space")
ax[0].legend(frameon=False,fontsize=8)
ax[1].set(xlabel="Pre-fire NIR reflectance (B5)",ylabel="dNBR_lc",title="(b) NIR vs burn severity"); ax[1].legend(frameon=False,fontsize=8,loc="lower right")
ax[2].set(xlabel="Pre-fire NDMI",ylabel="dNBR_lc",title="(c) NDMI vs burn severity")
f.tight_layout(); f.savefig("figures/fig_scatter.png",dpi=250); plt.close(f)
# ---- Fig 4.4
s=pd.read_csv("tables/reg_summary.csv").set_index("model"); fc=pd.read_csv("tables/reg_fuel_coefficients.csv")
e=np.load("L2_resid.npy")
f,ax=plt.subplots(1,3,figsize=(11.5,3.6),gridspec_kw=dict(width_ratios=[1.2,1.2,.9]))
labs=["Image only","Fuel class only","Image + fuel class"]; x=np.arange(3)
ref=[s.loc["L2","R2_image"],s.loc["L2","R2_fuel"],s.loc["L2","R2_both"]]; lcd=[s.loc["LCDB","test_image"],s.loc["LCDB","test_fuel"],s.loc["LCDB","test_both"]]
ax[0].bar(x-.19,ref,.38,color="#3E6B5A",label=f"Reference cells, FuelClass (R², n = {int(s.loc['L2', 'n']):,})"); ax[0].bar(x+.19,lcd,.38,color="#4A6A8A",label=f"All cells, LCDB 2012 (test R², n = {int(s.loc['LCDB', 'n_test']):,})")
for i in range(3): ax[0].text(i-.19,ref[i]+.008,f"{ref[i]:.2f}",ha="center",fontsize=8); ax[0].text(i+.19,lcd[i]+.008,f"{lcd[i]:.2f}",ha="center",fontsize=8)
ax[0].set_xticks(x); ax[0].set_xticklabels(labs,fontsize=8); ax[0].set_ylim(0,.42); ax[0].set_ylabel("R²"); ax[0].legend(frameon=False,fontsize=7.2,loc="upper left")
ax[0].set_title("(a) Variance explained")
nm={"gorse_broom":"Gorse/broom","broadleaf_scrub":"Broadleaf","exotic_pine":"Exotic pine"}
for i,cl in enumerate(["gorse_broom","broadleaf_scrub","exotic_pine"]):
    for dd,off_,col in (("Reference cells",-.12,"#3E6B5A"),("LCDB all cells",.12,"#4A6A8A")):
        rr=fc[(fc.data==dd)&(fc.fuel==cl)].iloc[0]; ax[1].errorbar(rr.coef,i+off_,xerr=[[rr.coef-rr.lo],[rr.hi-rr.coef]],fmt="o",color=col,capsize=3,label=dd if i==0 else None)
ax[1].axvline(0,color="grey"); ax[1].set_yticks(range(3)); ax[1].set_yticklabels([nm[k] for k in ["gorse_broom","broadleaf_scrub","exotic_pine"]]); ax[1].invert_yaxis()
ax[1].set_xlabel("dNBR_lc difference from pasture (±95% CI)\ncontrolling for spectra and terrain"); ax[1].legend(frameon=False,fontsize=7.5,loc="upper left",bbox_to_anchor=(0,-.32),ncol=2); ax[1].set_title("(b) Fuel-class effects")
(osm,osr),(sl,ic,_)=stats.probplot(e); ax[2].scatter(osm,osr,s=8,c="#22303A"); ax[2].plot(osm,sl*osm+ic,color="#9E3B2A")
ax[2].set(xlabel="Theoretical quantiles",ylabel="Residuals",title="(c) Residual Q–Q, reference model")
f.tight_layout(); f.savefig("figures/fig_regression.png",dpi=250); plt.close(f)
print("ok")
