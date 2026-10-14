#==========================
# KT4 | Figure 5, severity class by fuel class
#==========================
#reads the table from 06_fuel_severity_chisq.py, so the bars match the test
#bars are row percentages
#run after 06: python3 08_fig_fuel_severity.py
#output: figures/fig_fuel_severity.png
import matplotlib.pyplot as plt
import pandas as pd

tab = pd.read_csv("tables/fuel_severity_crosstab.csv", index_col=0)
stats = pd.read_csv("tables/fuel_severity_chisq.csv").iloc[0]
LABEL = {"pasture": "Pasture", "gorse_broom": "Gorse / broom",
         "broadleaf_scrub": "Broadleaf scrub", "exotic_pine": "Exotic pine"}
COLOUR = {"Unburned": "#4C8C4A", "Low": "#E8D27A", "Moderate": "#E59A3C", "High": "#B83A26"}

pct = tab.div(tab.sum(axis=1), axis=0) * 100
pct = pct.iloc[::-1]                                   # first class at the top
fig, ax = plt.subplots(figsize=(9, 3.6))
left = pd.Series(0.0, index=pct.index)
for sev in COLOUR:
    ax.barh([LABEL[k] for k in pct.index], pct[sev], left=left, color=COLOUR[sev],
            edgecolor="white", height=0.65, label=sev)
    for k in pct.index:                                # label segments wide enough to read
        if pct.loc[k, sev] >= 6:
            ax.text(left[k] + pct.loc[k, sev] / 2, LABEL[k], f"{pct.loc[k, sev]:.0f}%", ha="center",
                    va="center", fontsize=9, color="white" if sev in ("Unburned", "High") else "black")
    left += pct[sev]
for k in pct.index:                                    # sample size at the end of each bar
    ax.text(101, LABEL[k], f"n = {int(tab.loc[k].sum())}", va="center", fontsize=9)
ax.set_xlim(0, 112)
ax.set_xlabel("Reference cells in each burn-severity class (%)")
ax.legend(ncol=4, loc="lower center", bbox_to_anchor=(0.45, 1.0), frameon=False, fontsize=9)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
p_txt = "p < 0.001" if stats.p < 0.001 else f"p = {stats.p:.3f}"
ax.text(0, -0.32, f"χ² = {stats.chi2:.1f}, df = {int(stats.df)}, {p_txt}, Cramér's V = {stats.cramers_v:.2f}",
        transform=ax.transAxes, fontsize=9)
plt.tight_layout()
plt.savefig("figures/fig_fuel_severity.png", dpi=200, bbox_inches="tight")
print("saved figures/fig_fuel_severity.png")
