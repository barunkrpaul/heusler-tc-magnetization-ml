"""Stage 9: per-family breakdown figure (Fig10)."""
import sys, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "scripts")
from common import TARGETS

BLUE, ORANGE, MUTED = "#2a78d6", "#eb6834", "#898781"
INK, SEC, GRID, SURFACE = "#0b0b0b", "#52514e", "#e1e0d9", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "font.size": 9,
    "axes.edgecolor": "#c3c2b7", "axes.linewidth": 0.8,
    "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": SEC, "ytick.color": SEC,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "legend.frameon": False,
})

comp = pd.read_csv("results/pooled_vs_family.csv")
SHORT = {"curie_temperature_K": "Curie temperature",
         "formation_energy_eV_atom": "Formation energy",
         "hull_distance_eV_atom": "Hull distance",
         "total_magnetization_muB_fu": "Magnetization"}
UNIT = {"curie_temperature_K": "K", "formation_energy_eV_atom": "eV/at.",
        "hull_distance_eV_atom": "eV/at.", "total_magnetization_muB_fu": "$\\mu_B$"}
TYPES = ["Full", "Half", "Inverse", "Quaternary"]
CLIP = -0.30

fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.8))
for ax, target in zip(axes.ravel(), TARGETS):
    sub = comp[comp.target == target].set_index("family").loc[TYPES]
    x = np.arange(len(TYPES)); w = 0.36
    v1 = np.clip(sub["pooled_cv_r2"].values, CLIP, None)
    v2 = np.clip(sub["family_cv_r2"].values, CLIP, None)
    ax.bar(x - w / 2, v1, w - 0.02, color=BLUE, label="Pooled model (within-family score)")
    ax.bar(x + w / 2, v2, w - 0.02, color=ORANGE, label="Family-specific model")
    # hold-out scores as diamond markers
    ax.scatter(x - w / 2, np.clip(sub["pooled_holdout_r2"], CLIP, None), marker="D", s=22,
               color="#104281", zorder=3, label="Hold-out (pooled)")
    ax.scatter(x + w / 2, np.clip(sub["family_holdout_r2"], CLIP, None), marker="D", s=22,
               color="#8a3312", zorder=3, label="Hold-out (family)")
    # MAE annotation under each family
    for xi, t in zip(x, TYPES):
        ax.text(xi, CLIP - 0.14,
                f"MAE {sub.loc[t,'pooled_cv_mae']:.3g}/{sub.loc[t,'family_cv_mae']:.3g} {UNIT[target]}",
                ha="center", fontsize={"curie_temperature_K": 6.8}.get(target, 6.4), color=SEC)
    ax.axhline(0, color="#c3c2b7", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(TYPES)
    ax.set_ylabel("Within-family $R^2$ (CV)")
    ax.set_ylim(CLIP - 0.22, 1.0)
    ax.set_title(SHORT[target], fontsize=9)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=4, fontsize=7.5, bbox_to_anchor=(0.5, -0.005))
fig.suptitle("Per-family breakdown: pooled vs family-specific models "
             f"(bars clipped at {CLIP}; MAE shown pooled/family)", fontsize=10)
fig.tight_layout(rect=[0, 0.045, 1, 0.97])
fig.savefig("figures/Fig10_per_family.pdf", bbox_inches="tight")
fig.savefig("figures/Fig10_per_family.png", dpi=200, bbox_inches="tight")
print("Fig10 written")
