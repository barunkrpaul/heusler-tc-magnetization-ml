"""Stage 7: publication figures mirroring Figs. 1-9 of the previous manuscript."""
import sys, json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
sys.path.insert(0, "scripts")
from common import load_prepared, TARGETS, FAMILY_BLOCKS

# ---- palette (validated reference palette, light mode) ----------------
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
VIOLET, RED = "#4a3aa7", "#e34948"
INK, SEC, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
TYPE_COLOR = {"Full": BLUE, "Half": ORANGE, "Inverse": AQUA, "Quaternary": YELLOW}
FAMILY_COLOR = {"Valence electrons": AQUA, "Electronegativity": ORANGE,
                "Atomic number": MAGENTA, "Atomic radii": YELLOW,
                "Stoichiometry": VIOLET, "Slater-Pauling": RED,
                "Prototype spec": BLUE}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "font.family": "DejaVu Sans",
    "font.size": 9, "axes.edgecolor": "#c3c2b7", "axes.linewidth": 0.8,
    "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": SEC, "ytick.color": SEC,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "legend.frameon": False,
})

df = load_prepared()
dev = df[~df["is_holdout"]]
bench = pd.read_csv("results/benchmark_scenario1.csv")
hold_best = json.load(open("results/holdout_best.json"))
hold_pred = json.load(open("results/holdout_predictions.json"))
stage3 = json.load(open("results/stage3.json"))
unc = json.load(open("results/uncertainty.json"))
imp = json.load(open("results/importance.json"))
leo = json.load(open("results/leave_element_out.json"))

SHORT = {"curie_temperature_K": "Curie temperature",
         "formation_energy_eV_atom": "Formation energy",
         "hull_distance_eV_atom": "Hull distance",
         "total_magnetization_muB_fu": "Magnetization"}
UNIT = {"curie_temperature_K": "K", "formation_energy_eV_atom": "eV/atom",
        "hull_distance_eV_atom": "eV/atom", "total_magnetization_muB_fu": "$\\mu_B$/f.u."}

def style_ax(ax):
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)

# ================= Fig 0: dataset overview ============================
fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2))
type_order = ["Full", "Half", "Inverse", "Quaternary"]
raw = pd.read_csv("data/Heusler_All.csv")
ax = axes[0]
counts = raw["type"].value_counts()[type_order]
ax.bar(type_order, counts.values, color=[TYPE_COLOR[t] for t in type_order], width=0.62)
for i, v in enumerate(counts.values):
    ax.text(i, v + 8, str(v), ha="center", color=SEC, fontsize=8)
ax.set_ylabel("Entries"); ax.set_title("(a) Dataset composition", fontsize=9); style_ax(ax)
ax.tick_params(axis="x", rotation=20)
ax = axes[1]
for t in type_order:
    vals = raw.loc[raw["type"] == t, "curie_temperature_K"]
    ax.hist(vals, bins=30, histtype="step", lw=1.8, color=TYPE_COLOR[t], label=t)
ax.set_xlabel("Curie temperature (K)"); ax.set_ylabel("Count")
ax.set_title("(b) $T_C$ distribution by type", fontsize=9); ax.legend(fontsize=7.5); style_ax(ax)
ax = axes[2]
for t in type_order:
    sub = raw[raw["type"] == t]
    ax.scatter(sub["total_magnetization_muB_fu"], sub["curie_temperature_K"],
               s=9, alpha=0.55, color=TYPE_COLOR[t], label=t, edgecolors="none")
ax.set_xlabel("Magnetization ($\\mu_B$/f.u.)"); ax.set_ylabel("Curie temperature (K)")
ax.set_title("(c) $T_C$ vs magnetization", fontsize=9); style_ax(ax)
fig.tight_layout(); fig.savefig("figures/Fig0_dataset_overview.pdf"); fig.savefig("figures/Fig0_dataset_overview.png", dpi=200)
plt.close(fig)

# ================= Fig 1: modelling stages ============================
fig, ax = plt.subplots(figsize=(8.6, 3.6))
stages = ["Best CV (Scenario I)", "Tuned GB (CV)", "Hold-out", "Scenario II (CV)"]
width = 0.19
xpos = np.arange(len(TARGETS))
vals = {s: [] for s in stages}
for t in TARGETS:
    b = bench[bench.target == t].sort_values("r2_mean", ascending=False).iloc[0]
    vals["Best CV (Scenario I)"].append(b.r2_mean)
    vals["Tuned GB (CV)"].append(stage3["tuned_gb"][t]["cv_tuned_r2"])
    vals["Hold-out"].append(hold_best[t]["holdout_r2"])
    if t == "curie_temperature_K":
        s2 = max(stage3["scenario2_curie"].values(), key=lambda d: d["cv_r2"])["cv_r2"]
        vals["Scenario II (CV)"].append(s2)
    else:
        vals["Scenario II (CV)"].append(np.nan)
colors = [BLUE, ORANGE, AQUA, YELLOW]
for i, s in enumerate(stages):
    v = np.array(vals[s], float)
    bars = ax.bar(xpos + (i - 1.5) * width, v, width - 0.02, color=colors[i], label=s)
    for x, y in zip(xpos + (i - 1.5) * width, v):
        if np.isfinite(y):
            ax.text(x, y + 0.015, f"{y:.2f}", ha="center", fontsize=7, color=SEC)
ax.set_xticks(xpos); ax.set_xticklabels([SHORT[t] for t in TARGETS])
ax.set_ylabel("$R^2$"); ax.set_ylim(0, 1.0)
ax.set_title("Modelling stages: leave-composition-out CV, tuning, locked hold-out, and post-DFT descriptors", fontsize=9)
ax.legend(fontsize=7.5, ncol=4, loc="upper left")
style_ax(ax); fig.tight_layout()
fig.savefig("figures/Fig1_modelling_stages.pdf"); fig.savefig("figures/Fig1_modelling_stages.png", dpi=200)
plt.close(fig)

# ================= Fig 2: algorithm comparison ========================
fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.6))
for ax, target in zip(axes.ravel(), TARGETS):
    sub = bench[bench.target == target].sort_values("r2_mean")
    best_idx = sub["r2_mean"].idxmax()
    cols = [BLUE if i == best_idx else "#b7d3f6" for i in sub.index]
    ax.barh(sub["model"], sub["r2_mean"], xerr=sub["r2_std"], color=cols,
            error_kw=dict(ecolor=MUTED, lw=0.9, capsize=2), height=0.62)
    for y, (r2, mae) in enumerate(zip(sub["r2_mean"], sub["mae_mean"])):
        ax.text(max(r2, 0) + 0.02, y, f"{r2:.3f}", va="center", fontsize=7, color=SEC)
    ax.set_title(f"{SHORT[target]}  (MAE best = {sub.loc[best_idx,'mae_mean']:.3g} {UNIT[target]})", fontsize=9)
    ax.set_xlabel("$R^2$ (leave-composition-out CV)")
    ax.set_xlim(min(-0.05, sub["r2_mean"].min() - 0.1), 1.0)
    style_ax(ax)
fig.tight_layout()
fig.savefig("figures/Fig2_algorithm_comparison.pdf"); fig.savefig("figures/Fig2_algorithm_comparison.png", dpi=200)
plt.close(fig)

# ================= Fig 3: hold-out parity with conformal intervals ====
fig, axes = plt.subplots(2, 2, figsize=(9.6, 8.6))
for ax, target in zip(axes.ravel(), TARGETS):
    d = unc[target]["holdout_pred_conformal"]
    y, p = np.array(d["y"]), np.array(d["p"])
    q = unc[target]["conformal_halfwidth"]
    ax.errorbar(y, p, yerr=q, fmt="none", ecolor="#cde2fb", elinewidth=0.8, zorder=1)
    ax.scatter(y, p, s=14, color=BLUE, alpha=0.75, edgecolors="none", zorder=2)
    lims = [min(y.min(), p.min()), max(y.max(), p.max())]
    pad = 0.05 * (lims[1] - lims[0])
    lims = [lims[0] - pad, lims[1] + pad]
    ax.plot(lims, lims, ls="--", lw=1, color=MUTED, zorder=0)
    ax.set_xlim(lims); ax.set_ylim(lims)
    from sklearn.metrics import r2_score
    ax.set_title(f"{SHORT[target]} — {unc[target]['model']}", fontsize=9)
    ax.text(0.04, 0.95,
            f"hold-out $R^2$ = {r2_score(y, p):.3f}\n90% conformal coverage = {unc[target]['conformal_coverage']:.3f}\nhalf-width = ±{q:.3g} {UNIT[target]}",
            transform=ax.transAxes, va="top", fontsize=7.5, color=SEC)
    ax.set_xlabel(f"DFT/reference {SHORT[target].lower()} ({UNIT[target]})")
    ax.set_ylabel(f"Predicted ({UNIT[target]})")
    style_ax(ax)
fig.tight_layout()
fig.savefig("figures/Fig3_holdout_predictions.pdf"); fig.savefig("figures/Fig3_holdout_predictions.png", dpi=200)
plt.close(fig)

# ================= Fig 4: calibration =================================
fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.7))
ax = axes[0]
tcolors = [BLUE, ORANGE, AQUA, YELLOW]
for c, target in zip(tcolors, TARGETS):
    rel = np.array(unc[target]["reliability"])
    ax.plot(rel[:, 0], rel[:, 1], "-o", ms=3.5, lw=1.6, color=c, label=SHORT[target])
ax.plot([0.5, 1], [0.5, 1], ls="--", color=MUTED, lw=1)
ax.set_xlabel("Nominal coverage"); ax.set_ylabel("Empirical coverage (hold-out)")
ax.set_title("(a) Split-conformal reliability", fontsize=9)
ax.legend(fontsize=7.5); style_ax(ax)
ax = axes[1]
methods = ["quantile_gb_coverage", "bootstrap_coverage", "conformal_coverage"]
mlabels = ["Quantile GB", "Bootstrap", "Split-conformal"]
x = np.arange(len(TARGETS)); w = 0.24
for i, (m, lab, c) in enumerate(zip(methods, mlabels, [ORANGE, MAGENTA, BLUE])):
    v = [unc[t][m] for t in TARGETS]
    ax.bar(x + (i - 1) * w, v, w - 0.02, color=c, label=lab)
ax.axhline(0.9, ls="--", color=INK, lw=1)
ax.text(len(TARGETS) - 0.4, 0.915, "nominal 90%", fontsize=7, color=INK)
ax.set_xticks(x); ax.set_xticklabels([SHORT[t] for t in TARGETS], fontsize=7.5)
ax.set_ylabel("Empirical coverage"); ax.set_ylim(0, 1.05)
ax.set_title("(b) Coverage of nominal 90% intervals", fontsize=9)
ax.legend(fontsize=7.5, loc="lower right"); style_ax(ax)
fig.tight_layout()
fig.savefig("figures/Fig4_calibration.pdf"); fig.savefig("figures/Fig4_calibration.png", dpi=200)
plt.close(fig)

# ================= Fig 5: type classification =========================
cls = pd.read_csv("results/classification_type.csv")
tc = stage3["type_classification"]
fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.9))
ax = axes[0]
sub = cls.sort_values("acc_mean")
cols = [BLUE if m == tc["best_model"] else "#b7d3f6" for m in sub["model"]]
ax.barh(sub["model"], sub["acc_mean"], xerr=sub["acc_std"], color=cols,
        error_kw=dict(ecolor=MUTED, lw=0.9, capsize=2), height=0.6)
ax.axvline(tc["baseline_majority"], ls="--", color=INK, lw=1)
ax.text(tc["baseline_majority"] + 0.008, 0.2, "majority baseline", rotation=90, fontsize=7, color=INK)
ax.set_xlabel("Accuracy (leave-composition-out CV)"); ax.set_xlim(0, 1)
ax.set_title("(a) Heusler-type classification, Tier A descriptors only", fontsize=9)
style_ax(ax)
ax = axes[1]
cm = np.array(tc["holdout_confusion"], float)
cmn = cm / cm.sum(1, keepdims=True)
im = ax.imshow(cmn, cmap=matplotlib.colors.LinearSegmentedColormap.from_list(
    "blues", ["#fcfcfb", "#cde2fb", "#3987e5", "#0d366b"]), vmin=0, vmax=1)
classes = tc["classes"]
ax.set_xticks(range(4), classes, fontsize=8); ax.set_yticks(range(4), classes, fontsize=8)
for i in range(4):
    for j in range(4):
        ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=8,
                color="#ffffff" if cmn[i, j] > 0.55 else INK)
ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.grid(False)
ax.set_title(f"(b) Hold-out confusion (acc = {tc['holdout_accuracy']:.3f}, AUC = {tc['holdout_auc_ovr']:.3f})", fontsize=9)
fig.colorbar(im, ax=ax, fraction=0.046, label="Row fraction")
fig.tight_layout()
fig.savefig("figures/Fig5_type_classification.pdf"); fig.savefig("figures/Fig5_type_classification.png", dpi=200)
plt.close(fig)

# ================= Fig 6: permutation importance ======================
def family_of(feature):
    for fam, cols in FAMILY_BLOCKS.items():
        if feature in cols:
            return fam
    return "Prototype spec"

fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.8))
for ax, target in zip(axes.ravel(), TARGETS):
    ser = pd.Series(imp["permutation"][target]).sort_values(ascending=False)[:10][::-1]
    cols = [FAMILY_COLOR[family_of(f)] for f in ser.index]
    ax.barh(ser.index, ser.values, color=cols, height=0.62)
    ax.set_title(SHORT[target], fontsize=9)
    ax.set_xlabel("Permutation $\\Delta R^2$ (held-out folds)")
    style_ax(ax)
handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in FAMILY_COLOR.values()]
fig.legend(handles, FAMILY_COLOR.keys(), loc="lower center", ncol=4, fontsize=7.5,
           bbox_to_anchor=(0.5, -0.015))
fig.tight_layout(rect=[0, 0.05, 1, 1])
fig.savefig("figures/Fig6_importance.pdf", bbox_inches="tight")
fig.savefig("figures/Fig6_importance.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# ================= Fig 7: block importance ============================
fig, ax = plt.subplots(figsize=(8.6, 3.6))
blocks = list(imp["block"][list(TARGETS)[0]].keys())
x = np.arange(len(blocks)); w = 0.2
for i, (target, c) in enumerate(zip(TARGETS, [BLUE, ORANGE, AQUA, YELLOW])):
    v = [imp["block"][target][b] for b in blocks]
    ax.bar(x + (i - 1.5) * w, v, w - 0.02, color=c, label=SHORT[target])
ax.set_xticks(x); ax.set_xticklabels(blocks, fontsize=8, rotation=12)
ax.set_ylabel("Joint permutation $\\Delta R^2$")
ax.set_title("Collinearity-robust block importance (descriptor families permuted jointly)", fontsize=9)
ax.legend(fontsize=7.5); style_ax(ax)
fig.tight_layout()
fig.savefig("figures/Fig7_block_importance.pdf"); fig.savefig("figures/Fig7_block_importance.png", dpi=200)
plt.close(fig)

# ================= Fig 8: SHAP ========================================
import shap
npz = np.load("results/shap_values.npz", allow_pickle=True)
feats = npz["feats"].tolist(); X = npz["X"]
sv = npz["sv_curie_temperature_K"]
fig = plt.figure(figsize=(11, 4.6))
ax1 = fig.add_subplot(1, 2, 1)
shap.summary_plot(sv, X, feature_names=feats, max_display=12, show=False,
                  plot_size=None, color_bar_label="Descriptor value")
plt.gca().set_title("(a) SHAP summary — Curie temperature (GB surrogate)", fontsize=9)
plt.gca().set_xlabel("SHAP value (K)")
ax2 = fig.add_subplot(1, 2, 2)
j = feats.index("N_v")
type_lab = np.select([dev["type_Full"] == 1, dev["type_Half"] == 1,
                      dev["type_Inverse"] == 1, dev["type_Quaternary"] == 1],
                     ["Full", "Half", "Inverse", "Quaternary"], default="Full")
for t in type_order:
    m = type_lab == t
    ax2.scatter(X[m, j], sv[m, j], s=9, alpha=0.6, color=TYPE_COLOR[t], label=t, edgecolors="none")
ax2.set_xlabel("Total valence-electron count $N_v$")
ax2.set_ylabel("SHAP value for $T_C$ (K)")
ax2.set_title("(b) SHAP dependence on $N_v$ by Heusler type", fontsize=9)
ax2.legend(fontsize=7.5); style_ax(ax2)
fig.tight_layout()
fig.savefig("figures/Fig8_shap.pdf", bbox_inches="tight")
fig.savefig("figures/Fig8_shap.png", dpi=200, bbox_inches="tight")
plt.close("all")

# ================= Fig 9: leave-element-out ===========================
fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
CLIP = -0.4
for ax, site, ttl in zip(axes, ["Z", "Y"], ["Z site (main group)", "Y site (transition metal)"]):
    key_tc = f"{site}|curie_temperature_K"; key_m = f"{site}|total_magnetization_muB_fu"
    els = sorted(leo[key_tc]["per_element"], key=lambda e: -leo[key_tc]["per_element"][e]["r2"])
    x = np.arange(len(els)); w = 0.38
    v1 = np.clip([leo[key_tc]["per_element"][e]["r2"] for e in els], CLIP, None)
    v2 = np.clip([leo[key_m]["per_element"][e]["r2"] for e in els], CLIP, None)
    ax.bar(x - w / 2, v1, w - 0.02, color=BLUE, label="Curie temperature")
    ax.bar(x + w / 2, v2, w - 0.02, color=ORANGE, label="Magnetization")
    ax.axhline(0, color="#c3c2b7", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(els, fontsize=8)
    ax.set_ylabel("$R^2$ (element held out)")
    ax.set_ylim(CLIP - 0.05, 1.0)
    ax.set_title(f"Leave-element-out, {ttl}  (clipped at {CLIP})", fontsize=9)
    ax.legend(fontsize=7.5, loc="lower left"); style_ax(ax)
fig.tight_layout()
fig.savefig("figures/Fig9_leave_element_out.pdf"); fig.savefig("figures/Fig9_leave_element_out.png", dpi=200)
plt.close(fig)

print("figures written:", sorted(__import__("os").listdir("figures")))
