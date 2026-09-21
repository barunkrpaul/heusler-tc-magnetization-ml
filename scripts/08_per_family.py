"""Stage 8: per-family breakdown.

(a) The pooled model (best algorithm per target, trained on all families with
    the type flag) is evaluated WITHIN each family: out-of-fold grouped-CV
    predictions on the development set are split by Heusler type, and the
    locked hold-out is likewise scored per family.
(b) Family-specific models: the full 10-algorithm benchmark is re-run inside
    each family alone (its own development rows, its own grouped CV), and the
    family's best model is scored once on that family's hold-out rows.
(c) Pooled vs family-specific comparison on identical evaluation rows.

Note on interpretation: within-family R^2 is measured against the family's own
variance, which is smaller than the pooled variance, so within-family R^2 is
expected to be lower than pooled R^2 even at identical absolute error. MAE is
the normalization-free comparison.
"""
import sys, json, warnings
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import r2_score, mean_absolute_error

warnings.filterwarnings("ignore")
sys.path.insert(0, "scripts")
from common import load_prepared, make_models, grouped_folds, TIER_A, TARGETS, RNG

df = load_prepared()
FEATS = TIER_A + [c for c in df.columns if c.startswith("type_") or c.startswith("sg_")] + ["SP_selected"]
TYPES = ["Full", "Half", "Inverse", "Quaternary"]
df["family"] = np.select([df[f"type_{t}"] == 1 for t in TYPES], TYPES, default="Full")

dev = df[~df["is_holdout"]].reset_index(drop=True)
hold = df[df["is_holdout"]].reset_index(drop=True)
Xd, Xh = dev[FEATS].to_numpy(float), hold[FEATS].to_numpy(float)
groups = dev["isomer_group"].to_numpy()
best = json.load(open("results/holdout_best.json"))

out = {"note": "within-family R2 uses the family's own variance; compare MAE across settings"}

# ---------- (a) pooled model, evaluated per family --------------------
pooled = {}
N_REP = 3
for target, info in best.items():
    yd = dev[target].to_numpy(float)
    model = make_models()[info["best_model"]]
    # out-of-fold predictions per repeat
    oof = np.full((N_REP, len(dev)), np.nan)
    rep_folds = list(grouped_folds(groups, 5, N_REP))
    for i, (tr, te) in enumerate(rep_folds):
        rep = i // 5
        m = clone(model); m.fit(Xd[tr], yd[tr])
        oof[rep, te] = m.predict(Xd[te])
    # hold-out predictions from the full development fit
    m = clone(model); m.fit(Xd, yd)
    ph = m.predict(Xh); yh = hold[target].to_numpy(float)
    fam_res = {}
    for t in TYPES:
        md = (dev["family"] == t).to_numpy()
        r2s = [r2_score(yd[md], oof[r, md]) for r in range(N_REP)]
        maes = [mean_absolute_error(yd[md], oof[r, md]) for r in range(N_REP)]
        mh = (hold["family"] == t).to_numpy()
        fam_res[t] = {"n_dev": int(md.sum()), "n_holdout": int(mh.sum()),
                      "cv_r2": float(np.mean(r2s)), "cv_r2_std": float(np.std(r2s)),
                      "cv_mae": float(np.mean(maes)),
                      "holdout_r2": float(r2_score(yh[mh], ph[mh])),
                      "holdout_mae": float(mean_absolute_error(yh[mh], ph[mh]))}
    pooled[target] = {"model": info["best_model"], "per_family": fam_res}
    print("pooled", target, {t: round(fam_res[t]["cv_r2"], 3) for t in TYPES})
out["pooled_per_family"] = pooled

# ---------- (b) family-specific benchmark -----------------------------
records = []
fam_best = {}
for t in TYPES:
    dsub = dev[dev["family"] == t].reset_index(drop=True)
    hsub = hold[hold["family"] == t].reset_index(drop=True)
    Xs, Xhs = dsub[FEATS].to_numpy(float), hsub[FEATS].to_numpy(float)
    gsub = dsub["isomer_group"].to_numpy()
    for target in TARGETS:
        ys = dsub[target].to_numpy(float)
        yhs = hsub[target].to_numpy(float)
        best_name, best_r2 = None, -np.inf
        for name, model in make_models().items():
            r2s, maes = [], []
            for tr, te in grouped_folds(gsub, 5, 3):
                m = clone(model); m.fit(Xs[tr], ys[tr])
                p = m.predict(Xs[te])
                r2s.append(r2_score(ys[te], p))
                maes.append(mean_absolute_error(ys[te], p))
            records.append({"family": t, "target": target, "model": name,
                            "r2_mean": float(np.mean(r2s)), "r2_std": float(np.std(r2s)),
                            "mae_mean": float(np.mean(maes))})
            if np.mean(r2s) > best_r2:
                best_r2, best_name = np.mean(r2s), name
        m = clone(make_models()[best_name]); m.fit(Xs, ys)
        p = m.predict(Xhs)
        fam_best[f"{t}|{target}"] = {
            "best_model": best_name, "cv_r2": float(best_r2),
            "cv_mae": float(min(r["mae_mean"] for r in records
                                if r["family"] == t and r["target"] == target and r["model"] == best_name)),
            "holdout_r2": float(r2_score(yhs, p)),
            "holdout_mae": float(mean_absolute_error(yhs, p))}
        print(t, target, best_name, "CV R2=%.3f" % best_r2,
              "holdout R2=%.3f" % fam_best[f"{t}|{target}"]["holdout_r2"])

pd.DataFrame(records).to_csv("results/benchmark_per_family.csv", index=False)
out["family_specific_best"] = fam_best

# ---------- (c) pooled vs family-specific on identical rows -----------
comp = []
for t in TYPES:
    for target in TARGETS:
        p = pooled[target]["per_family"][t]
        f = fam_best[f"{t}|{target}"]
        comp.append({"family": t, "target": target,
                     "pooled_cv_r2": p["cv_r2"], "pooled_cv_mae": p["cv_mae"],
                     "family_cv_r2": f["cv_r2"], "family_cv_mae": f["cv_mae"],
                     "pooled_holdout_r2": p["holdout_r2"], "pooled_holdout_mae": p["holdout_mae"],
                     "family_holdout_r2": f["holdout_r2"], "family_holdout_mae": f["holdout_mae"],
                     "family_best_model": f["best_model"], "pooled_model": pooled[target]["model"]})
comp_df = pd.DataFrame(comp)
comp_df.to_csv("results/pooled_vs_family.csv", index=False)
json.dump(out, open("results/per_family.json", "w"), indent=2)
print(comp_df.round(3).to_string())
