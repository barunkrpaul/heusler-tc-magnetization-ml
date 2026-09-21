"""Stage 2: benchmark 10 algorithms under leave-composition-out CV (Scenario I)
and score the best model per target once on the locked hold-out."""
import sys, json, warnings
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.base import clone

warnings.filterwarnings("ignore")
sys.path.insert(0, "scripts")
from common import load_prepared, make_models, grouped_folds, TIER_A, tier_b_cols, TARGETS, RNG

df = load_prepared()
FEATS = TIER_A + [c for c in df.columns if c.startswith("type_") or c.startswith("sg_")] + ["SP_selected"]
dev = df[~df["is_holdout"]].reset_index(drop=True)
hold = df[df["is_holdout"]].reset_index(drop=True)

Xd, Xh = dev[FEATS].to_numpy(float), hold[FEATS].to_numpy(float)
groups = dev["isomer_group"].to_numpy()

records, fold_scores = [], {}
for target in TARGETS:
    yd = dev[target].to_numpy(float)
    for name, model in make_models().items():
        r2s, maes = [], []
        for tr, te in grouped_folds(groups, n_splits=5, n_repeats=3):
            m = clone(model)
            m.fit(Xd[tr], yd[tr])
            p = m.predict(Xd[te])
            r2s.append(r2_score(yd[te], p))
            maes.append(mean_absolute_error(yd[te], p))
        r2s, maes = np.array(r2s), np.array(maes)
        boot = np.random.default_rng(RNG).choice(r2s, (2000, len(r2s))).mean(1)
        records.append({
            "target": target, "model": name,
            "r2_mean": r2s.mean(), "r2_std": r2s.std(),
            "r2_ci_lo": np.percentile(boot, 2.5), "r2_ci_hi": np.percentile(boot, 97.5),
            "mae_mean": maes.mean(),
        })
        fold_scores[f"{target}|{name}"] = r2s.tolist()
        print(f"{target:28s} {name:20s} R2={r2s.mean():.3f}±{r2s.std():.3f} MAE={maes.mean():.4g}")

bench = pd.DataFrame(records)
bench.to_csv("results/benchmark_scenario1.csv", index=False)

# ---- best model per target -> locked hold-out, scored once ----------
hold_results, hold_pred = {}, {}
for target in TARGETS:
    sub = bench[bench.target == target].sort_values("r2_mean", ascending=False)
    best_name = sub.iloc[0]["model"]
    m = clone(make_models()[best_name])
    m.fit(Xd, dev[target].to_numpy(float))
    p = m.predict(Xh)
    yh = hold[target].to_numpy(float)
    hold_results[target] = {"best_model": best_name,
                            "holdout_r2": float(r2_score(yh, p)),
                            "holdout_mae": float(mean_absolute_error(yh, p))}
    hold_pred[target] = {"y_true": yh.tolist(), "y_pred": p.tolist(),
                         "composition": hold["composition"].tolist()}
    print(target, hold_results[target])

json.dump(hold_results, open("results/holdout_best.json", "w"), indent=2)
json.dump(hold_pred, open("results/holdout_predictions.json", "w"))
json.dump(fold_scores, open("results/fold_scores.json", "w"))
