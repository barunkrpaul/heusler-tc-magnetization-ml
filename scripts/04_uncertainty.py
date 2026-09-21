"""Stage 4: uncertainty quantification.
90% prediction intervals by (i) split-conformal calibration of the best model,
(ii) quantile Gradient Boosting, (iii) bootstrap ensemble; empirical coverage
measured on the locked hold-out (never assumed)."""
import sys, json, warnings
import numpy as np
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GroupShuffleSplit

warnings.filterwarnings("ignore")
sys.path.insert(0, "scripts")
from common import load_prepared, make_models, TIER_A, TARGETS, RNG

df = load_prepared()
FEATS = TIER_A + [c for c in df.columns if c.startswith("type_") or c.startswith("sg_")] + ["SP_selected"]
dev = df[~df["is_holdout"]].reset_index(drop=True)
hold = df[df["is_holdout"]].reset_index(drop=True)
Xd, Xh = dev[FEATS].to_numpy(float), hold[FEATS].to_numpy(float)

best = json.load(open("results/holdout_best.json"))
ALPHA = 0.10
out = {}
for target, info in best.items():
    yd, yh = dev[target].to_numpy(float), hold[target].to_numpy(float)
    model = make_models()[info["best_model"]]

    # (i) split-conformal: proper train / calibration split of the development
    # set, respecting isomer groups so a swap pair never straddles the split
    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=RNG)
    tr_i, cal_i = next(gss.split(Xd, yd, groups=dev["isomer_group"].to_numpy()))
    Xtr, Xcal, ytr, ycal = Xd[tr_i], Xd[cal_i], yd[tr_i], yd[cal_i]
    m = clone(model); m.fit(Xtr, ytr)
    resid = np.abs(ycal - m.predict(Xcal))
    n = len(resid)
    q = np.quantile(resid, min(1.0, np.ceil((n + 1) * (1 - ALPHA)) / n), method="higher")
    p_h = m.predict(Xh)
    conf_cov = float(np.mean(np.abs(yh - p_h) <= q))

    # conformal reliability curve (empirical vs nominal)
    rel = []
    for nom in np.arange(0.5, 1.0, 0.05):
        qn = np.quantile(resid, min(1.0, np.ceil((n + 1) * nom) / n), method="higher")
        rel.append([float(nom), float(np.mean(np.abs(yh - p_h) <= qn))])

    # (ii) quantile gradient boosting
    lo = GradientBoostingRegressor(loss="quantile", alpha=ALPHA / 2, random_state=RNG).fit(Xd, yd)
    hi = GradientBoostingRegressor(loss="quantile", alpha=1 - ALPHA / 2, random_state=RNG).fit(Xd, yd)
    qgb_cov = float(np.mean((yh >= lo.predict(Xh)) & (yh <= hi.predict(Xh))))

    # (iii) bootstrap ensemble of the point model (estimation uncertainty only)
    rng = np.random.default_rng(RNG)
    preds = []
    for b in range(30):
        idx = rng.choice(len(Xd), len(Xd), replace=True)
        mb = clone(model); mb.fit(Xd[idx], yd[idx])
        preds.append(mb.predict(Xh))
    preds = np.array(preds)
    blo, bhi = np.percentile(preds, [5, 95], axis=0)
    boot_cov = float(np.mean((yh >= blo) & (yh <= bhi)))

    out[target] = {"model": info["best_model"],
                   "conformal_halfwidth": float(q),
                   "conformal_coverage": conf_cov,
                   "quantile_gb_coverage": qgb_cov,
                   "bootstrap_coverage": boot_cov,
                   "reliability": rel,
                   "holdout_pred_conformal": {"y": yh.tolist(), "p": p_h.tolist()}}
    print(target, {k: v for k, v in out[target].items() if k not in ("reliability", "holdout_pred_conformal")})

json.dump(out, open("results/uncertainty.json", "w"))
