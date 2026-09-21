"""Stage 5: interpretability — held-out permutation importance, grouped block
permutation (collinearity-robust), and TreeSHAP on a GB surrogate."""
import sys, json, warnings
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import r2_score
from sklearn.ensemble import GradientBoostingRegressor

warnings.filterwarnings("ignore")
sys.path.insert(0, "scripts")
from common import load_prepared, make_models, grouped_folds, TIER_A, FAMILY_BLOCKS, TARGETS, RNG

df = load_prepared()
TIERB = [c for c in df.columns if c.startswith("type_") or c.startswith("sg_")] + ["SP_selected"]
FEATS = TIER_A + TIERB
dev = df[~df["is_holdout"]].reset_index(drop=True)
groups = dev["isomer_group"].to_numpy()
Xd = dev[FEATS].to_numpy(float)

best = json.load(open("results/holdout_best.json"))
blocks = dict(FAMILY_BLOCKS)
blocks["Prototype spec"] = [c for c in TIERB]

perm_out, block_out = {}, {}
rng = np.random.default_rng(RNG)
for target, info in best.items():
    yd = dev[target].to_numpy(float)
    model = make_models()[info["best_model"]]
    imp = np.zeros(len(FEATS)); blk = {b: 0.0 for b in blocks}
    nfold = 0
    NREP = 10
    for tr, te in grouped_folds(groups, 5, 1):
        m = clone(model); m.fit(Xd[tr], yd[tr])
        nte = len(te)
        base = r2_score(yd[te], m.predict(Xd[te]))
        # per-descriptor permutation (NREP repeats, batched into one predict)
        for j in range(len(FEATS)):
            Xbig = np.tile(Xd[te], (NREP, 1))
            for rep in range(NREP):
                Xbig[rep * nte:(rep + 1) * nte, j] = rng.permutation(Xd[te][:, j])
            pbig = m.predict(Xbig)
            drops = [base - r2_score(yd[te], pbig[rep * nte:(rep + 1) * nte])
                     for rep in range(NREP)]
            imp[j] += np.mean(drops)
        # block permutation (NREP repeats, batched)
        for bname, cols in blocks.items():
            jidx = [FEATS.index(c) for c in cols if c in FEATS]
            Xbig = np.tile(Xd[te], (NREP, 1))
            for rep in range(NREP):
                pi = rng.permutation(nte)
                Xbig[rep * nte:(rep + 1) * nte][:, jidx] = Xd[te][pi][:, jidx]
            pbig = m.predict(Xbig)
            drops = [base - r2_score(yd[te], pbig[rep * nte:(rep + 1) * nte])
                     for rep in range(NREP)]
            blk[bname] += np.mean(drops)
        nfold += 1
    perm_out[target] = dict(zip(FEATS, (imp / nfold).tolist()))
    block_out[target] = {b: v / nfold for b, v in blk.items()}
    top = sorted(perm_out[target].items(), key=lambda kv: -kv[1])[:5]
    print(target, "top5:", [(k, round(v, 3)) for k, v in top])
    print("  blocks:", {k: round(v, 3) for k, v in block_out[target].items()})

json.dump({"permutation": perm_out, "block": block_out},
          open("results/importance.json", "w"), indent=2)

# ---------------- TreeSHAP on a GB surrogate --------------------------
import shap
shap_store = {}
for target in TARGETS:
    yd = dev[target].to_numpy(float)
    gb = GradientBoostingRegressor(n_estimators=300, random_state=RNG).fit(Xd, yd)
    ex = shap.TreeExplainer(gb)
    sv = ex.shap_values(Xd)
    shap_store[target] = sv
np.savez_compressed("results/shap_values.npz",
                    feats=np.array(FEATS),
                    X=Xd,
                    **{f"sv_{t}": shap_store[t] for t in TARGETS})
print("SHAP done")
