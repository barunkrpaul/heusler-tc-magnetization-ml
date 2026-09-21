"""Stage 6: leave-element-out chemical extrapolation.
Hold out every compound containing a given element at the Y site (transition
metal) or the Z site (main group) in turn; train on the remainder."""
import sys, json, warnings
import numpy as np
from sklearn.base import clone
from sklearn.metrics import r2_score

warnings.filterwarnings("ignore")
sys.path.insert(0, "scripts")
from common import (load_prepared, make_models, TIER_A, TARGETS,
                    parse_composition, site_slots)

df = load_prepared()
FEATS = TIER_A + [c for c in df.columns if c.startswith("type_") or c.startswith("sg_")] + ["SP_selected"]
X = df[FEATS].to_numpy(float)

# element occupying each site slot
site_el = {"X": [], "Y": [], "Z": []}
for c in df["composition"]:
    slots = site_slots(parse_composition(c))
    site_el["X"].append(slots[0][0])
    site_el["Y"].append(slots[1][0] if slots[1] else None)
    site_el["Z"].append(slots[2][0] if slots[2] else None)
for k in site_el:
    site_el[k] = np.array(site_el[k], dtype=object)

best = json.load(open("results/holdout_best.json"))
MIN_N = 10
out = {}
for site in ["Y", "Z"]:
    elements = [e for e in sorted(set(site_el[site]) - {None})
                if (site_el[site] == e).sum() >= MIN_N]
    for target, info in best.items():
        model = make_models()[info["best_model"]]
        rows, wr2_num, wr2_den = {}, 0.0, 0
        for el in elements:
            te = np.where(site_el[site] == el)[0]
            tr = np.where(site_el[site] != el)[0]
            y = df[target].to_numpy(float)
            m = clone(model); m.fit(X[tr], y[tr])
            r2 = r2_score(y[te], m.predict(X[te]))
            rows[el] = {"n": int(len(te)), "r2": float(r2)}
            wr2_num += r2 * len(te); wr2_den += len(te)
        out[f"{site}|{target}"] = {"model": info["best_model"],
                                   "weighted_r2": wr2_num / wr2_den,
                                   "per_element": rows}
        worst = min(rows, key=lambda e: rows[e]["r2"])
        print(site, target, "weighted R2=%.3f" % (wr2_num / wr2_den),
              "worst:", worst, round(rows[worst]["r2"], 2))

json.dump(out, open("results/leave_element_out.json", "w"), indent=2)
