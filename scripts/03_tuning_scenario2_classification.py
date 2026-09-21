"""Stage 3: (a) hyperparameter-tuned Gradient Boosting per target,
(b) Scenario II (Tier A+B+C descriptors) for the Curie temperature,
(c) Heusler-type classification from composition-only (Tier A) descriptors."""
import sys, json, warnings
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, GroupKFold
from sklearn.metrics import (r2_score, mean_absolute_error, accuracy_score,
                             f1_score, roc_auc_score, confusion_matrix)

warnings.filterwarnings("ignore")
sys.path.insert(0, "scripts")
from common import (load_prepared, make_models, grouped_folds, TIER_A, TIER_C,
                    TARGETS, RNG)

df = load_prepared()
TIERB = [c for c in df.columns if c.startswith("type_") or c.startswith("sg_")] + ["SP_selected"]
FEATS = TIER_A + TIERB
dev = df[~df["is_holdout"]].reset_index(drop=True)
hold = df[df["is_holdout"]].reset_index(drop=True)
groups = dev["isomer_group"].to_numpy()
Xd, Xh = dev[FEATS].to_numpy(float), hold[FEATS].to_numpy(float)

out = {}

# ---------------- (a) tuned Gradient Boosting -------------------------
space = {"n_estimators": [200, 400, 600], "max_depth": [3, 4, 5, 6, 7],
         "learning_rate": [0.01, 0.03, 0.05, 0.1],
         "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
         "min_samples_leaf": [1, 2, 3, 5],
         "max_features": ["sqrt", "log2", None]}
tuned = {}
for target in TARGETS:
    yd = dev[target].to_numpy(float)
    search = RandomizedSearchCV(GradientBoostingRegressor(random_state=RNG),
                                space, n_iter=20, cv=GroupKFold(5),
                                scoring="r2", random_state=RNG, n_jobs=-1)
    search.fit(Xd, yd, groups=groups)
    best = search.best_estimator_
    # default vs tuned under leave-composition-out CV
    scores = {"default": [], "tuned": []}
    for tr, te in grouped_folds(groups, 5, 3):
        for tag, est in [("default", GradientBoostingRegressor(n_estimators=300, random_state=RNG)),
                         ("tuned", clone(best))]:
            est.fit(Xd[tr], yd[tr])
            scores[tag].append(r2_score(yd[te], est.predict(Xd[te])))
    m = clone(best); m.fit(Xd, yd)
    ph = m.predict(Xh); yh = hold[target].to_numpy(float)
    tuned[target] = {"params": search.best_params_,
                     "cv_default_r2": float(np.mean(scores["default"])),
                     "cv_tuned_r2": float(np.mean(scores["tuned"])),
                     "holdout_tuned_r2": float(r2_score(yh, ph)),
                     "holdout_tuned_mae": float(mean_absolute_error(yh, ph))}
    print("tuned GB", target, {k: v for k, v in tuned[target].items() if k != "params"})
out["tuned_gb"] = tuned

# ---------------- (b) Scenario II for Curie temperature ---------------
FEATS2 = FEATS + TIER_C
Xd2, Xh2 = dev[FEATS2].to_numpy(float), hold[FEATS2].to_numpy(float)
sc2 = {}
for target in ["curie_temperature_K"]:
    yd = dev[target].to_numpy(float); yh = hold[target].to_numpy(float)
    for name in ["Random Forest", "Gradient Boosting", "LightGBM"]:
        model = make_models()[name]
        r2s, maes = [], []
        for tr, te in grouped_folds(groups, 5, 3):
            m = clone(model); m.fit(Xd2[tr], yd[tr])
            p = m.predict(Xd2[te])
            r2s.append(r2_score(yd[te], p)); maes.append(mean_absolute_error(yd[te], p))
        m = clone(model); m.fit(Xd2, yd)
        sc2[name] = {"cv_r2": float(np.mean(r2s)), "cv_r2_std": float(np.std(r2s)),
                     "cv_mae": float(np.mean(maes)),
                     "holdout_r2": float(r2_score(yh, m.predict(Xh2)))}
        print("ScenarioII", name, sc2[name])
out["scenario2_curie"] = sc2

# permutation-style check: which Tier C descriptor carries the gain
best_s2 = max(sc2, key=lambda k: sc2[k]["cv_r2"])
gain_detail = {}
for add in TIER_C:
    cols = FEATS + [add]
    Xa = dev[cols].to_numpy(float)
    r2s = []
    for tr, te in grouped_folds(groups, 5, 2):
        m = clone(make_models()[best_s2]); m.fit(Xa[tr], dev["curie_temperature_K"].to_numpy(float)[tr])
        r2s.append(r2_score(dev["curie_temperature_K"].to_numpy(float)[te], m.predict(Xa[te])))
    gain_detail[add] = float(np.mean(r2s))
out["scenario2_single_addition_r2"] = gain_detail
print("single Tier C additions:", gain_detail)

# ---------------- (c) Heusler-type classification (Tier A only) -------
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

ycls = dev["type_Full"]*0  # placeholder
type_label = np.select(
    [dev["type_Full"] == 1, dev["type_Half"] == 1, dev["type_Inverse"] == 1, dev["type_Quaternary"] == 1],
    ["Full", "Half", "Inverse", "Quaternary"], default="Full")
type_label_h = np.select(
    [hold["type_Full"] == 1, hold["type_Half"] == 1, hold["type_Inverse"] == 1, hold["type_Quaternary"] == 1],
    ["Full", "Half", "Inverse", "Quaternary"], default="Full")
XdA, XhA = dev[TIER_A].to_numpy(float), hold[TIER_A].to_numpy(float)

def scaled(est): return Pipeline([("s", StandardScaler()), ("m", est)])
clfs = {
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=RNG, n_jobs=-1),
    "Gradient Boosting": GradientBoostingClassifier(random_state=RNG),
    "XGBoost": XGBClassifier(n_estimators=300, random_state=RNG, verbosity=0, n_jobs=-1),
    "LightGBM": LGBMClassifier(n_estimators=300, random_state=RNG, verbose=-1, n_jobs=-1),
    "SVC": scaled(SVC(probability=True, random_state=RNG)),
    "KNN": scaled(KNeighborsClassifier()),
    "MLP (sklearn)": scaled(MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=1500,
                                          random_state=RNG, early_stopping=True)),
    "Logistic Regression": scaled(LogisticRegression(max_iter=3000)),
}
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder(); yenc = le.fit_transform(type_label)
cls_rows = []
for name, clf in clfs.items():
    accs, f1s, aucs = [], [], []
    for tr, te in grouped_folds(groups, 5, 2):
        m = clone(clf); m.fit(XdA[tr], yenc[tr])
        p = m.predict(XdA[te]); prob = m.predict_proba(XdA[te])
        accs.append(accuracy_score(yenc[te], p))
        f1s.append(f1_score(yenc[te], p, average="macro"))
        aucs.append(roc_auc_score(yenc[te], prob, multi_class="ovr"))
    cls_rows.append({"model": name, "acc_mean": np.mean(accs), "acc_std": np.std(accs),
                     "f1_macro": np.mean(f1s), "auc_ovr": np.mean(aucs)})
    print("cls", name, cls_rows[-1])
cls_df = pd.DataFrame(cls_rows).sort_values("acc_mean", ascending=False)
cls_df.to_csv("results/classification_type.csv", index=False)

best_clf_name = cls_df.iloc[0]["model"]
m = clone(clfs[best_clf_name]); m.fit(XdA, yenc)
ph = m.predict(XhA); yh = le.transform(type_label_h)
cm = confusion_matrix(yh, ph)
out["type_classification"] = {
    "baseline_majority": float(max(np.bincount(yenc)) / len(yenc)),
    "best_model": best_clf_name,
    "holdout_accuracy": float(accuracy_score(yh, ph)),
    "holdout_f1_macro": float(f1_score(yh, ph, average="macro")),
    "holdout_auc_ovr": float(roc_auc_score(yh, m.predict_proba(XhA), multi_class="ovr")),
    "holdout_confusion": cm.tolist(), "classes": le.classes_.tolist()}
print(out["type_classification"])

json.dump(out, open("results/stage3.json", "w"), indent=2)
