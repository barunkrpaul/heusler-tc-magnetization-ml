"""Common utilities for the Heusler Curie-temperature ML pipeline.

Methodology follows Paul, Giri, Datta & Pal (2026), "Multi-Target Machine
Learning and Deep Learning Framework for Predicting Structural, Energetic,
and Magnetic Properties of Ternary Heusler Alloys": descriptor-provenance
audit (Tier A / B / C), leave-composition-out grouped cross-validation,
locked composition-level hold-out scored once, split-conformal uncertainty,
permutation / block / SHAP interpretability, leave-element-out extrapolation.
"""
import re
import numpy as np
import pandas as pd

RNG = 42
DATA_DIR = "data"
RESULTS_DIR = "results"
FIG_DIR = "figures"

# ----------------------------------------------------------------------
# Elemental property tables (a priori, composition-level  ->  Tier A)
# Valence electron count = group number, the convention used in
# Slater-Pauling electron counting for Heusler alloys.
# ----------------------------------------------------------------------
VALENCE = {
    "Sc": 3, "Y": 3, "Ti": 4, "Zr": 4, "Hf": 4, "V": 5, "Nb": 5, "Ta": 5,
    "Cr": 6, "Mo": 6, "W": 6, "Mn": 7, "Tc": 7, "Re": 7, "Fe": 8, "Ru": 8,
    "Os": 8, "Co": 9, "Rh": 9, "Ir": 9, "Ni": 10, "Pd": 10, "Pt": 10,
    "Cu": 11, "Ag": 11, "Au": 11, "Zn": 12,
    "Al": 3, "Ga": 3, "In": 3, "Tl": 3, "Si": 4, "Ge": 4, "Sn": 4, "Pb": 4,
    "P": 5, "As": 5, "Sb": 5, "Bi": 5,
}
ELECTRONEG = {
    "Sc": 1.36, "Y": 1.22, "Ti": 1.54, "Zr": 1.33, "Hf": 1.30, "V": 1.63,
    "Nb": 1.60, "Ta": 1.50, "Cr": 1.66, "Mo": 2.16, "W": 2.36, "Mn": 1.55,
    "Tc": 1.90, "Re": 1.90, "Fe": 1.83, "Ru": 2.20, "Os": 2.20, "Co": 1.88,
    "Rh": 2.28, "Ir": 2.20, "Ni": 1.91, "Pd": 2.20, "Pt": 2.28, "Cu": 1.90,
    "Ag": 1.93, "Au": 2.54, "Zn": 1.65, "Al": 1.61, "Ga": 1.81, "In": 1.78,
    "Tl": 1.62, "Si": 1.90, "Ge": 2.01, "Sn": 1.96, "Pb": 2.33, "P": 2.19,
    "As": 2.18, "Sb": 2.05, "Bi": 2.02,
}
ATOMIC_NUMBER = {
    "Sc": 21, "Y": 39, "Ti": 22, "Zr": 40, "Hf": 72, "V": 23, "Nb": 41,
    "Ta": 73, "Cr": 24, "Mo": 42, "W": 74, "Mn": 25, "Tc": 43, "Re": 75,
    "Fe": 26, "Ru": 44, "Os": 76, "Co": 27, "Rh": 45, "Ir": 77, "Ni": 28,
    "Pd": 46, "Pt": 78, "Cu": 29, "Ag": 47, "Au": 79, "Zn": 30, "Al": 13,
    "Ga": 31, "In": 49, "Tl": 81, "Si": 14, "Ge": 32, "Sn": 50, "Pb": 82,
    "P": 15, "As": 33, "Sb": 51, "Bi": 83,
}
MAGNETIC_3D = {"Cr", "Mn", "Fe", "Co", "Ni"}  # strongly magnetic 3d species

TARGETS = {
    "curie_temperature_K": "Curie temperature (K)",
    "formation_energy_eV_atom": "Formation energy (eV/atom)",
    "hull_distance_eV_atom": "Hull distance (eV/atom)",
    "total_magnetization_muB_fu": "Magnetization ($\\mu_B$/f.u.)",
}

TOKEN_RE = re.compile(r"([A-Z][a-z]?)(\d*)")


def parse_composition(comp):
    """Return ordered [(element, count), ...] for a Heusler formula string."""
    return [(el, int(n) if n else 1) for el, n in TOKEN_RE.findall(comp) if el]


def site_slots(tokens):
    """Map ordered formula tokens onto X, Y, Z, R site slots (R absent -> None)."""
    slots = [None, None, None, None]
    for i, tok in enumerate(tokens[:4]):
        slots[i] = tok
    return slots


def build_features(df):
    """Engineer Tier A / Tier B descriptor blocks from the raw table."""
    rows = []
    for _, r in df.iterrows():
        toks = parse_composition(r["composition"])
        slots = site_slots(toks)
        elems = [s[0] if s else None for s in slots]
        counts = [s[1] if s else 0 for s in slots]
        n_atoms = sum(counts)
        v = [VALENCE.get(e, 0.0) if e else 0.0 for e in elems]
        en = [ELECTRONEG.get(e, 0.0) if e else 0.0 for e in elems]
        zn = [ATOMIC_NUMBER.get(e, 0.0) if e else 0.0 for e in elems]
        # corrected files: radius_R is NaN (not-applicable) for ternary rows;
        # encode as 0 with an explicit has_R_site indicator alongside
        rad = [r["radius_X"], r["radius_Y"], r["radius_Z"],
               0.0 if pd.isna(r["radius_R"]) else r["radius_R"]]
        w = np.array(counts, float)
        active = w > 0
        nv = float(np.dot(w, v))
        en_mean = float(np.dot(w, en) / n_atoms)
        rad_arr = np.array(rad, float)
        rad_active = rad_arr[active] if active.any() else np.array([0.0])
        n_mag = int(sum(c for (e, c) in toks if e in MAGNETIC_3D))
        d = {
            "composition": r["composition"],
            # ---------------- Tier A: composition-level ----------------
            "V_X": v[0], "V_Y": v[1], "V_Z": v[2], "V_R": v[3],
            "N_v": nv,
            "dV_XY": v[0] - v[1], "dV_XZ": v[0] - v[2], "dV_YZ": v[1] - v[2],
            "EN_X": en[0], "EN_Y": en[1], "EN_Z": en[2], "EN_R": en[3],
            "EN_mean": en_mean,
            "dEN_XY": en[0] - en[1], "dEN_XZ": en[0] - en[2],
            "Z_X": zn[0], "Z_Y": zn[1], "Z_Z": zn[2], "Z_R": zn[3],
            "R_X": rad[0], "R_Y": rad[1], "R_Z": rad[2], "R_R": rad[3],
            "R_mean": float(rad_active.mean()),
            "R_spread": float(rad_active.max() - rad_active.min()),
            "dR_XY": rad[0] - rad[1], "dR_XZ": rad[0] - rad[2],
            "n_atoms_fu": n_atoms,
            "n_magnetic_3d": n_mag,
            "has_R_site": int(len(toks) >= 4),
            "n_unique_elements": len({e for e, _ in toks}),
            # "is_reduced" is merged from the corrected files in stage 01
            # (family-relative definition, e.g. Cr3Al labelled Full)
            "SP_24": nv - 24.0,       # full/inverse Slater-Pauling prior
            "SP_18": nv - 18.0,       # half-Heusler Slater-Pauling prior
            # ---------------- Tier B: structure/prototype specification ----
            "type": r["type"],
            "space_group": r["space_group"],
        }
        rows.append(d)
    feat = pd.DataFrame(rows)
    # one-hot encode Tier B categoricals
    feat = pd.get_dummies(feat, columns=["type", "space_group"], prefix=["type", "sg"])
    # SP prior selected by the specified prototype (Tier B, cf. SP_mag_pred)
    half_col = feat.get("type_Half")
    feat["SP_selected"] = np.where(half_col.astype(bool), feat["SP_18"], feat["SP_24"]) \
        if half_col is not None else feat["SP_24"]
    return feat


TIER_A = [
    "V_X", "V_Y", "V_Z", "V_R", "N_v", "dV_XY", "dV_XZ", "dV_YZ",
    "EN_X", "EN_Y", "EN_Z", "EN_R", "EN_mean", "dEN_XY", "dEN_XZ",
    "Z_X", "Z_Y", "Z_Z", "Z_R",
    "R_X", "R_Y", "R_Z", "R_R", "R_mean", "R_spread", "dR_XY", "dR_XZ",
    "n_atoms_fu", "n_magnetic_3d", "has_R_site", "n_unique_elements",
    "is_reduced", "SP_24", "SP_18",
]


def isomer_group(comp):
    """Element-multiset key: site-swap isomers (same elements, different site
    order, e.g. Co2CrAlGa vs Co2CrGaAl) map to the same group, so grouped CV
    and the hold-out never split an isomer pair across train/test."""
    toks = parse_composition(comp)
    agg = {}
    for e, n in toks:
        agg[e] = agg.get(e, 0) + n
    return "|".join(f"{e}{n}" for e, n in sorted(agg.items()))
# Tier B columns are one-hot; resolved at runtime:
def tier_b_cols(feat):
    return [c for c in feat.columns if c.startswith("type_") or c.startswith("sg_")] + ["SP_selected"]

# Tier C: outputs of the compound-specific DFT calculation (post-DFT)
TIER_C = ["formation_energy_eV_atom", "hull_distance_eV_atom", "total_magnetization_muB_fu"]

FAMILY_BLOCKS = {
    "Valence electrons": ["V_X", "V_Y", "V_Z", "V_R", "N_v", "dV_XY", "dV_XZ", "dV_YZ"],
    "Electronegativity": ["EN_X", "EN_Y", "EN_Z", "EN_R", "EN_mean", "dEN_XY", "dEN_XZ"],
    "Atomic number": ["Z_X", "Z_Y", "Z_Z", "Z_R"],
    "Atomic radii": ["R_X", "R_Y", "R_Z", "R_R", "R_mean", "R_spread", "dR_XY", "dR_XZ"],
    "Stoichiometry": ["n_atoms_fu", "n_magnetic_3d", "has_R_site",
                      "n_unique_elements", "is_reduced"],
    "Slater-Pauling": ["SP_24", "SP_18", "SP_selected"],
    # "Prototype spec" resolved at runtime (one-hot type + space group)
}


def load_prepared():
    df = pd.read_csv(f"{DATA_DIR}/prepared.csv")
    bool_cols = [c for c in df.columns if c.startswith("type_") or c.startswith("sg_")]
    df[bool_cols] = df[bool_cols].astype(int)
    return df


def make_models(random_state=RNG):
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LinearRegression, Ridge, Lasso
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.svm import SVR
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.neural_network import MLPRegressor
    from xgboost import XGBRegressor
    from lightgbm import LGBMRegressor

    def scaled(est):
        return Pipeline([("scaler", StandardScaler()), ("model", est)])

    return {
        "Linear Regression": scaled(LinearRegression()),
        "Ridge": scaled(Ridge(alpha=1.0)),
        "Lasso": scaled(Lasso(alpha=0.01, max_iter=20000)),
        "KNN": scaled(KNeighborsRegressor()),
        "SVR": scaled(SVR(kernel="rbf", C=10, epsilon=0.1)),
        "MLP (sklearn)": scaled(MLPRegressor(hidden_layer_sizes=(128, 64),
                                             max_iter=2000, random_state=random_state,
                                             early_stopping=True)),
        "Random Forest": RandomForestRegressor(n_estimators=300, random_state=random_state, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=300, random_state=random_state),
        "XGBoost": XGBRegressor(n_estimators=300, random_state=random_state,
                                verbosity=0, n_jobs=-1),
        "LightGBM": LGBMRegressor(n_estimators=300, random_state=random_state,
                                  verbose=-1, n_jobs=-1),
    }


def grouped_folds(groups, n_splits=5, n_repeats=3, seed=RNG):
    """Leave-composition-out grouped K-fold with repeats via permuted group assignment."""
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    for rep in range(n_repeats):
        perm = rng.permutation(ug)
        fold_of = {g: i % n_splits for i, g in enumerate(perm)}
        fold_idx = np.array([fold_of[g] for g in groups])
        for k in range(n_splits):
            te = np.where(fold_idx == k)[0]
            tr = np.where(fold_idx != k)[0]
            yield tr, te
