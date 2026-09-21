"""Stage 1 (corrected data): merge, audit, feature-engineer, and lock the
isomer-group-aware composition hold-out.

Changes vs the first-pass pipeline, following the corrected files' notes:
- hull_distance target/descriptor uses the clipped column (negatives -> 0);
- radius_R NaN (ternary rows) encoded as 0 + has_R_site indicator;
- is_reduced_composition and n_unique_elements added as Tier A descriptors;
- fu_per_cell_ratio added as a Tier B structure-specification descriptor
  (stored as `sg_fu_ratio` so it joins the prototype-spec block);
- grouping key for CV and hold-out is the element multiset (isomer_group),
  so site-swap isomer pairs (180 rows) never straddle a train/test split.
"""
import sys, json
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from common import (build_features, tier_b_cols, TIER_A, TIER_C, TARGETS,
                    parse_composition, site_slots, isomer_group, RNG)

raw = pd.read_csv("data/Heusler_All.csv")
audit = {"raw_records": len(raw)}

# --- integrity checks -------------------------------------------------
assert raw["composition"].nunique() == len(raw), "duplicate compositions"
audit["missing_values_targets"] = int(raw[list(TARGETS)].isna().sum().sum())
audit["negative_hull_rows_clipped"] = int(raw["hull_distance_flag_negative"].sum())
audit["site_swap_isomer_rows"] = int(raw["is_site_swap_isomer"].sum())
audit["reduced_composition_rows"] = int(raw["is_reduced_composition"].sum())

# verify parsed site elements consistent with tabulated radii
mismatch = 0
for _, r in raw.iterrows():
    slots = site_slots(parse_composition(r["composition"]))
    for slot, col in zip(slots, ["radius_X", "radius_Y", "radius_Z", "radius_R"]):
        if slot is None and (pd.notna(r[col]) and r[col] > 0):
            mismatch += 1
audit["site_parse_radius_mismatches"] = mismatch

# --- feature engineering ---------------------------------------------
feat = build_features(raw)
extra = raw[["composition", "is_reduced_composition", "fu_per_cell_ratio"]].copy()
extra["is_reduced"] = extra["is_reduced_composition"].astype(int)
df = feat.merge(extra[["composition", "is_reduced", "fu_per_cell_ratio"]],
                on="composition", validate="1:1")
df = df.merge(raw[["composition"] + list(TARGETS) + ["total_magnetization_muB_cell"]],
              on="composition", validate="1:1")

# Tier B: f.u.-per-cell ratio (structure/normalization specification).
# One row has NaN (zero-moment compound); impute with the mode of its
# (type, space-group) cell, else the global mode.
key_cols = [c for c in df.columns if c.startswith("type_") or c.startswith("sg_")]
if df["fu_per_cell_ratio"].isna().any():
    glob_mode = df["fu_per_cell_ratio"].mode()[0]
    for i in df.index[df["fu_per_cell_ratio"].isna()]:
        same = (df[key_cols] == df.loc[i, key_cols]).all(axis=1)
        vals = df.loc[same, "fu_per_cell_ratio"].dropna()
        df.loc[i, "fu_per_cell_ratio"] = vals.mode()[0] if len(vals) else glob_mode
df = df.rename(columns={"fu_per_cell_ratio": "sg_fu_ratio"})

# --- isomer group + locked hold-out (15% of rows, whole groups, by type) ---
df["isomer_group"] = df["composition"].map(isomer_group)
audit["isomer_groups"] = int(df["isomer_group"].nunique())
audit["largest_isomer_group"] = int(df.groupby("isomer_group").size().max())

rng = np.random.default_rng(RNG)
holdout_mask = np.zeros(len(df), bool)
types = ["Full", "Half", "Inverse", "Quaternary"]
df["_type"] = np.select([df[f"type_{t}"] for t in types], types, default="Full")
for t in types:
    sub = df[df["_type"] == t]
    target_n = int(round(0.15 * len(sub)))
    groups = np.array(sub["isomer_group"].unique(), dtype=object)
    rng.shuffle(groups)
    taken = 0
    for g in groups:
        idx = df.index[(df["isomer_group"] == g) & (df["_type"] == t)]
        # a group spanning types is assigned wholly when first encountered
        idx_all = df.index[df["isomer_group"] == g]
        if holdout_mask[idx_all].any():
            continue
        if taken >= target_n:
            break
        holdout_mask[idx_all] = True
        taken += len(idx)
df["is_holdout"] = holdout_mask
df = df.drop(columns=["_type"])

df.to_csv("data/prepared.csv", index=False)

audit.update({
    "final_records": len(df),
    "n_descriptors_tierA": len(TIER_A),
    "n_descriptors_tierB": len([c for c in df.columns
                                if c.startswith("type_") or c.startswith("sg_")]) + 1,
    "holdout_entries": int(holdout_mask.sum()),
    "development_entries": int((~holdout_mask).sum()),
    "per_type": raw["type"].value_counts().to_dict(),
})
with open("results/audit.json", "w") as f:
    json.dump({"audit": audit,
               "descriptor_tiers": {"Tier A": TIER_A,
                                    "Tier B": tier_b_cols(feat) + ["sg_fu_ratio"],
                                    "Tier C": TIER_C}}, f, indent=2)
print(json.dumps(audit, indent=2))
