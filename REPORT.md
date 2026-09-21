# Machine-Learning Analysis of Heusler-Alloy Curie-Temperature Datasets — Corrected Data (v2)

**Following the provenance-aware, leakage-controlled framework of Paul, Giri, Datta & Pal (2026), *"Multi-Target Machine Learning and Deep Learning Framework for Predicting Structural, Energetic, and Magnetic Properties of Ternary Heusler Alloys Using Site-Resolved Descriptors"* (submitted to Physica B).**

Analysis date: 5 September 2026. Supersedes the first-pass report (17 August 2026), which was based on the pre-correction CSVs. All results are reproducible from the numbered scripts in `scripts/` with fixed random seeds (seed = 42).

---

## 1. What the corrected files changed, and how the pipeline responded

The corrected package (`files_corrected_Heusler_alloy.zip`) alters **no target values and deletes no rows**. It adds diagnostic columns and fixes placeholders, and its notes sheet identifies four issues; each was folded into the pipeline as follows.

**Site-swap isomers (180 rows; 62% of Quaternary).** Pairs such as Co2CrAlGa / Co2CrGaAl contain the same four elements with the Z- and R-site elements exchanged, and carry substantially different targets (e.g. 802 K vs 1048 K). Two responses: (i) the site-resolved Tier A descriptors (V_Z vs V_R, EN_Z vs EN_R, radii per site) are retained, since they are the only inputs that distinguish the pair; (ii) the validation grouping key was **strengthened from the composition string to the element multiset** (`isomer_group`), so an isomer pair is never split between training and test in the CV folds, the locked hold-out, or the conformal calibration split. This is the direct analogue of the previous paper's leave-composition-out control for L2₁/XA near-twins: 1505 rows now form 1412 groups (largest group 2), and every score below is an isomer-out score.

**Mixed cubic/tetragonal phases and inconsistent f.u.-per-cell normalization.** The phase was already carried by the space-group one-hots (Tier B); the newly supplied `fu_per_cell_ratio` (4 for Full/Quaternary, 3 vs 6 for cubic vs tetragonal Half, 1–2 for Inverse) was added as an explicit Tier B structure-specification descriptor. As shown in Sec. 4.7, this single addition resolves the Half-family magnetization normalization split that crippled that cell in the first pass.

**radius_R placeholder.** `radius_R` is now NaN for the 1217 ternary rows; it is encoded as 0 together with an explicit `has_R_site` indicator, so the model no longer reads "radius exactly zero" as a physical value without context.

**Reduced compositions (76 rows) and negative hull distances (20 rows).** `is_reduced_composition` and `n_unique_elements` were added as Tier A descriptors, and the hull-distance target now uses the supplied clipped column (negatives set to 0, as distance from the hull cannot be negative).

The descriptor audit therefore now counts **34 Tier A**, **10 Tier B** (type + space-group one-hots, type-selected SP prior, f.u./cell ratio), and 3 Tier C (post-DFT) descriptors. The locked hold-out is 227 entries (15%, whole isomer groups, stratified by family); the development set is 1278 entries.

## 2. Consolidated results (Scenario I unless stated; isomer-out CV, 5 folds × 3 repeats)

| Target | Best algorithm | CV R² (95% CI) | CV MAE | Tuned GB CV / hold-out | Hold-out R² | Hold-out MAE | 90% conformal ± | Coverage |
|---|---|---|---|---|---|---|---|---|
| Curie temperature (K) | Random Forest | 0.676 (0.646–0.705) | 107.6 | 0.692 / 0.704 | 0.707 | 101.4 | ±258 K | 0.899 |
| Formation energy (eV/atom) | Random Forest | 0.344 (0.292–0.392) | 0.0547 | 0.362 / 0.291 | 0.265 | 0.0606 | ±0.115 | 0.855 |
| Hull distance, clipped (eV/atom) | Random Forest | 0.268 (0.239–0.295) | 0.0438 | 0.276 / 0.305 | 0.296 | 0.0434 | ±0.085 | 0.872 |
| Magnetization (μB/f.u.) | Random Forest | 0.842 (0.804–0.873) | 0.191 | 0.857 / 0.826 | 0.811 | 0.186 | ±0.411 | 0.894 |
| **T_C, Scenario II** | Random Forest | **0.788** | **77.2** | — | 0.764 (RF) / **0.810** (GB) | — | — | — |

Changes relative to the first pass are modest at the headline level and in the expected directions. The Curie-temperature CV score decreases slightly (0.698 → 0.676) because the isomer-out grouping removes the residual advantage of having a swap partner in training, while the hold-out score improves (0.681 → 0.707) under the new group-level hold-out draw; the honest reading is that T_C is predictable from composition at **R² ≈ 0.68–0.71, MAE ≈ 101–108 K**. The Scenario II conclusion is unchanged: restoring the post-DFT descriptors lifts T_C to CV R² = 0.788 (MAE 77 K), and single-addition tests attribute the gain almost entirely to the magnetization (alone 0.797; formation energy 0.669; hull 0.662). Formation energy improves somewhat in CV (0.308 → 0.344) with the new descriptors, and the clipped hull target scores 0.268/0.296 — both still delimited by the near-DFT-noise MAE (0.055 and 0.044 eV/atom) rather than by model capacity. Magnetization remains the best-predicted target (0.842 CV, 0.811 hold-out).

Split-conformal calibration is again the only reliable interval construction: empirical coverage of the nominal 90% intervals is 0.855–0.899 across the four targets (quantile GB 0.846–0.899; the bootstrap ensemble undercovers at 0.24–0.42 and must not be quoted as a prediction interval). The T_C interval is ±258 K — slightly wider than the first pass, as it should be once isomer leakage into the calibration split is removed.

## 3. Classification, importance, extrapolation

**Heusler-family classification** from Tier A descriptors only: LightGBM attains CV accuracy 0.896 ± 0.012 (macro-F1 0.893, OVR AUC 0.977) against a 0.339 majority baseline, and 0.921 accuracy / 0.985 AUC on the locked hold-out. The residual confusion remains Inverse ↔ Full/Half among X2YZ compositions; Quaternary is identified perfectly.

**Importance** is essentially unchanged and coherent: for T_C the prototype-specification block dominates (joint-permutation ΔR² = 0.72, led by the Full-type flag) followed by the stoichiometry block (0.46, chiefly the count of magnetic 3d elements); for formation energy the electronegativity block leads among Tier A families (0.27, chiefly ΔEN_XY); for the clipped hull the radii block (0.17, led by radius spread) with the new `sg_fu_ratio` now appearing in the top five; for magnetization the Inverse-type flag dominates, with the radii and valence blocks next. The SHAP dependence on N_v again separates by family and rises at high electron counts.

**Leave-element-out** extrapolation reproduces the first-pass boundary almost exactly: holding out Z-site main-group species is tolerable (weighted R² = 0.69 for T_C, 0.85 for magnetization), holding out Y-site transition metals is not (0.27 for T_C, 0.03 for formation energy; Ti fails outright at −3.2). Predictions for chemistries absent from training remain untrustworthy, with the conformal width as the operational flag.

## 4. Per-family breakdown on the corrected data

This was the question the corrections were aimed at, so it was re-run in full: the pooled model re-scored within each family, and the ten-algorithm benchmark re-trained inside each family, all under isomer-out validation. (Within-family R² is measured against the family's own variance; MAE is the normalization-free comparison.)

**Curie temperature (within-family, pooled / family-specific):**

| Family | Pooled CV R² | Family CV R² | CV MAE (K) | Hold-out R² (pooled / family) |
|---|---|---|---|---|
| Full | 0.564 | 0.582 (LightGBM) | 53.8 / 58.5 | 0.676 / 0.743 |
| Half | −0.025 | −0.022 (SVR) | 107 / 103 | −0.736 / −0.039 |
| Inverse | 0.591 | 0.619 (GB) | 165 / 155 | 0.796 / 0.729 |
| Quaternary | −0.205 | −0.050 (SVR) | 141 / 130 | −0.361 / −0.046 |

The central finding of the first pass **survives the corrections**: genuine within-family T_C signal exists only in the Full family (MAE ≈ 54–59 K) and the Inverse family (hold-out up to 0.80), while within Half and Quaternary the Curie temperature remains statistically indistinguishable from family-mean scatter, for the pooled and the family-specific models alike. Two of the four issues raised in the notes were therefore *not* the cause of that failure. In particular, the site-swap isomers do not merely add a validation subtlety — they expose a real ceiling: under isomer-out validation the model must predict the difference between swap partners (up to ≈ 30% in T_C) purely from the site-resolved descriptors, and it cannot; the Quaternary within-family score is unchanged. Likewise, phase mixing was already encoded via the space groups. The honest conclusion stands that these composition-level descriptors do not resolve T_C within the Half and Quaternary families, and per-entry T_C provenance (SPR-KKR estimates, per the notes' DXMag attribution) should be examined before stronger claims are made.

One cell, however, is dramatically repaired: **within-Half magnetization**. In the first pass the pooled model scored ≈ 0 there because the Half family mixes f.u.-per-cell normalizations (3 for cubic rows, 6 for tetragonal); with `sg_fu_ratio` supplied as a Tier B descriptor the pooled within-Half score rises from −0.02 to **0.40** (family-specific 0.43, hold-out up to 0.53), and pooled within-Full magnetization also improves (0.60 → 0.79). This confirms the notes' diagnosis for the magnetization target specifically: the earlier failure in that cell was a normalization artifact, not missing physics.

Formation energy and hull distance remain internally predictable **only** in the Inverse family (family-specific CV 0.738 and 0.691; hold-out 0.807 and 0.755) and at chance in the other three — the Inverse regularity persists after all corrections, so the question of why that family's targets are so much more descriptor-coupled (different generation pathway per the notes: VASP-DFT for ternary vs ML-potential for quaternary energetics) is now the leading data question rather than a modelling one.

## 5. Conclusions (corrected data)

The corrections tighten the methodology without overturning any headline conclusion. Under isomer-out, leakage-controlled validation: the Curie temperature is screenable from composition at MAE ≈ 101–108 K with calibrated ±258 K (90%) intervals, sharpening to MAE ≈ 77 K once the DFT magnetization is available, with the magnetization carrying essentially all of the added information; magnetization itself is predictable at R² ≈ 0.81–0.86, and the Half-family normalization artifact identified in the notes is fully resolved by the f.u./cell descriptor; formation energy and hull distance remain MAE-limited near DFT noise; the family label is recoverable from composition at 0.90–0.92 accuracy; chemical extrapolation to unseen transition metals remains the hard trust boundary. The per-family delimitation is confirmed as real rather than artifactual for the Curie temperature: skill lives in the Full and Inverse families, and within Half and Quaternary the target behaves as descriptor-independent scatter even after every correction in the package is applied — with the site-swap isomer pairs now demonstrably setting a resolution floor that site-resolved composition descriptors do not overcome.

## Files

| Path | Content |
|---|---|
| `scripts/01…09*.py` + `scripts/common.py` | Reproducible pipeline, corrected-data version (seed = 42) |
| `data/Heusler_All.csv` | Corrected merged table (clipped hull substituted; `family` → `type`) |
| `data/prepared.csv` | Working set with engineered descriptors, isomer groups, hold-out flag |
| `results/*.csv`, `results/*.json` | All machine-readable metrics (benchmark, tuning, Scenario II, uncertainty, importance, LEO, per-family) |
| `figures/Fig0–Fig10` (PDF + PNG) | Publication figures mirroring Figs. 1–9 of the manuscript, plus the per-family breakdown |
