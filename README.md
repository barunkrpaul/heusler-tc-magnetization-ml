# Heusler Alloy Machine Learning Pipeline

Predicts the Curie temperature ($T_C$), total magnetization, formation energy and hull distance of Heusler alloys from composition-level descriptors, using a leakage-controlled, isomer-group-aware validation protocol. This is the corrected-data (v2) run: it reproduces, on the corrected Half/Full/Inverse/Quaternary datasets (isomer-out validation, clipped hull-distance target, formula-unit-per-cell descriptor), the provenance-tiered framework of Paul, Giri, Datta & Pal, *"Multi-Target Machine Learning and Deep Learning Framework for Predicting Structural, Energetic, and Magnetic Properties of Ternary Heusler Alloys Using Site-Resolved Descriptors"* (submitted to *Physica B*).

**Scope note.** This pipeline computes all four targets end to end and `REPORT.md` documents all four. The manuscript drafted from this work is restricted to Curie temperature and magnetization only, since formation energy and hull distance did not reach a level of predictive performance worth reporting there. Anyone using this repository as a reference for the paper should read only the $T_C$ and magnetization results as representative of the submitted claims; the other two targets are kept here for completeness and transparency of the full pipeline.

## What the pipeline does

1. Merges the four family-specific CSVs plus the corrected diagnostic columns into one working table, audits descriptor provenance into three tiers (Tier A: composition-level, available pre-DFT; Tier B: structure specification, valid for regression only; Tier C: post-DFT outputs), and separates a locked 15% hold-out set drawn as whole isomer groups.
2. Benchmarks ten regression algorithms (Linear Regression, Ridge, Lasso, KNN, SVR, MLP, Random Forest, Gradient Boosting, XGBoost, LightGBM) under grouped cross-validation, with the isomer group (defined by element multiset) as the grouping key, so that site-swap isomer pairs never split across train and test.
3. Applies randomized hyperparameter search to Gradient Boosting only, and evaluates a second scenario for $T_C$ that adds the DFT magnetization as an extra input.
4. Builds 90% prediction intervals three ways (split-conformal, quantile Gradient Boosting, bootstrap ensemble) and measures their empirical coverage on hold-out.
5. Computes descriptor importance by permutation, by collinearity-robust block permutation, and by TreeSHAP.
6. Runs leave-element-out validation to test extrapolation to chemical elements absent from training.
7. Repeats the whole comparison within each structural family (pooled model scored per family vs. family-specific models), to separate between-family separation from genuine within-family resolution.

See `REPORT.md` for the full write-up of results, including the per-family breakdown and the corrections applied relative to the first-pass (pre-correction) run.

## Requirements

```
pip install pandas numpy scikit-learn xgboost lightgbm shap matplotlib
```

Python 3.11 was used for the original run; the scripts have no version-specific syntax and should run on any reasonably recent Python 3.

## Running

From the repository root, in order:

```
python scripts/01_prepare_data.py                       # merge, audit, Tier A/B/C descriptors, locked 15% hold-out
python scripts/02_benchmark.py                           # 10 algorithms, isomer-group-out CV, hold-out scored once
python scripts/03_tuning_scenario2_classification.py     # tuned GB, Scenario II (T_C + DFT magnetization), family classification
python scripts/04_uncertainty.py                         # split-conformal / quantile-GB / bootstrap 90% intervals
python scripts/05_importance.py                          # permutation, block permutation, TreeSHAP
python scripts/06_leave_element_out.py                    # chemical extrapolation (Y-site and Z-site elements)
python scripts/07_figures.py                              # Fig0-Fig9 (PDF + PNG)
python scripts/08_per_family.py                           # per-family breakdown (pooled vs. family-specific)
python scripts/09_family_figures.py                       # Fig10
```

All stages use a fixed random seed (42) for reproducibility. `scripts/common.py` holds the shared elemental property tables, descriptor construction, and cross-validation helpers used by every stage.

## Repository layout

```
scripts/    the nine numbered pipeline stages plus common.py (shared utilities)
data/       Heusler_All.csv (merged raw input) and prepared.csv (after descriptor construction and hold-out split)
results/    JSON/CSV outputs of every stage (benchmark scores, uncertainty, importance, leave-element-out, per-family)
figures/    Fig0-Fig10, each as vector PDF and PNG
README.md   this file
REPORT.md   full narrative write-up of the corrected-data (v2) analysis
```

## Data provenance

The dataset's column layout and units are consistent with the computational Heusler database of Xiao and Tadano. This attribution is not yet independently confirmed against the original data release and should be verified before the results are cited or submitted.

## Citation

If this pipeline or its results are used, please cite the associated manuscript (Paul, Das, Giri, De, Datta & Pal, submitted) once it is available, and the original dataset source once its provenance is confirmed.
