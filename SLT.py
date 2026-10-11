"""Can a voice model work for a new patient?

Run: python SLT.py
Data: UCI Parkinsons Telemonitoring, DOI 10.24432/C5ZS3N, CC BY 4.0.
The target is interpolated motor UPDRS, not a diagnosis or a future score.
"""

from pathlib import Path
import argparse
import hashlib
import io
import json
import platform
import urllib.request
import zipfile

import matplotlib
matplotlib.use("Agg")  # Save figures without requiring a desktop display.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
DATA_URL = "https://archive.ics.uci.edu/static/public/189/parkinsons+telemonitoring.zip"
FEATURES = [
    "Jitter(%)", "Jitter(Abs)", "Jitter:RAP", "Jitter:PPQ5", "Jitter:DDP",
    "Shimmer", "Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5",
    "Shimmer:APQ11", "Shimmer:DDA", "NHR", "HNR", "RPDE", "DFA", "PPE",
]


# 1. Load and inspect. A local copy is included; download only if it is absent.
def load_data(data_dir):
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / "parkinsons_updrs.data"
    if not path.exists():
        with urllib.request.urlopen(DATA_URL, timeout=60) as response:
            archive_bytes = response.read()
        with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
            for name in ["parkinsons_updrs.data", "parkinsons_updrs.names"]:
                (data_dir / name).write_bytes(archive.read(name))
    df = pd.read_csv(path)
    # Stop rather than silently clean a different or damaged dataset.
    assert df.shape == (5875, 22), "Unexpected dataset shape."
    assert df["subject#"].nunique() == 42, "Unexpected patient count."
    assert not df.isna().any().any(), "Missing values need an explicit policy."
    assert np.isfinite(df.to_numpy()).all(), "Non-finite data found."
    assert not df.duplicated().any(), "Duplicate rows need review."
    return df, hashlib.sha256(path.read_bytes()).hexdigest()


# 2. Define the question before fitting any models.
# Use voice features only. ID is for grouping, not prediction. Leave out age,
# sex and time to keep the question focused. total_UPDRS is another clinical
# score and would give away information about motor_UPDRS.
def define_task(df):
    return df[FEATURES], df["motor_UPDRS"], df["subject#"].astype(int)


def patient_errors(y, prediction, groups):
    """Give every patient equal weight, regardless of recording count."""
    errors = pd.DataFrame({
        "patient": np.asarray(groups),
        "absolute_error": np.abs(np.asarray(y) - np.asarray(prediction)),
    })
    return errors.groupby("patient")["absolute_error"].mean()


def summarize(y, prediction, groups):
    return {
        "patient_mae": float(patient_errors(y, prediction, groups).mean()),
        "recording_mae": float(mean_absolute_error(y, prediction)),
        "rmse": float(np.sqrt(mean_squared_error(y, prediction))),
        "r2": float(r2_score(y, prediction)),
    }


# 3. Compare a small, pre-specified set of models. We are choosing a model,
# not searching dozens of settings until the test result looks impressive.
def candidate_models():
    return {
        "Mean baseline": DummyRegressor(strategy="mean"),
        # A pipeline fits scaling on the training fold only. Correlated voice
        # measures motivate ridge's regularization; alpha=10 is fixed in advance.
        "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
        # Leaf size limits memorization. The forest needs no feature scaling.
        "Random forest": RandomForestRegressor(
            n_estimators=200, min_samples_leaf=5, random_state=SEED, n_jobs=1,
        ),
    }


# 4. Select using development patients only. Each out-of-fold prediction is
# made by a model that has never seen any recordings from that patient.
def compare_on_development(X, y, groups, models):
    folds = GroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    rows = []
    for name, model in models.items():
        prediction = np.full(len(y), np.nan)
        for train, valid in folds.split(X, y, groups):
            assert set(groups.iloc[train]).isdisjoint(groups.iloc[valid])
            fitted = clone(model).fit(X.iloc[train], y.iloc[train])
            prediction[valid] = fitted.predict(X.iloc[valid])
        assert np.isfinite(prediction).all()
        rows.append({"model": name, **summarize(y, prediction, groups)})
    # Mean of the 33 patient-specific errors is the selection metric. These
    # development results guide selection; they are not the final test estimate.
    return pd.DataFrame(rows).sort_values("patient_mae").reset_index(drop=True)


def bootstrap_patient_mean(errors):
    """Resample patients, not correlated recordings, for a rough 95% interval."""
    values = np.asarray(errors)
    rng = np.random.default_rng(SEED)
    means = rng.choice(values, size=(2000, len(values)), replace=True).mean(axis=1)
    return np.quantile(means, [0.025, 0.975]).tolist()


def save_figures(results, cv, patient_mae, predictions, out):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(cv["model"], cv["patient_mae"], color=["#c15b3d", "#286777", "#aaa49a"])
    ax.invert_yaxis()
    ax.set_xlabel("Mean patient MAE (motor UPDRS points); lower is better")
    ax.set_title("Model selection: development patients only")
    fig.tight_layout(); fig.savefig(out / "development_comparison.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(np.arange(len(patient_mae)), patient_mae.to_numpy(), color="#286777")
    ax.axhline(patient_mae.mean(), color="#c15b3d", linestyle="--", label="Patient mean")
    ax.set(xticks=np.arange(len(patient_mae)),
           xticklabels=[str(v) for v in patient_mae.index],
           xlabel="Held-out patient ID", ylabel="MAE (motor UPDRS points)",
           title="Averages hide differences between patients")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(out / "patient_errors.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(predictions["actual"], predictions["predicted"], s=12, alpha=.25,
               color="#286777", edgecolors="none")
    lo = min(predictions[["actual", "predicted"]].min())
    hi = max(predictions[["actual", "predicted"]].max())
    ax.plot([lo, hi], [lo, hi], color="#c15b3d", linestyle="--")
    ax.set(xlabel="Recorded target (interpolated motor UPDRS)",
           ylabel="Predicted motor UPDRS", title="Unseen-patient test predictions")
    fig.tight_layout(); fig.savefig(out / "test_predictions.png", dpi=180)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args(argv)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    df, checksum = load_data(args.data_dir)
    X, y, groups = define_task(df)

    # GroupShuffleSplit's 20% refers to patients, not recordings: 9 test,
    # 33 development patients. Never choose the seed based on test performance.
    split = GroupShuffleSplit(n_splits=1, test_size=.20, random_state=SEED)
    dev, test = next(split.split(X, y, groups))
    assert set(groups.iloc[dev]).isdisjoint(groups.iloc[test])
    models = candidate_models()
    cv = compare_on_development(X.iloc[dev], y.iloc[dev], groups.iloc[dev], models)
    selected_name = cv.loc[0, "model"]
    selected = clone(models[selected_name]).fit(X.iloc[dev], y.iloc[dev])

    # 5. Final evaluation happens after selection. Compare against the same
    # training-mean baseline on the same untouched test patients.
    prediction = selected.predict(X.iloc[test])
    baseline = clone(models["Mean baseline"]).fit(X.iloc[dev], y.iloc[dev])
    baseline_prediction = baseline.predict(X.iloc[test])
    test_metrics = summarize(y.iloc[test], prediction, groups.iloc[test])
    baseline_metrics = summarize(y.iloc[test], baseline_prediction, groups.iloc[test])
    errors = patient_errors(y.iloc[test], prediction, groups.iloc[test])
    interval = bootstrap_patient_mean(errors)
    predictions = pd.DataFrame({
        "row": test, "patient": groups.iloc[test].to_numpy(),
        "actual": y.iloc[test].to_numpy(), "predicted": prediction,
        "baseline_prediction": baseline_prediction,
    })

    # 6. Additional diagnostic: contrast the fixed random forest under the
    # two split methods. This does not alter the selected baseline or tune
    # any forest settings. Patient overlap makes the row split an easier,
    # different question: more recordings from already represented patients.
    # The two test sets differ, so their error gap is not a controlled causal
    # estimate of leakage. Do not use this demonstration for model selection.
    row_train, row_test = train_test_split(np.arange(len(df)), test_size=.20,
                                          random_state=SEED)
    forest = clone(models["Random forest"]).fit(X.iloc[dev], y.iloc[dev])
    forest_test_metrics = summarize(y.iloc[test], forest.predict(X.iloc[test]),
                                    groups.iloc[test])
    row_model = clone(models["Random forest"]).fit(X.iloc[row_train], y.iloc[row_train])
    row_metrics = summarize(y.iloc[row_test], row_model.predict(X.iloc[row_test]),
                            groups.iloc[row_test])
    overlap = sorted(set(groups.iloc[row_train]) & set(groups.iloc[row_test]))

    results = {
        "question": "Can voice measurements estimate motor UPDRS for an unseen patient?",
        "seed": SEED, "data_sha256": checksum,
        "recordings": len(df), "patients": int(groups.nunique()),
        "features": FEATURES, "target": "motor_UPDRS (linearly interpolated)",
        "development_rows": len(dev), "test_rows": len(test),
        "development_patients": sorted(groups.iloc[dev].unique().tolist()),
        "test_patients": sorted(groups.iloc[test].unique().tolist()),
        "selected_model": selected_name,
        "development_cv": cv.to_dict(orient="records"),
        "test": test_metrics, "test_baseline": baseline_metrics,
        "fixed_forest_test": forest_test_metrics,
        "patient_mae_95pct_bootstrap_interval": interval,
        "patient_errors": [{"patient": int(k), "mae": float(v)} for k, v in errors.items()],
        "random_recording_split": {**row_metrics, "overlapping_patients": overlap,
                                    "test_rows": len(row_test)},
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "scikit-learn": sklearn.__version__,
                     "matplotlib": matplotlib.__version__},
    }
    (out / "metrics.json").write_text(json.dumps(results, indent=2) + "\n")
    cv.to_csv(out / "development_comparison.csv", index=False)
    predictions.to_csv(out / "test_predictions.csv", index=False)
    errors.rename("mae").to_csv(out / "patient_errors.csv")
    # Save membership so peers can confirm that subjects never cross splits.
    pd.DataFrame({"row": np.arange(len(df)), "patient": groups,
                  "split": np.where(np.isin(np.arange(len(df)), test), "test", "development")
                  }).to_csv(out / "split_membership.csv", index=False)
    save_figures(results, cv, errors, predictions, out)
    print(cv[["model", "patient_mae"]].to_string(index=False))
    print(f"\nSelected: {selected_name}")
    print(f"Unseen-patient MAE: {test_metrics['patient_mae']:.2f} points")
    print(f"Same-test baseline MAE: {baseline_metrics['patient_mae']:.2f} points")
    print(f"Patient bootstrap 95% interval: {interval[0]:.2f} to {interval[1]:.2f}")
    print(f"Fixed forest, unseen-patient MAE: {forest_test_metrics['patient_mae']:.2f} points")
    print(f"Fixed forest, random-recording MAE: {row_metrics['patient_mae']:.2f} points")
    print(f"Patient overlap: group split 0; recording split {len(overlap)}")
    return results


if __name__ == "__main__":
    main()
