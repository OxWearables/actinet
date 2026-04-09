import numpy as np
import pandas as pd
import re
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    cohen_kappa_score,
    balanced_accuracy_score,
)


class DivDict(dict):
    """Dictionary subclass that allows division by a number."""

    def __truediv__(self, n):
        if not isinstance(n, (int, float)):
            raise TypeError("Can only divide by a number (int or float)")
        return DivDict({k: v / n for k, v in self.items()})


def calculate_metrics(y_true, y_pred):
    """Calculates accuracy, F1, Cohen's Kappa, and balanced accuracy."""
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    kappa = cohen_kappa_score(y_true, y_pred)
    bacc = balanced_accuracy_score(y_true, y_pred)
    return accuracy, f1, kappa, bacc


def extract_activity_predictions(
    results: pd.DataFrame, activity, age_band=None, sex=None, return_true_labels=False
):
    """Extracts incidence of predicted activity label for actinet and accelerometer based on filtering conditions."""
    model_results = results.copy()
    if age_band is not None:
        model_results = model_results[model_results["Age Band"] == age_band]
    if sex is not None:
        model_results = model_results[model_results["Sex"] == sex]

    activity_bbaa_pred = np.array(
        [
            x.get(activity, 0)
            for x in model_results.loc[
                model_results["Model"] == "accelerometer", "Pred_dict"
            ]
        ]
    )
    activity_actinet_pred = np.array(
        [
            x.get(activity, 0)
            for x in model_results.loc[model_results["Model"] == "actinet", "Pred_dict"]
        ]
    )

    population = len(model_results["Participant"].unique())

    if return_true_labels:
        activity_bbaa_true = np.array(
            [
                x.get(activity, 0)
                for x in model_results.loc[
                    model_results["Model"] == "accelerometer", "True_dict"
                ]
            ]
        )
        activity_actinet_true = np.array(
            [
                x.get(activity, 0)
                for x in model_results.loc[
                    model_results["Model"] == "actinet", "True_dict"
                ]
            ]
        )
        return (
            activity_bbaa_pred,
            activity_actinet_pred,
            activity_bbaa_true,
            activity_actinet_true,
            population,
        )

    else:
        return activity_bbaa_pred, activity_actinet_pred, population


def build_mae_cell(true_values, pred_values):
    """Builds a MAE cell for a given set of true and predicted values."""
    mae = np.abs(true_values - pred_values)
    return f"{np.mean(mae):.3f} ± {np.std(mae):.3f}"


def build_pvalue_cell(true_values, pred_values):
    """Builds a p-value cell for a given set of true and predicted values."""
    _, p_value = stats.ttest_rel(true_values, pred_values)
    if p_value < 0.001:
        return "<0.001"
    return f"{p_value:.3f}"


def build_mae_table(df: pd.DataFrame, activities):
    df_maes = pd.DataFrame(
        columns=activities, index=["Baseline", "ActiNet", "p-value"], dtype=float
    )
    for activity in activities:
        (
            activity_bbaa_pred,
            activity_actinet_pred,
            activity_bbaa_true,
            activity_actinet_true,
            _,
        ) = extract_activity_predictions(df, activity, return_true_labels=True)
        df_maes.loc["Baseline", activity] = build_mae_cell(
            activity_bbaa_true, activity_bbaa_pred
        )
        df_maes.loc["ActiNet", activity] = build_mae_cell(
            activity_actinet_true, activity_actinet_pred
        )
        df_maes.loc["p-value", activity] = build_pvalue_cell(
            activity_bbaa_true, activity_bbaa_pred
        )
    return df_maes


def convert_version(s: str) -> str:
    """Converts a version string from 'x.y.z+number' to 'vX-Y-Z'."""
    match = re.match(r"(\d+)\.(\d+)\.(\d+)(?:\+(\d+))?", s)
    if not match:
        raise ValueError(f"String '{s}' is not in the expected format")

    major, minor, patch, _ = match.groups()
    base = f"v{major}-{minor}-{patch}"
    return base
