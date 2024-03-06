from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, f1_score
import numpy as np
import pandas as pd
import os
from imblearn.ensemble import BalancedRandomForestClassifier

from actinet.models import ActivityClassifier
from actinet.hmm import HMM
from actinet.utils.utils import safe_indexer

WINSEC = 30


def evaluate_preprocessing(
    classifier: ActivityClassifier,
    X,
    Y,
    groups=None,
    T=None,
    weights_path="models/weights.pt",
    verbose=True,
):
    skf = StratifiedGroupKFold(n_splits=5)

    le = LabelEncoder().fit(Y)
    Y_encoded = le.transform(Y)

    Y_preds = np.empty_like(Y_encoded)

    for fold, (train_index, test_index) in enumerate(skf.split(X, Y_encoded, groups)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y_encoded[train_index], Y_encoded[test_index]
        groups_train = safe_indexer(groups, train_index)
        t_train = safe_indexer(T, train_index)

        classifier.fit(
            X_train,
            y_train,
            groups_train,
            t_train,
            weights_path.format(fold),
            n_splits=1,
        )
        y_pred = classifier.predict(X_test, False)

        if verbose:
            print(
                f"Fold {fold+1} Test Scores - Accuracy: {accuracy_score(y_test, y_pred):.3f}, "
                + f"Macro F1: {f1_score(y_test, y_pred, average='macro'):.3f}"
            )

        Y_preds[test_index] = y_pred

    Y_preds = le.inverse_transform(Y_preds)

    if verbose:
        print(classification_report(Y, Y_preds))

    return Y_preds


def evaluate_models(
    actinet_classifier: ActivityClassifier,
    rf_classifier: BalancedRandomForestClassifier,
    X_actinet,
    X_rf,
    Y_actinet,
    Y_rf,
    groups_actinet,
    groups_rf,
    T_actinet=None,
    T_rf=None,
    weights_path="models/weights.pt",
    out_dir=None,
    verbose=True,
):
    skf = StratifiedGroupKFold(n_splits=5)

    le = LabelEncoder().fit(Y_rf)
    Y_encoded_rf = le.transform(Y_rf)
    Y_encoded_actinet = le.transform(Y_actinet)

    Y_preds_rf = np.empty_like(Y_encoded_rf)
    Y_preds_actinet = np.empty_like(Y_encoded_actinet)

    results_rf = []
    results_actinet = []

    for fold, (train_index_rf, test_index_rf) in enumerate(
        skf.split(X_rf, Y_encoded_rf, groups_rf)
    ):
        # Ensure the same train and test split for groups are used in both models in each fold
        train_index_actinet = np.isin(
            groups_actinet, np.unique(groups_rf[train_index_rf])
        )
        test_index_actinet = np.isin(
            groups_actinet, np.unique(groups_rf[test_index_rf])
        )

        # Analysis of actinet model
        X_train_actinet, X_test_actinet = (
            X_actinet[train_index_actinet],
            X_actinet[test_index_actinet],
        )
        y_train_actinet, y_test_actinet = (
            Y_encoded_actinet[train_index_actinet],
            Y_encoded_actinet[test_index_actinet],
        )
        groups_train_actinet = groups_actinet[train_index_actinet]
        groups_test_actinet = groups_actinet[test_index_actinet]

        t_train_actinet = safe_indexer(T_actinet, train_index_actinet)
        t_test_actinet = safe_indexer(T_actinet, test_index_actinet)

        actinet_classifier.fit(
            X_train_actinet,
            y_train_actinet,
            groups_train_actinet,
            t_train_actinet,
            weights_path.format(fold),
            n_splits=5,
        )
        y_pred_actinet = actinet_classifier.predict(
            X_test_actinet, True, t_test_actinet
        )

        # Analysis of accelerometer random forest model
        X_train_rf, X_test_rf = X_rf[train_index_rf], X_rf[test_index_rf]
        y_train_rf, y_test_rf = (
            Y_encoded_rf[train_index_rf],
            Y_encoded_rf[test_index_rf],
        )

        t_train_rf = safe_indexer(T_rf, train_index_rf)
        t_test_rf = safe_indexer(T_rf, test_index_rf)

        groups_test_rf = groups_rf[test_index_rf]

        rf_classifier.fit(
            X_train_rf,
            y_train_rf,
        )

        hmm_rf = HMM()
        hmm_rf.fit(
            rf_classifier.oob_decision_function_,
            y_train_rf,
            t_train_rf,
            WINSEC,
        )
        y_pred_rf = hmm_rf.predict(rf_classifier.predict(X_test_rf), t_test_rf, WINSEC)

        # Display model performance for each fold
        if verbose:
            print(
                f"Actinet test scores for fold {fold+1}\n"
                + f"Accuracy: {accuracy_score(y_test_actinet, y_pred_actinet):.3f}, "
                + f"Macro F1: {f1_score(y_test_actinet, y_pred_actinet, average='macro'):.3f}"
            )
            print(
                f"Accelerometer test scores for fold {fold+1}\n"
                + f"Accuracy: {accuracy_score(y_test_rf, y_pred_rf):.3f}, "
                + f"Macro F1: {f1_score(y_test_rf, y_pred_rf, average='macro'):.3f}"
            )

        Y_preds_actinet[test_index_actinet] = y_pred_actinet
        Y_preds_rf[test_index_rf] = y_pred_rf

        results_actinet = results_actinet.append(
            {
                "fold": [fold] * len(test_index_actinet),
                "group": groups_test_actinet,
                "Y_pred": le.inverse_transform(y_pred_actinet),
                "Y_true": le.inverse_transform(y_test_actinet),
            }
        )
        results_rf = results_rf.append(
            {
                "fold": [fold] * len(test_index_rf),
                "group": groups_test_rf,
                "Y_pred": y_pred_rf,
                "Y_true": y_test_rf,
            }
        )

    Y_preds_actinet = le.inverse_transform(Y_preds_actinet)
    Y_preds_rf = le.inverse_transform(Y_preds_rf)

    # Report performance across all folds
    if verbose:
        print("Actinet performance:")
        print(classification_report(Y_rf, Y_preds_rf))
        print("Accelerometer performance:")
        print(classification_report(Y_rf, Y_preds_rf))

    # Save results to CSV files
    results_actinet = pd.DataFrame(results_actinet)
    results_rf = pd.DataFrame(results_rf)

    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        results_actinet.to_csv(f"{out_dir}/actinet_results.csv", index=False)
        results_rf.to_csv(f"{out_dir}/rf_results.csv", index=False)

    return results_actinet, results_rf
