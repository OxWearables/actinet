from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, f1_score
import numpy as np

from actinet.models import ActivityClassifier
from actinet.utils.utils import safe_indexer


def evaluate(
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
