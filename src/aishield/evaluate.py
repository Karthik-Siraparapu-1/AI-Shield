import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)


def score_of(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    d = model.decision_function(X)
    return 1 / (1 + np.exp(-d))


def metrics(model, X, y, threshold=0.5):
    s = score_of(model, X)
    p = (s >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
    return {
        "n": int(len(y)), "positives": int(np.sum(y)),
        "accuracy": round(accuracy_score(y, p), 4),
        "precision": round(precision_score(y, p, zero_division=0), 4),
        "recall": round(recall_score(y, p, zero_division=0), 4),
        "f1": round(f1_score(y, p, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y, s), 4) if len(set(y)) > 1 else None,
        "pr_auc": round(average_precision_score(y, s), 4) if len(set(y)) > 1 else None,
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }
