"""Train and compare models for all three AI Shield modules.

    python train.py            # trains everything, writes models/*.joblib and reports/metrics.json
"""
import json
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from src.aishield.config import MODEL_DIR, RANDOM_STATE, RAW_DIR, RAW_FILES, REPORT_DIR
from src.aishield.evaluate import metrics
from src.aishield.text_clean import clean_text
from src.aishield.url_features import BASIC12, UCI8

warnings.filterwarnings("ignore")
RS = RANDOM_STATE
MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)
RESULTS = {}


def raw(key, **kw):
    return pd.read_csv(RAW_DIR / RAW_FILES[key], **kw)


def log(*a):
    print(*a, flush=True)


# ----------------------------------------------------------------------------------------
# MODULE 1 - Scam text (Composite Scam Transcript dataset)
# ----------------------------------------------------------------------------------------
def train_text():
    log("\n=== MODULE 1: scam text ===")
    tr, te = raw("composite_train"), raw("composite_test")
    audit = {"train_rows_raw": len(tr), "test_rows_raw": len(te),
             "train_duplicates": int(tr.duplicated().sum()), "test_duplicates": int(te.duplicated().sum()),
             "test_texts_also_in_train": int(te.text.isin(set(tr.text)).sum())}
    tr = tr.dropna().drop_duplicates()
    conflict = tr.groupby("text").label.nunique()
    tr = tr[~tr.text.isin(conflict[conflict > 1].index)]
    te = te.dropna().drop_duplicates()
    te = te[~te.text.isin(set(tr.text))]                       # remove leakage
    audit.update(train_rows_clean=len(tr), test_rows_clean=len(te),
                 train_pos_rate=round(tr.label.mean(), 4), test_pos_rate=round(te.label.mean(), 4))
    log(audit)
    for d in (tr, te):
        d["clean"] = d.text.map(clean_text)
        d["len"] = d.text.str.len()

    Xa, Xv, ya, yv = train_test_split(tr.clean, tr.label, test_size=0.2, stratify=tr.label, random_state=RS)
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=150000, sublinear_tf=True)
    A, V = vec.fit_transform(Xa), vec.transform(Xv)
    cands = {
        "Logistic Regression": LogisticRegression(C=5, max_iter=2000, class_weight="balanced"),
        "Naive Bayes": ComplementNB(alpha=0.3),
        "Linear SVM": CalibratedClassifierCV(LinearSVC(C=0.5, class_weight="balanced"), cv=3),
        "Decision Tree": DecisionTreeClassifier(max_depth=30, class_weight="balanced", random_state=RS),
    }
    comp = {}
    for n, m in cands.items():
        t = time.time(); m.fit(A, ya); comp[n] = metrics(m, V, yv.values); comp[n]["fit_s"] = round(time.time() - t, 1)
        log(n, comp[n]["f1"], comp[n]["roc_auc"])
    rv = TfidfVectorizer(ngram_range=(1, 1), min_df=3, max_features=20000, sublinear_tf=True)
    rf = RandomForestClassifier(150, n_jobs=-1, class_weight="balanced", random_state=RS, min_samples_leaf=2)
    t = time.time(); RA = rv.fit_transform(Xa); rf.fit(RA, ya)
    comp["Random Forest"] = metrics(rf, rv.transform(Xv), yv.values); comp["Random Forest"]["fit_s"] = round(time.time() - t, 1)
    log("Random Forest", comp["Random Forest"]["f1"])

    best = max(["Logistic Regression", "Naive Bayes", "Linear SVM", "Decision Tree"], key=lambda k: comp[k]["f1"])
    log("best (validation F1, linear-text candidates):", best)
    # refit best on full cleaned train and evaluate once on the held-out, leakage-free test set
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=150000, sublinear_tf=True)
    Xtr = vec.fit_transform(tr.clean)
    final = cands[best]; final.fit(Xtr, tr.label)
    Xte = vec.transform(te.clean)
    test = metrics(final, Xte, te.label.values)
    log("TEST", test)

    # honest diagnostics: by length bucket, and a length-only baseline
    buckets = {}
    s = final.predict_proba(Xte)[:, 1]
    for name, lo, hi in [("<200 chars (SMS-like)", 0, 200), ("200-1000", 200, 1000), (">1000 (long dialogue)", 1000, 10**9)]:
        mk = ((te.len >= lo) & (te.len < hi)).values
        if mk.sum():
            buckets[name] = {"n": int(mk.sum()), "scam_share": round(float(te.label.values[mk].mean()), 3),
                             "accuracy": round(float(((s[mk] >= .5) == te.label.values[mk]).mean()), 4)}
    len_model = LogisticRegression().fit(np.log1p(tr[["len"]]), tr.label)
    len_acc = float((len_model.predict(np.log1p(te[["len"]])) == te.label).mean())
    log("by length:", buckets, "| length-only baseline acc:", round(len_acc, 4))

    names = np.array(vec.get_feature_names_out())
    if hasattr(final, "coef_"):
        co = final.coef_[0]
    else:
        co = np.log(final.feature_count_[1] / final.class_count_[1] + 1e-9) - np.log(final.feature_count_[0] / final.class_count_[0] + 1e-9) if hasattr(final, "feature_count_") else np.zeros(len(names))
    top = {"scam": names[np.argsort(-co)[:25]].tolist(), "legit": names[np.argsort(co)[:25]].tolist()}
    joblib.dump({"vectorizer": vec, "model": final, "name": best}, MODEL_DIR / "scam_text.joblib", compress=3)
    RESULTS["scam_text"] = {"audit": audit, "validation_comparison": comp, "selected": best, "test": test,
                            "length_buckets": buckets, "length_only_baseline_accuracy": round(len_acc, 4),
                            "top_terms": top}


# ----------------------------------------------------------------------------------------
# MODULE 2 - URL (two lexical datasets, both computable from the URL string)
# ----------------------------------------------------------------------------------------
def fit_tabular(X, y, scale_models=("Logistic Regression",)):
    Xa, Xt, ya, yt = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RS)
    cands = {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")),
        "Decision Tree": DecisionTreeClassifier(max_depth=8, min_samples_leaf=5, class_weight="balanced", random_state=RS),
        "Random Forest": RandomForestClassifier(300, n_jobs=-1, class_weight="balanced", min_samples_leaf=2, random_state=RS),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RS),
    }
    skf = StratifiedKFold(5, shuffle=True, random_state=RS)
    comp = {}
    for n, m in cands.items():
        p = cross_val_predict(m, Xa, ya, cv=skf, method="predict_proba")[:, 1]
        from sklearn.metrics import f1_score, roc_auc_score
        comp[n] = {"cv_f1": round(f1_score(ya, p >= .5), 4), "cv_roc_auc": round(roc_auc_score(ya, p), 4)}
        log(n, comp[n])
    best = max(comp, key=lambda k: comp[k]["cv_f1"])
    m = cands[best].fit(Xa, ya)
    return best, comp, m, metrics(m, Xt, yt.values), Xa.columns.tolist()


def train_url():
    log("\n=== MODULE 2: URL ===")
    b = raw("url_basic")
    n0 = len(b)
    b = b.drop_duplicates()
    X, y = b[BASIC12], b["target"]
    best, comp, model, test, cols = fit_tabular(X, y)
    model.fit(X, y)
    joblib.dump({"model": model, "features": BASIC12, "name": best}, MODEL_DIR / "url_basic.joblib", compress=3)
    RESULTS["url_basic"] = {"rows_raw": n0, "rows_clean": len(b), "pos_rate": round(float(y.mean()), 4),
                            "comparison_cv": comp, "selected": best, "test": test}

    u = raw("url_uci")
    u.columns = [c.strip() for c in u.columns]
    n1 = len(u)
    u = u.drop_duplicates()
    X = u[UCI8]
    y = (u["Result"] == -1).astype(int)                          # 1 = phishing
    best, comp, model, test, cols = fit_tabular(X, y)
    model.fit(X, y)
    joblib.dump({"model": model, "features": UCI8, "name": best}, MODEL_DIR / "url_uci.joblib", compress=3)
    RESULTS["url_uci"] = {"rows_raw": n1, "rows_clean": len(u), "pos_rate": round(float(y.mean()), 4),
                          "comparison_cv": comp, "selected": best, "test": test,
                          "note": "Only the 8 features computable from a URL string are used (page-content features excluded)."}


# ----------------------------------------------------------------------------------------
# MODULE 3 - Email
# ----------------------------------------------------------------------------------------
def train_email_text():
    log("\n=== MODULE 3a: email text (Enron fraud-labelled) ===")
    rng = np.random.RandomState(RS)
    parts, total, pos = [], 0, 0
    for ch in pd.read_csv(RAW_DIR / RAW_FILES["enron"], usecols=["Subject", "Body", "Label"], chunksize=100000):
        total += len(ch); pos += int(ch.Label.sum())
        keep = (ch.Label == 1) | (rng.rand(len(ch)) < 0.02)
        parts.append(ch[keep])
    d = pd.concat(parts)
    d["text"] = (d.Subject.fillna("") + " . " + d.Body.fillna("")).str[:6000]
    d = d[d.text.str.len() > 20].drop_duplicates("text")
    d["clean"] = d.text.map(clean_text)
    log("enron total", total, "fraud", pos, "sampled", len(d), d.Label.value_counts().to_dict())
    Xa, Xt, ya, yt = train_test_split(d.clean, d.Label, test_size=0.2, stratify=d.Label, random_state=RS)
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=120000, sublinear_tf=True)
    A, T = vec.fit_transform(Xa), vec.transform(Xt)
    cands = {"Logistic Regression": LogisticRegression(C=5, max_iter=3000, class_weight="balanced"),
             "Naive Bayes": ComplementNB(alpha=0.3),
             "Linear SVM": CalibratedClassifierCV(LinearSVC(C=0.5, class_weight="balanced"), cv=3)}
    comp = {}
    for n, m in cands.items():
        m.fit(A, ya); comp[n] = metrics(m, T, yt.values); log(n, comp[n]["f1"], comp[n]["roc_auc"])
    best = max(comp, key=lambda k: comp[k]["f1"])
    # external check on an independent, modern spam-email file the model never saw
    ext = raw("email_spam")
    ext_text = ext.iloc[:, 0].astype(str)
    lab_col = [c for c in ext.columns if c.lower() in ("label", "type", "spam", "class")]
    ext_res = None
    try:
        txt_col = [c for c in ext.columns if c.lower() in ("text", "email", "message", "body")][0]
        ycol = ext[lab_col[0]]
        yext = (ycol.astype(str).str.lower().isin(["spam", "1", "fraud", "phishing"])).astype(int)
        ext_res = metrics(cands[best], vec.transform(ext[txt_col].map(clean_text)), yext.values)
        ext_res["columns_used"] = [txt_col, lab_col[0]]
        log("external (email_spam.csv):", ext_res)
    except Exception as e:  # keep training robust if the schema differs
        log("external check skipped:", e)
    vec2 = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=120000, sublinear_tf=True)
    X = vec2.fit_transform(d.clean)
    final = cands[best].fit(X, d.Label)
    joblib.dump({"vectorizer": vec2, "model": final, "name": best}, MODEL_DIR / "email_text.joblib", compress=3)
    RESULTS["email_text"] = {"enron_rows_total": total, "enron_fraud_total": pos, "sample_rows": len(d),
                             "comparison_holdout": comp, "selected": best, "external_email_spam": ext_res}


def train_email_structural():
    log("\n=== MODULE 3b: email structural features (524K set) ===")
    d = raw("email_features")
    n0 = len(d)
    d = d.drop_duplicates()
    X = d.drop(columns=["label", "num_spelling_errors"])        # spelling errors not reproducible offline
    y = d.label
    Xa, Xt, ya, yt = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RS)
    cands = {"Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")),
             "Random Forest": RandomForestClassifier(200, n_jobs=-1, class_weight="balanced", min_samples_leaf=5, random_state=RS)}
    comp = {}
    for n, m in cands.items():
        m.fit(Xa, ya); comp[n] = metrics(m, Xt, yt.values); log(n, comp[n]["pr_auc"], comp[n]["roc_auc"])
    RESULTS["email_structural"] = {"rows_raw": n0, "rows_clean": len(d), "duplicates_removed": n0 - len(d),
                                   "pos_rate": round(float(y.mean()), 4), "comparison_holdout": comp,
                                   "note": "Supplementary experiment; weak signal (see PR-AUC vs positive rate). Not used in the app."}


if __name__ == "__main__":
    train_text(); train_url(); train_email_text(); train_email_structural()
    (REPORT_DIR / "metrics.json").write_text(json.dumps(RESULTS, indent=2))
    log("\nSaved models to", MODEL_DIR, "and metrics to reports/metrics.json")
