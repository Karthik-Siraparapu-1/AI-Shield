"""Exploratory data analysis + result figures -> reports/figures/*.png and reports/dataset_audit.csv"""
import json
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from src.aishield.config import FIG_DIR, MODEL_DIR, RAW_DIR, RAW_FILES, REPORT_DIR

warnings.filterwarnings("ignore")
FIG_DIR.mkdir(parents=True, exist_ok=True)
BLUE, NAVY, RED, GREY, GREEN = "#0F62FE", "#161616", "#DA1E28", "#8D8D8D", "#24A148"
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "bold", "figure.dpi": 130, "axes.titlesize": 13})
M = json.load(open(REPORT_DIR / "metrics.json"))
rd = lambda k, **kw: pd.read_csv(RAW_DIR / RAW_FILES[k], **kw)


def save(name):
    plt.tight_layout(); plt.savefig(FIG_DIR / name, bbox_inches="tight"); plt.close()


# ---- dataset audit table ---------------------------------------------------------------
rows = []
def audit(name, df, target, task, note=""):
    rows.append({"dataset": name, "rows": len(df), "columns": df.shape[1], "missing_cells": int(df.isna().sum().sum()),
                 "duplicate_rows": int(df.duplicated().sum()), "target": target, "task": task, "finding": note})
tr, te = rd("composite_train"), rd("composite_test")
audit("Composite Scam (train)", tr, "label", "Scam text", "17% duplicate rows")
audit("Composite Scam (test)", te, "label", "Scam text", "2,900 texts leak from train")
gct = rd("gct"); audit("GCT 100K", gct, "human_verified_label", "Not used", "labels at chance level; 65% duplicate texts")
ub = rd("url_basic"); audit("Phishing URL features (2.5K)", ub, "target", "URL", "46% duplicates")
uu = rd("url_uci"); audit("UCI Phishing features (11K)", uu, "Result", "URL", "page-content features not usable")
ef = rd("email_features"); audit("Email features (525K)", ef, "label", "Email (suppl.)", "61% duplicate rows")
es = rd("email_spam"); audit("Email spam (84)", es, "type", "External email test", "tiny; used only for validation")
cy = rd("cyber"); audit("Global Cyber Threats", cy, "-", "Dashboard", "analytics only")
cr = rd("crypto"); audit("CryptoScamDB URIs", cr, "-", "Demo", "17 rows; no labels")
rows.append({"dataset": "Enron fraud-labelled emails", "rows": M["email_text"]["enron_rows_total"], "columns": 14, "missing_cells": None,
             "duplicate_rows": None, "target": "Label", "task": "Email text", "finding": "0.5% fraud; corporate-only negatives"})
A = pd.DataFrame(rows); A.to_csv(REPORT_DIR / "dataset_audit.csv", index=False)

# ---- 1. GCT label randomness -------------------------------------------------------------
u = gct.drop_duplicates("text_transcript")
Xa, Xt, ya, yt = train_test_split(u.text_transcript, u.human_verified_label, test_size=.2, random_state=0)
v = TfidfVectorizer(ngram_range=(1, 2), min_df=2); m = LogisticRegression(max_iter=500).fit(v.fit_transform(Xa), ya)
gct_acc = m.score(v.transform(Xt), yt)
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
g = gct.groupby("attack_type").human_verified_label.mean().sort_values()
ax[0].barh(g.index, g.values, color=BLUE); ax[0].axvline(.5, color=RED, ls="--"); ax[0].set_xlim(0, 1)
ax[0].set_title("GCT: share labelled 'fraud' per attack type"); ax[0].set_xlabel("share with label = 1")
ax[1].bar(["Text-only model\n(unique texts)", "Majority baseline"], [gct_acc, max(yt.mean(), 1 - yt.mean())], color=[RED, GREY])
ax[1].axhline(.5, color=NAVY, ls=":"); ax[1].set_ylim(0, 1); ax[1].set_title("GCT labels cannot be learned from text")
for i, val in enumerate([gct_acc, max(yt.mean(), 1 - yt.mean())]): ax[1].text(i, val + .02, f"{val:.1%}", ha="center")
save("01_gct_label_quality.png")
M["_gct_text_only_accuracy"] = round(float(gct_acc), 4)

# ---- 2. data quality: duplicates -----------------------------------------------------------
q = A.dropna(subset=["duplicate_rows"]).copy(); q["dup_pct"] = 100 * q.duplicate_rows / q.rows
q = q.sort_values("dup_pct")
plt.figure(figsize=(8, 4.5)); plt.barh(q.dataset, q.dup_pct, color=[RED if x > 30 else BLUE for x in q.dup_pct])
plt.xlabel("% duplicate rows"); plt.title("Data audit: duplicate rows per dataset"); save("02_duplicates.png")

# ---- 3. composite class + length -----------------------------------------------------------
tr = tr.drop_duplicates(); tr["len"] = tr.text.str.len()
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
c = tr.label.value_counts().sort_index(); ax[0].bar(["Legitimate", "Scam"], c.values, color=[BLUE, RED])
for i, val in enumerate(c.values): ax[0].text(i, val, f"{val:,}", ha="center", va="bottom")
ax[0].set_title("Scam text: class distribution (deduplicated train)")
for lab, col, nm in ((0, BLUE, "Legitimate"), (1, RED, "Scam")):
    ax[1].hist(np.log10(tr[tr.label == lab].len.clip(lower=1)), bins=40, alpha=.6, color=col, label=nm)
ax[1].set_xlabel("log10(message length in characters)"); ax[1].legend(); ax[1].set_title("Message length by class")
save("03_text_distribution.png")

# ---- 4. top terms -----------------------------------------------------------------------------
b = joblib.load(MODEL_DIR / "scam_text.joblib"); mdl = b["model"]
w = np.mean([np.ravel(cc.estimator.coef_) for cc in mdl.calibrated_classifiers_], axis=0) if hasattr(mdl, "calibrated_classifiers_") else np.ravel(mdl.coef_)
names = b["vectorizer"].get_feature_names_out(); o = np.argsort(w)
fig, ax = plt.subplots(1, 2, figsize=(11, 5))
ax[0].barh(names[o[::-1][:15]][::-1], w[o[::-1][:15]][::-1], color=RED); ax[0].set_title("Terms pushing toward SCAM")
ax[1].barh(names[o[:15]][::-1], -w[o[:15]][::-1], color=BLUE); ax[1].set_title("Terms pushing toward LEGITIMATE")
save("04_top_terms.png")

# ---- 5. URL distributions -----------------------------------------------------------------------
ub = ub.drop_duplicates()
fig, ax = plt.subplots(1, 3, figsize=(12, 3.8))
for a_, col in zip(ax, ["url_length", "nb_dots", "sensitive_words_count"]):
    for lab, c_, nm in ((0, BLUE, "Safe"), (1, RED, "Phishing")):
        a_.hist(ub[ub.target == lab][col].clip(upper=ub[col].quantile(.98)), bins=25, alpha=.6, color=c_, label=nm)
    a_.set_title(col)
ax[0].legend(); fig.suptitle("URL lexical features by class (deduplicated)", fontweight="bold"); save("05_url_features.png")

# ---- 6. email imbalance ----------------------------------------------------------------------------
e = M["email_structural"]; en = M["email_text"]
plt.figure(figsize=(6.5, 4)); vals = [100 * e["pos_rate"], 100 * en["enron_fraud_total"] / en["enron_rows_total"]]
plt.bar(["Email features\n(deduplicated)", "Enron corpus"], vals, color=RED)
for i, val in enumerate(vals): plt.text(i, val + .05, f"{val:.2f}%", ha="center")
plt.ylabel("% fraud"); plt.title("Email datasets are heavily imbalanced"); save("06_email_imbalance.png")

# ---- 7. model comparison -------------------------------------------------------------------------------
cmp_ = M["scam_text"]["validation_comparison"]
fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
ks = list(cmp_); ax[0].bar(ks, [cmp_[k]["f1"] for k in ks], color=BLUE); ax[0].set_ylim(.95, 1); ax[0].tick_params(axis="x", rotation=25)
ax[0].set_title("Scam text: validation F1"); [ax[0].text(i, cmp_[k]["f1"] + .0005, f"{cmp_[k]['f1']:.3f}", ha="center") for i, k in enumerate(ks)]
uc = M["url_uci"]["comparison_cv"]; ub_ = M["url_basic"]["comparison_cv"]; ks = list(uc); x = np.arange(len(ks))
ax[1].bar(x - .2, [ub_[k]["cv_roc_auc"] for k in ks], .4, color=BLUE, label="Basic URL set (1.35K)")
ax[1].bar(x + .2, [uc[k]["cv_roc_auc"] for k in ks], .4, color=NAVY, label="UCI lexical set (11K)")
ax[1].set_xticks(x); ax[1].set_xticklabels(ks, rotation=25); ax[1].set_ylim(.5, 1); ax[1].legend(); ax[1].set_title("URL models: 5-fold CV ROC-AUC")
save("07_model_comparison.png")

# ---- 8. confusion matrices ----------------------------------------------------------------------------------
fig, ax = plt.subplots(1, 3, figsize=(13, 4))
for a_, key, ttl in zip(ax, ["scam_text", "url_basic", "url_uci"], ["Scam text (test)", "URL basic (hold-out)", "URL UCI (hold-out)"]):
    cm = np.array(M[key]["test"]["confusion_matrix"]); a_.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2): a_.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=13)
    a_.set_xticks([0, 1]); a_.set_xticklabels(["Pred safe", "Pred fraud"]); a_.set_yticks([0, 1]); a_.set_yticklabels(["Safe", "Fraud"]); a_.set_title(ttl)
save("08_confusion_matrices.png")

# ---- 9. email external validation -------------------------------------------------------------------------------
ext = M["email_text"]["external_email_spam"]; hold = M["email_text"]["comparison_holdout"][M["email_text"]["selected"]]
plt.figure(figsize=(6.5, 4)); plt.bar(["Enron hold-out", "Independent emails\n(n=84)"], [hold["roc_auc"], ext["roc_auc"]], color=[BLUE, RED])
plt.axhline(.5, color=NAVY, ls=":"); plt.ylim(0, 1.05); plt.ylabel("ROC-AUC")
for i, val in enumerate([hold["roc_auc"], ext["roc_auc"]]): plt.text(i, val + .02, f"{val:.2f}", ha="center")
plt.title("Email text model: domain-shift test"); save("09_email_domain_shift.png")

# ---- 10. cyber dashboard ---------------------------------------------------------------------------------------------
fig, ax = plt.subplots(2, 2, figsize=(12, 8))
cy.groupby("Year").size().plot(ax=ax[0, 0], marker="o", color=BLUE); ax[0, 0].set_title("Incidents per year")
cy["Attack Type"].value_counts().sort_values().plot.barh(ax=ax[0, 1], color=BLUE); ax[0, 1].set_title("Incidents by attack type")
cy.groupby("Target Industry")["Financial Loss (in Million $)"].sum().sort_values().plot.barh(ax=ax[1, 0], color=RED); ax[1, 0].set_title("Financial loss by industry ($M)")
cy.groupby("Country")["Financial Loss (in Million $)"].sum().sort_values().tail(10).plot.barh(ax=ax[1, 1], color=NAVY); ax[1, 1].set_title("Top 10 countries by loss ($M)")
for a_ in ax.ravel(): a_.set_ylabel(""); a_.set_xlabel("")
save("10_cyber_dashboard.png")
json.dump(M, open(REPORT_DIR / "metrics.json", "w"), indent=2)
print("figures:", sorted(p.name for p in FIG_DIR.glob("*.png")))
print(A[["dataset", "rows", "duplicate_rows", "finding"]].to_string())
