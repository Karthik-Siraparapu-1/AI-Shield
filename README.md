# 🛡️ AI Shield — Unified Fraud Detection (Scam Text · Phishing URL · Email)

Final project for the **AICTE | IBM SkillsBuild Machine Learning & Applied AI Internship 2026 (BharatCares)**.
Supports **SDG 9** (Industry, Innovation & Infrastructure) and **SDG 16** (Peace, Justice & Strong Institutions).

**Live app:** _add your Streamlit Community Cloud URL here_ · **Author:** Karthik Siraparapu, Marwadi University

## What it does
Paste a suspicious SMS/WhatsApp message, URL or email. AI Shield returns **Legitimate / Suspicious / Fraudulent**, a fraud
probability and **the reasons** (fraud-lexicon hits such as OTP/KYC/urgency, the TF-IDF terms the model weighted most,
and URL red flags such as IP address, shortener, many subdomains). A dashboard shows global cyber-threat trends.

## Results (all numbers reproducible with `python train.py`)
| Module | Data | Model (selected) | Held-out result |
|---|---|---|---|
| Scam text | Composite Scam Transcripts, 30,540 train / 5,949 leakage-free test | TF-IDF (1-2 grams) + calibrated Linear SVM | Accuracy 99.7 %, F1 0.998, ROC-AUC 0.9997 |
| URL (basic set) | 1,354 unique URLs | Random Forest, 12 lexical features | ROC-AUC 0.81, F1 0.59 |
| URL (UCI set) | 11,055 URLs, 8 lexical features | Random Forest | ROC-AUC 0.83, F1 0.74 |
| Email text | Enron fraud-labelled sample (10,451) | TF-IDF + Linear SVM | Hold-out F1 0.88 — **but ROC-AUC 0.49 on an independent email set** |

### Honest findings from the data audit (see `reports/`)
* **GCT 100K labels are random** (text-only model: 49.9 % accuracy = chance, labels independent of attack type) → excluded from training.
* Composite Scam: 17 % duplicate rows and **2,900 of 9,188 test texts also appear in train** → deduplicated and leak-removed. A length-only baseline reaches 85 %, so results are also reported by length bucket (short, SMS-like messages: 98.8 %).
* URL datasets contain no raw URLs, only features; models use only features that can be recomputed from a URL string. Page-content features (domain age, traffic…) are impossible for a pasted URL.
* Email feature dataset: 61 % duplicates, 3 % positives; the structural model is weak (PR-AUC 0.23) and not used in the app.
* The email text model learns Enron-corporate style and fails on independent emails — the app labels the module *experimental*.
* Accuracy on templated scam dialogues is probably optimistic: near-duplicate scripts can remain after exact-duplicate removal.

## Project structure
```
app.py                  Streamlit app (Home, Scam, URL, Email, Dashboard, Model report)
train.py                Cleans data, compares models, saves models/*.joblib + reports/metrics.json
eda.py                  Audit table + 10 figures in reports/figures/
src/aishield/           text_clean, url_features, email_features, predict (explainable AI), evaluate
models/                 Trained models (~11 MB, committed so the app deploys as-is)
data/sample/            Small samples + cyber-threat CSV used by the dashboard
data/raw/               (not committed) download the Kaggle files here
tests/                  Unit tests for features and predictions
```

## Install & run
```bash
pip install -r requirements.txt
streamlit run app.py            # models are already included
```
To retrain: put the Kaggle files in `data/raw/` (names in `src/aishield/config.py`), then `python train.py && python eda.py`.

## Deploy (free) — Streamlit Community Cloud
1. Push this folder to a public GitHub repo. 2. Go to share.streamlit.io → *New app* → pick the repo, branch `main`, main file `app.py`. 3. Copy the app URL into this README and the submission form.

## Datasets (Kaggle)
`ibrahimbagwan12/composite-scam-transcript-dataset`, `sunnykusawa/phishing-urls`, `hasibur013/phishing-data`, `tapakah68/email-spam-classification`,
`advaithsrao/enron-fraud-email-dataset`, `atharvasoundankar/global-cybersecurity-threats-2015-2024`, `ethancratchley/email-phishing-dataset`,
`dspraneeth07/global-cognitive-threat-dataset-gct-100k` (audited, not used), `yaseenabdulghany/cryptoscamdb-scam-uris-and-details` (17 rows, demo only).

## Future work
Add a labelled Indian SMS-spam corpus (e.g. UCI SMS Spam Collection) for short messages; real phishing-email corpora; URL reputation/WHOIS features;
multilingual (Hindi/Gujarati) text; browser extension; QR-code scam detection.

## Disclaimer
Educational prototype; not a substitute for bank, police or CERT-In guidance. Report fraud at cybercrime.gov.in or call 1930 (India).
