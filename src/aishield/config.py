from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"          # put the downloaded Kaggle files here (not committed)
SAMPLE_DIR = DATA_DIR / "sample"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
FIG_DIR = REPORT_DIR / "figures"
RANDOM_STATE = 42

# Expected raw file names (extract the Kaggle zips into data/raw/)
RAW_FILES = {
    "composite_train": "composite_train.csv",
    "composite_test": "composite_test.csv",
    "gct": "GCT_phase1_100k.csv",
    "url_basic": "phishing_url_dataset.csv",
    "url_uci": "PhishingData.csv",
    "email_features": "email_phishing_data.csv",
    "email_spam": "email_spam.csv",
    "enron": "enron_data_fraud_labeled.csv",
    "cyber": "Global_Cybersecurity_Threats_2015-2024.csv",
    "crypto": "uris.csv",
}
