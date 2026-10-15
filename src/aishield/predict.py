"""Prediction + explainable-AI layer used by the Streamlit app and the CLI."""
import re
import warnings
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
try:
    from sklearn.exceptions import InconsistentVersionWarning
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
except ImportError:
    pass

from .config import MODEL_DIR
from .email_features import structural_features
from .text_clean import clean_text
from .url_features import BASIC12, UCI8, basic_features, red_flags, uci_features

# Indian-context fraud lexicon used for rule-based explanations (not for training)
LEXICON = {
    "Credential / OTP request": ["otp", "pin", "cvv", "password", "passcode", "verification code"],
    "KYC / account threat": ["kyc", "account blocked", "account suspended", "account will be", "blocked", "suspended", "deactivated", "expire", "pan card", "aadhaar", "aadhar"],
    "Urgency": ["urgent", "immediately", "within 24 hours", "last chance", "act now", "final notice", "hurry", "today only"],
    "Prize / reward bait": ["congratulations", "you have won", "you won", "lottery", "prize", "reward", "cashback", "lucky", "free gift", "claim"],
    "Payment pressure": ["upi", "paytm", "phonepe", "google pay", "gpay", "transfer", "wire", "send money", "processing fee", "refund", "bank"],
    "Call-to-action link": ["click here", "click the link", "click below", "tap the link", "login here", "verify now", "update now"],
    "Authority impersonation": ["rbi", "sbi", "hdfc", "icici", "income tax", "police", "customs", "cbi", "trai", "courier", "fedex"],
}
_URL = re.compile(r"https?://\S+|www\.\S+", re.I)


@lru_cache(maxsize=None)
def _load(name):
    return joblib.load(MODEL_DIR / f"{name}.joblib")


def _weights(model):
    if hasattr(model, "coef_"):
        return np.ravel(model.coef_)
    if hasattr(model, "calibrated_classifiers_"):
        try:
            return np.mean([np.ravel(c.estimator.coef_) for c in model.calibrated_classifiers_], axis=0)
        except Exception:
            return None
    return None


def keyword_flags(text: str):
    low = (text or "").lower()
    out = []
    for cat, words in LEXICON.items():
        hit = [w for w in words if w in low]
        if hit:
            out.append((cat, hit[:4]))
    if _URL.search(text or ""):
        out.append(("Contains a link", _URL.findall(text)[:2]))
    return out


def top_terms(bundle, text, k=6):
    vec, model = bundle["vectorizer"], bundle["model"]
    w = _weights(model)
    if w is None:
        return []
    x = vec.transform([clean_text(text)])
    contrib = x.multiply(w).tocoo()
    names = vec.get_feature_names_out()
    items = sorted(zip(contrib.col, contrib.data), key=lambda t: -t[1])
    return [names[i] for i, v in items if v > 0][:k]


def _verdict(p, hi=0.5):
    return "Fraudulent" if p >= hi else "Legitimate"


def predict_text(text: str, module="scam_text"):
    """module: 'scam_text' (messages / dialogues) or 'email_text'."""
    b = _load(module)
    p = float(b["model"].predict_proba(b["vectorizer"].transform([clean_text(text)]))[0, 1])
    flags = keyword_flags(text)
    n = len(text or "")
    notes = []
    if module == "scam_text" and n < 200:
        notes.append("Short message: the model was trained mostly on longer scam call/chat transcripts, "
                     "so the keyword indicators below matter more than the probability.")
    if module == "email_text":
        notes.append("Experimental: this email model scored F1 0.88 on held-out Enron data but did not "
                     "generalise to an independent email set (see reports). Treat as a hint only.")
    # transparent hybrid: strong rule evidence can raise a 'Legitimate' model verdict to 'Suspicious'
    verdict = _verdict(p)
    if verdict == "Legitimate" and len(flags) >= 3:
        verdict = "Suspicious"
    return {"verdict": verdict, "fraud_probability": round(p, 4), "confidence": round(max(p, 1 - p), 4),
            "model": b["name"], "model_terms": top_terms(b, text) if p >= 0.5 else [], "keyword_flags": flags, "notes": notes,
            "length": n}


def predict_url(url: str):
    ps = []
    for name, feats, fn in (("url_basic", BASIC12, basic_features), ("url_uci", UCI8, uci_features)):
        b = _load(name)
        row = pd.DataFrame([fn(url)])[b["features"]]
        ps.append(float(b["model"].predict_proba(row)[0, 1]))
    p = float(np.mean(ps))
    flags = red_flags(url)
    verdict = _verdict(p, 0.5)
    if verdict == "Legitimate" and len(flags) >= 4:
        verdict = "Suspicious"
    return {"verdict": verdict, "fraud_probability": round(p, 4), "confidence": round(max(p, 1 - p), 4),
            "model_scores": {"basic_lexical_model": round(ps[0], 4), "uci_lexical_model": round(ps[1], 4)},
            "red_flags": flags,
            "notes": ["URL models use only lexical features (no page content, domain age or reputation); "
                      "cross-validated ROC-AUC is about 0.81-0.83, so treat the result as a screening signal."]}


def predict_email(subject: str, body: str):
    text = f"{subject or ''} . {body or ''}"
    r = predict_text(text, "email_text")
    r["structure"] = structural_features(text)
    return r
