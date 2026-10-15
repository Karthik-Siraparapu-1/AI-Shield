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


def compute_text_confidence(p: float, flags: list, text_len: int, verdict: str) -> float:
    """Computes a multi-signal diagnostic certainty index distinct from raw fraud probability.

    - Fraud Probability: P(Fraud) measuring the threat likelihood (0% to 100%).
    - Confidence Score: Diagnostic certainty based on decision margin (|p - 0.5|),
      corroboration by fraud-lexicon indicators, and text length sufficiency.
    """
    margin = 2.0 * abs(p - 0.5)  # 0 at boundary (0.5), 1.0 at extremes (0.0 or 1.0)
    base = 0.52 + 0.40 * (margin ** 1.3)

    num_flag_cats = len(flags)
    if verdict == "Fraudulent":
        if num_flag_cats >= 3:
            corroboration = 0.05
        elif num_flag_cats >= 1:
            corroboration = 0.02
        else:
            corroboration = -0.10  # Model says fraud but no explicit threat keywords detected
    elif verdict == "Suspicious":
        corroboration = -0.06  # Conflicting signals by definition
    else:  # Legitimate
        if num_flag_cats == 0:
            corroboration = 0.04  # Clean text reinforces safe diagnosis
        else:
            corroboration = -0.12 * num_flag_cats  # Conflicting signals reduce certainty

    # Brevity / entropy penalty for very short texts (<30 chars)
    if text_len < 30:
        length_penalty = 0.82
    elif text_len < 60:
        length_penalty = 0.92
    else:
        length_penalty = 1.0

    score = (base + corroboration) * length_penalty
    return round(float(np.clip(score, 0.51, 0.992)), 4)


def compute_url_confidence(p: float, ps: list, flags: list, verdict: str) -> float:
    """Computes URL diagnostic certainty based on model consensus and lexical red-flags."""
    margin = 2.0 * abs(p - 0.5)
    base = 0.52 + 0.38 * (margin ** 1.2)

    # Ensemble consensus between basic12 and uci8 models (|p_basic - p_uci|)
    diff = abs(ps[0] - ps[1])
    consensus = 0.05 if diff < 0.12 else (-0.14 * diff)

    # Lexical red-flag alignment
    num_flags = len(flags)
    if verdict in ("Fraudulent", "Suspicious"):
        corroboration = 0.06 if num_flags >= 3 else (0.03 if num_flags >= 1 else -0.08)
    else:
        corroboration = 0.04 if num_flags == 0 else (-0.08 * num_flags)

    score = base + consensus + corroboration
    return round(float(np.clip(score, 0.51, 0.992)), 4)


def predict_text(text: str, module="scam_text"):
    """module: 'scam_text' (messages / dialogues) or 'email_text'."""
    b = _load(module)
    p = float(b["model"].predict_proba(b["vectorizer"].transform([clean_text(text)]))[0, 1])
    flags = keyword_flags(text)
    n = len(text or "")
    notes = []
    if module == "scam_text" and n < 200:
        notes.append("Short message: the model was trained mostly on longer scam call/chat transcripts, "
                     "so the keyword indicators below matter more than the raw probability.")
    if module == "email_text":
        notes.append("Experimental: this email model scored F1 0.88 on held-out Enron data but did not "
                     "generalise to an independent email set (see reports). Treat as a hint only.")

    # Transparent hybrid: strong rule evidence can raise a 'Legitimate' model verdict to 'Suspicious'
    verdict = _verdict(p)
    if verdict == "Legitimate" and len(flags) >= 3:
        verdict = "Suspicious"

    confidence = compute_text_confidence(p, flags, n, verdict)

    return {
        "verdict": verdict,
        "fraud_probability": round(p, 4),
        "confidence": confidence,
        "model": b["name"],
        "model_terms": top_terms(b, text) if (p >= 0.3 or verdict != "Legitimate") else [],
        "keyword_flags": flags,
        "notes": notes,
        "length": n,
    }


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

    confidence = compute_url_confidence(p, ps, flags, verdict)

    return {
        "verdict": verdict,
        "fraud_probability": round(p, 4),
        "confidence": confidence,
        "model": "Dual-Ensemble Random Forest (Basic12 + UCI8)",
        "model_scores": {"basic_lexical_model": round(ps[0], 4), "uci_lexical_model": round(ps[1], 4)},
        "red_flags": flags,
        "notes": [
            "URL models use only lexical features (no page content, domain age or reputation); "
            "cross-validated ROC-AUC is about 0.81-0.83, so treat the result as a screening signal."
        ],
    }


def predict_email(subject: str, body: str):
    text = f"{subject or ''} . {body or ''}"
    r = predict_text(text, "email_text")
    struct = structural_features(text)
    r["structure"] = struct
    if r["verdict"] == "Fraudulent" and struct.get("num_links", 0) > 0:
        r["confidence"] = round(min(0.992, r["confidence"] + 0.02), 4)
    return r
