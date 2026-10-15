import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

_LINK = re.compile(r"https?://[^\s<>\"']+|www\.[^\s<>\"']+", re.I)
_MAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_WORD = re.compile(r"[A-Za-z']+")
URGENT = ["urgent", "immediately", "verify", "suspend", "act now", "expire", "final notice",
          "limited time", "confirm your", "password", "click here", "winner", "prize", "claim",
          "wire", "refund", "security alert"]


def structural_features(text: str) -> dict:
    """Features mirroring the 524K email-phishing feature dataset (spelling errors not computed)."""
    text = text or ""
    words = [w.lower() for w in _WORD.findall(text)]
    links = _LINK.findall(text)
    domains = {re.sub(r"^https?://(www\.)?", "", l, flags=re.I).split("/")[0].lower() for l in links}
    low = text.lower()
    return {
        "num_words": len(words),
        "num_unique_words": len(set(words)),
        "num_stopwords": sum(w in ENGLISH_STOP_WORDS for w in words),
        "num_links": len(links),
        "num_unique_domains": len(domains),
        "num_email_addresses": len(_MAIL.findall(text)),
        "num_urgent_keywords": sum(low.count(k) for k in URGENT),
    }
