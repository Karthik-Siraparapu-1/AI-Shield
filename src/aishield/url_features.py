"""Lexical URL features that can be computed from the URL string alone.

Two feature sets match the two training datasets:
  * basic12 -> phishing_url_dataset.csv (numeric lexical features; 'valid_url' dropped
               because it cannot be reproduced from a string)
  * uci8    -> 8 lexical features from the UCI phishing dataset (-1 / 0 / 1 coding,
               following Mohammad et al. 2012)
"""
import re
from urllib.parse import urlparse

SENSITIVE = ["login", "signin", "sign-in", "verify", "verification", "update", "secure", "account",
             "bank", "confirm", "password", "paypal", "kyc", "wallet", "bonus", "free", "gift",
             "upi", "otp", "support", "billing", "suspend", "unlock"]
SHORTENERS = ("bit.ly", "goo.gl", "tinyurl.com", "t.co", "ow.ly", "is.gd", "buff.ly", "adf.ly",
              "cutt.ly", "rb.gy", "shorturl.at", "tiny.cc", "bitly.com", "rebrand.ly")
SUSPICIOUS_TLDS = (".xyz", ".top", ".win", ".info", ".click", ".gq", ".tk", ".ml", ".cf", ".ga",
                   ".work", ".loan", ".zip", ".icu", ".buzz")
_IP = re.compile(r"^(\d{1,3}\.){3}\d{1,3}$")

BASIC12 = ["url_length", "at_symbol", "sensitive_words_count", "path_length", "isHttps",
           "nb_dots", "nb_hyphens", "nb_and", "nb_or", "nb_www", "nb_com", "nb_underscore"]
UCI8 = ["having_IPhaving_IP_Address", "URLURL_Length", "Shortining_Service", "having_At_Symbol",
        "double_slash_redirecting", "Prefix_Suffix", "having_Sub_Domain", "HTTPS_token"]


def normalise(url: str) -> str:
    url = (url or "").strip()
    if not re.match(r"^[a-z][a-z0-9+.-]*://", url, re.I):
        url = "http://" + url
    return url


def parse(url: str):
    u = normalise(url)
    p = urlparse(u)
    host = (p.hostname or "").lower()
    return u, p, host


def basic_features(url: str) -> dict:
    raw = (url or "").strip()
    u, p, host = parse(url)
    low = u.lower()
    return {
        "url_length": len(raw),
        "at_symbol": int("@" in raw),
        "sensitive_words_count": sum(low.count(w) for w in SENSITIVE[:6]),
        "path_length": len(p.path or ""),
        "isHttps": int(raw.lower().startswith("https")),
        "nb_dots": raw.count("."),
        "nb_hyphens": raw.count("-"),
        "nb_and": raw.count("&"),
        "nb_or": low.count("or"),
        "nb_www": low.count("www"),
        "nb_com": low.count("com"),
        "nb_underscore": raw.count("_"),
    }


def uci_features(url: str) -> dict:
    raw = (url or "").strip()
    u, p, host = parse(url)
    has_ip = bool(_IP.match(host))
    n = len(raw)
    host_nw = host[4:] if host.startswith("www.") else host
    dots = host_nw.count(".")
    return {
        "having_IPhaving_IP_Address": -1 if has_ip else 1,
        "URLURL_Length": 1 if n < 54 else (0 if n <= 75 else -1),
        "Shortining_Service": -1 if any(host.endswith(s) for s in SHORTENERS) else 1,
        "having_At_Symbol": -1 if "@" in raw else 1,
        "double_slash_redirecting": -1 if raw.rfind("//") > 7 else 1,
        "Prefix_Suffix": -1 if "-" in host else 1,
        "having_Sub_Domain": 1 if dots <= 1 else (0 if dots == 2 else -1),
        "HTTPS_token": -1 if "https" in host else 1,
    }


def red_flags(url: str):
    """Human-readable reasons a URL looks risky (used by the explainable-AI panel)."""
    raw = (url or "").strip()
    u, p, host = parse(url)
    flags = []
    if _IP.match(host):
        flags.append("Uses a raw IP address instead of a domain name")
    if any(host.endswith(s) for s in SHORTENERS):
        flags.append("Uses a URL-shortening service that hides the real destination")
    if "@" in raw:
        flags.append("Contains '@' (can disguise the real host)")
    if not raw.lower().startswith("https"):
        flags.append("Does not use HTTPS")
    if len(raw) > 75:
        flags.append(f"Very long URL ({len(raw)} characters)")
    host_nw = host[4:] if host.startswith("www.") else host
    if host_nw.count(".") >= 3:
        flags.append("Many subdomains (%d dots in host)" % host_nw.count("."))
    if "-" in host:
        flags.append("Hyphen in domain name (common in look-alike domains)")
    found = sorted({w for w in SENSITIVE if w in u.lower()})
    if found:
        flags.append("Sensitive keywords: " + ", ".join(found[:6]))
    if host.endswith(SUSPICIOUS_TLDS):
        flags.append("Domain ends in a TLD frequently abused for scams (." + host.rsplit('.', 1)[-1] + ")")
    if re.search(r"(amaz0n|paypa1|g00gle|micr0soft|faceb00k|sbi-|hdfc-|paytm-)", host):
        flags.append("Looks like an imitation of a well-known brand")
    if raw.rfind("//") > 7:
        flags.append("Contains a '//' redirect inside the path")
    return flags
