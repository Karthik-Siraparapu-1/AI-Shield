import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.aishield.email_features import structural_features
from src.aishield.predict import predict_email, predict_text, predict_url
from src.aishield.text_clean import clean_text
from src.aishield.url_features import basic_features, red_flags, uci_features


def test_clean_text_masks_urls_and_html():
    assert clean_text("<b>Click</b> http://x.com/a NOW") == "click urltoken now"
    assert clean_text(None) == ""


def test_url_features():
    f = uci_features("http://192.168.0.1/login")
    assert f["having_IPhaving_IP_Address"] == -1
    assert basic_features("https://a-b.com/x_y")["nb_hyphens"] == 1
    assert any("IP address" in r for r in red_flags("http://1.2.3.4/x"))


def test_email_structure():
    f = structural_features("Verify your password at http://a.com and http://b.com now")
    assert f["num_links"] == 2 and f["num_urgent_keywords"] >= 2


def test_scam_and_safe_text():
    assert predict_text("Your SBI account is blocked. Share OTP and update KYC at http://sbi-kyc.xyz")["verdict"] == "Fraudulent"
    assert predict_text("Hey, are we still meeting at 6 for dinner?")["verdict"] == "Legitimate"


def test_url_verdicts():
    assert predict_url("https://www.google.com")["verdict"] == "Legitimate"
    assert predict_url("http://secure-paypal.com.account-verify.info/update")["verdict"] in ("Fraudulent", "Suspicious")


def test_email_returns_structure():
    r = predict_email("Hello", "see you tomorrow")
    assert "structure" in r and 0 <= r["fraud_probability"] <= 1
