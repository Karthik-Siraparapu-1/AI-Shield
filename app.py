"""AI Shield - Streamlit app.  Run:  streamlit run app.py"""
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))
from src.aishield.config import FIG_DIR, RAW_DIR, RAW_FILES, REPORT_DIR, SAMPLE_DIR  # noqa: E402
from src.aishield.predict import predict_email, predict_text, predict_url  # noqa: E402

st.set_page_config(page_title="AI Shield", page_icon="🛡️", layout="wide")

EXAMPLES = {
    "Bank KYC scam": "Dear customer, your SBI account will be blocked today. Update KYC immediately: http://sbi-kyc-update.xyz/login and share the OTP to verify.",
    "Lottery scam": "Congratulations! You have won Rs 50,000 in the lucky draw. Click here to claim your prize now: bit.ly/3xYz",
    "Digital-arrest call": "This is the police. A parcel in your name contains illegal items. Transfer money to a safe account immediately or you will be arrested. Do not tell anyone.",
    "Normal message": "Hey, are we still meeting at 6 for dinner? I'll bring the notes.",
}


@st.cache_data
def load_metrics():
    p = REPORT_DIR / "metrics.json"
    return json.loads(p.read_text()) if p.exists() else {}


def show_result(r, kind):
    colour = {"Fraudulent": "red", "Suspicious": "orange", "Legitimate": "green"}[r["verdict"]]
    c1, c2, c3 = st.columns(3)
    c1.markdown(f"### :{colour}[{r['verdict']}]")
    c2.metric("Fraud probability", f"{r['fraud_probability']:.1%}")
    c3.metric("Model confidence", f"{r['confidence']:.1%}")
    st.progress(min(max(r["fraud_probability"], 0.0), 1.0))
    st.subheader("Why?")
    if kind == "url":
        if r["red_flags"]:
            for f in r["red_flags"]:
                st.write("⚠️ " + f)
        else:
            st.write("✅ No lexical red flags found.")
        st.caption(f"Model scores: {r['model_scores']}")
    else:
        if r["keyword_flags"]:
            for cat, words in r["keyword_flags"]:
                st.write(f"⚠️ **{cat}**: {', '.join(map(str, words))}")
        else:
            st.write("✅ No fraud-lexicon indicators found.")
        if r["model_terms"]:
            st.write("Terms the model weighted most toward fraud: " + ", ".join(f"`{t}`" for t in r["model_terms"]))
    for n in r["notes"]:
        st.info(n)


page = st.sidebar.radio("AI Shield", ["🏠 Home", "💬 Scam Message", "🔗 URL Check", "📧 Email Check",
                                      "📊 Threat Dashboard", "🧪 Model Report", "ℹ️ About"])
st.sidebar.caption("Educational prototype. Not a substitute for bank or police advice.")

if page.startswith("🏠"):
    st.title("🛡️ AI Shield")
    st.subheader("Unified fraud detection for scam messages, phishing URLs and emails")
    M = load_metrics()
    a, b, c = st.columns(3)
    if M:
        a.metric("Scam-text test accuracy", f"{M['scam_text']['test']['accuracy']:.1%}")
        b.metric("URL model ROC-AUC", f"{M['url_uci']['test']['roc_auc']:.2f}")
        c.metric("Email model (Enron hold-out F1)", f"{M['email_text']['comparison_holdout'][M['email_text']['selected']]['f1']:.2f}")
    st.write("Paste a suspicious SMS/WhatsApp message, URL or email and get a verdict, a probability and the reasons behind it. "
             "Supports **SDG 9** (secure digital infrastructure) and **SDG 16** (reducing cyber-enabled financial crime).")
    st.warning("Honest limits: URL models use lexical features only (ROC-AUC ≈ 0.81–0.83) and the email model does not generalise "
               "beyond its training corpus. See the **Model Report** page.")

elif page.startswith("💬"):
    st.header("Scam message analyzer")
    ex = st.selectbox("Try an example", ["(type your own)"] + list(EXAMPLES))
    text = st.text_area("Message", value="" if ex.startswith("(") else EXAMPLES[ex], height=160)
    if st.button("Analyze message", type="primary") and text.strip():
        show_result(predict_text(text), "text")

elif page.startswith("🔗"):
    st.header("Phishing URL detector")
    ex = st.selectbox("Try an example", ["(type your own)", "https://www.google.com", "http://sbi-kyc-update.xyz/login/verify?acc=1",
                                         "http://192.168.4.5/paytm/login.php", "http://secure-paypal.com.account-verify.info/update"])
    url = st.text_input("URL", value="" if ex.startswith("(") else ex)
    if st.button("Check URL", type="primary") and url.strip():
        show_result(predict_url(url), "url")

elif page.startswith("📧"):
    st.header("Email fraud checker (experimental)")
    subj = st.text_input("Subject")
    body = st.text_area("Body", height=200)
    if st.button("Check email", type="primary") and (subj.strip() or body.strip()):
        r = predict_email(subj, body)
        show_result(r, "text")
        st.write("Structural features:", r["structure"])

elif page.startswith("📊"):
    st.header("Cyber threat dashboard (2015-2024)")
    f = RAW_DIR / RAW_FILES["cyber"]
    f = f if f.exists() else SAMPLE_DIR / RAW_FILES["cyber"]
    if f.exists():
        d = pd.read_csv(f)
        yrs = st.slider("Years", int(d.Year.min()), int(d.Year.max()), (int(d.Year.min()), int(d.Year.max())))
        d = d[d.Year.between(*yrs)]
        c1, c2 = st.columns(2)
        c1.subheader("Incidents by attack type"); c1.bar_chart(d["Attack Type"].value_counts())
        c2.subheader("Financial loss by industry ($M)"); c2.bar_chart(d.groupby("Target Industry")["Financial Loss (in Million $)"].sum())
        c1.subheader("Incidents per year"); c1.line_chart(d.groupby("Year").size())
        c2.subheader("Top countries by loss ($M)"); c2.bar_chart(d.groupby("Country")["Financial Loss (in Million $)"].sum().nlargest(10))
    else:
        st.error("Dataset not found. Place the Global Cybersecurity Threats CSV in data/sample/.")

elif page.startswith("🧪"):
    st.header("Model report")
    M = load_metrics()
    for k, t in [("scam_text", "Scam text"), ("url_basic", "URL (basic lexical set)"), ("url_uci", "URL (UCI lexical set)")]:
        if k in M:
            st.subheader(t); st.json({kk: v for kk, v in M[k]["test"].items()})
    for img in sorted(FIG_DIR.glob("*.png")):
        st.image(str(img), caption=img.stem.replace("_", " "))

else:
    st.header("About")
    st.write("AI Shield was built for the AICTE | IBM SkillsBuild Machine Learning & Applied AI Internship 2026 (BharatCares). "
             "Models: TF-IDF + calibrated Linear SVM (scam text), Random Forest on URL lexical features, TF-IDF Linear SVM/LR (email). "
             "Every dataset was audited before use; unusable data (GCT labels) was excluded and leakage removed.")
