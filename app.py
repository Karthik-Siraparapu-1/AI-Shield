"""AI Shield - Unified Fraud Detection Platform
Built for AICTE | IBM SkillsBuild Machine Learning & Applied AI Internship 2026.
Author: Karthik Siraparapu (Marwadi University)
"""
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))
from src.aishield.config import FIG_DIR, RAW_DIR, RAW_FILES, REPORT_DIR, SAMPLE_DIR  # noqa: E402
from src.aishield.predict import predict_email, predict_text, predict_url  # noqa: E402

st.set_page_config(
    page_title="AI Shield — Unified Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
}
code, pre { font-family: 'JetBrains Mono', monospace !important; }

.hero-card {
    background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0F62FE 100%);
    border-radius: 16px; padding: 2.2rem 2.4rem; color: #fff;
    box-shadow: 0 10px 30px rgba(15,98,254,0.15);
    margin-bottom: 2rem; border: 1px solid rgba(255,255,255,0.1);
}
.hero-badge {
    display: inline-block; padding: 0.35rem 0.85rem;
    background: rgba(255,255,255,0.12); backdrop-filter: blur(8px);
    border: 1px solid rgba(255,255,255,0.2); border-radius: 9999px;
    font-size: 0.8rem; font-weight: 600; letter-spacing: 0.04em;
    text-transform: uppercase; color: #93c5fd; margin-bottom: 0.8rem;
}
.hero-title {
    font-size: 2.2rem; font-weight: 800; line-height: 1.2;
    margin: 0.2rem 0 0.8rem 0; color: #ffffff; letter-spacing: -0.02em;
}
.hero-subtitle { font-size: 1.05rem; color: #cbd5e1; line-height: 1.5; }

.metric-box {
    background: #ffffff; border: 1px solid #e2e8f0; border-radius: 14px;
    padding: 1.2rem 1.4rem; box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.metric-box:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(15,98,254,0.08); border-color: #bfdbfe; }
.metric-label { font-size: 0.82rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; color: #64748b; margin-bottom: 0.3rem; }
.metric-val { font-size: 1.75rem; font-weight: 800; color: #0f172a; letter-spacing: -0.02em; }
.metric-sub { font-size: 0.8rem; color: #10b981; font-weight: 600; }

.verdict-banner-Fraudulent { background: linear-gradient(135deg,#fef2f2,#fee2e2); border: 1.5px solid #ef4444; border-radius: 14px; padding: 1.5rem; margin: 1rem 0; }
.verdict-banner-Suspicious  { background: linear-gradient(135deg,#fffbeb,#fef3c7); border: 1.5px solid #f59e0b; border-radius: 14px; padding: 1.5rem; margin: 1rem 0; }
.verdict-banner-Legitimate  { background: linear-gradient(135deg,#f0fdf4,#dcfce7); border: 1.5px solid #10b981; border-radius: 14px; padding: 1.5rem; margin: 1rem 0; }

.verdict-tag { display: inline-block; padding: 0.4rem 1rem; border-radius: 9999px; font-weight: 800; font-size: 1.1rem; }
.tag-Fraudulent { background: #ef4444; color: #fff; }
.tag-Suspicious { background: #f59e0b; color: #fff; }
.tag-Legitimate { background: #10b981; color: #fff; }

.flag-chip { display: inline-flex; align-items: center; background: #f1f5f9; border: 1px solid #cbd5e1; border-radius: 8px; padding: 0.35rem 0.65rem; font-size: 0.82rem; font-weight: 600; color: #1e293b; margin: 0.2rem 0.3rem 0.2rem 0; }
.term-chip { display: inline-block; background: #eff6ff; color: #1d4ed8; border: 1px solid #bfdbfe; border-radius: 6px; padding: 0.2rem 0.5rem; font-size: 0.8rem; font-weight: 600; margin: 0.2rem; font-family: 'JetBrains Mono', monospace; }

.sidebar-profile { background: linear-gradient(180deg,#f8fafc,#f1f5f9); border: 1px solid #e2e8f0; border-radius: 12px; padding: 1rem; margin-top: 1.5rem; }
.profile-title { font-weight: 700; font-size: 0.92rem; color: #0f172a; }
.profile-subtitle { font-size: 0.78rem; color: #64748b; margin-bottom: 0.5rem; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

EXAMPLES = {
    "Bank KYC Threat (SMS)": "Dear customer, your SBI account will be blocked today. Update KYC immediately: http://sbi-kyc-update.xyz/login and share the OTP to verify.",
    "Prize / Lottery Bait": "Congratulations! You have won Rs 50,000 in the lucky draw. Click here to claim your prize now: bit.ly/3xYz",
    "Digital Arrest / Extortion": "This is the police cyber cell. A parcel in your name contains narcotics. Transfer money to the safe account immediately or you will be arrested. Do not disconnect.",
    "Normal Conversation": "Hey, are we still meeting at 6 for dinner? I'll bring the project notes.",
}

URL_EXAMPLES = [
    "https://www.google.com",
    "http://sbi-kyc-update.xyz/login/verify?acc=1",
    "http://192.168.4.5/paytm/login.php",
    "http://secure-paypal.com.account-verify.info/update",
    "https://github.com/Karthik-Siraparapu-1/AI-Shield",
]


@st.cache_data
def load_metrics():
    p = REPORT_DIR / "metrics.json"
    return json.loads(p.read_text()) if p.exists() else {}


def render_verdict_card(r, kind):
    verdict = r["verdict"]
    prob = r["fraud_probability"]
    conf = r["confidence"]
    st.markdown(
        f"""
        <div class="verdict-banner-{verdict}">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.8rem;">
                <span class="verdict-tag tag-{verdict}">{verdict.upper()}</span>
                <span style="font-size:0.9rem;font-weight:600;color:#475569;">
                    Model: <strong>{r.get('model','Ensemble RF')}</strong>
                </span>
            </div>
            <div style="display:grid;grid-template-columns:repeat(2,1fr);gap:1rem;margin-top:0.8rem;">
                <div>
                    <div style="font-size:0.8rem;font-weight:600;color:#64748b;text-transform:uppercase;">Fraud Probability</div>
                    <div style="font-size:1.8rem;font-weight:800;color:#0f172a;">{prob:.1%}</div>
                    <div style="font-size:0.75rem;color:#64748b;font-weight:500;">Malicious Threat Risk</div>
                </div>
                <div>
                    <div style="font-size:0.8rem;font-weight:600;color:#64748b;text-transform:uppercase;">Confidence Score</div>
                    <div style="font-size:1.8rem;font-weight:800;color:#0f172a;">{conf:.1%}</div>
                    <div style="font-size:0.75rem;color:#64748b;font-weight:500;">Multi-Signal Diagnostic Certainty</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.progress(min(max(prob, 0.0), 1.0))

    with st.expander("ℹ️ Understanding Fraud Probability vs. Confidence Score", expanded=False):
        st.markdown(
            """
            - **Fraud Probability ($0\\% - 100\\%$)**: Quantifies **threat severity** — the model's calibrated probability that this communication or URL is an active attack vector.
            - **Confidence Score ($50\\% - 99\\%+$)**: Quantifies **diagnostic certainty** — measuring how decisive the evidence is by evaluating model decision margins ($|p - 0.5|$), agreement between sub-models, and corroborating domain red flags (such as OTP requests, KYC threats, or suspicious URL structures).
            """
        )

    st.markdown("### 🔍 Explainable Insights & Red Flags")
    if kind == "url":
        c1, c2 = st.columns([1.2, 1])
        with c1:
            st.markdown("##### Lexical Heuristics")
            if r["red_flags"]:
                for f in r["red_flags"]:
                    st.markdown(f"🚨 <span class='flag-chip'>{f}</span>", unsafe_allow_html=True)
            else:
                st.success("✅ No deceptive lexical anomalies detected.")
        with c2:
            st.markdown("##### Sub-Model Scores")
            for m_name, score in r["model_scores"].items():
                st.write(f"• **{m_name}**: `{score:.1%}` risk")
    else:
        c1, c2 = st.columns([1.2, 1])
        with c1:
            st.markdown("##### Fraud-Lexicon Indicators")
            if r["keyword_flags"]:
                for cat, words in r["keyword_flags"]:
                    chips = "".join(f"<span class='flag-chip'>{w}</span>" for w in words)
                    st.markdown(f"**{cat}**: {chips}", unsafe_allow_html=True)
            else:
                st.success("✅ No high-risk keywords detected.")
        with c2:
            st.markdown("##### TF-IDF Feature Weights")
            if r["model_terms"]:
                chips = "".join(f"<span class='term-chip'>{t}</span>" for t in r["model_terms"])
                st.markdown(f"Strongest signals: {chips}", unsafe_allow_html=True)
            else:
                st.info("Neutral feature activation.")

    for n in r.get("notes", []):
        st.info(f"ℹ️ {n}")


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🛡️ **AI Shield**")
    st.caption("Explainable Multi-Vector Cyber Defense")
    page = st.radio(
        "Navigation",
        [
            "🏠 Overview",
            "💬 Scam Message Analyzer",
            "🔗 Phishing URL Detector",
            "📧 Email Fraud Checker",
            "📊 Threat Intelligence Dashboard",
            "🧪 Model Performance & Audit",
            "ℹ️ Project Information",
        ],
    )
    st.markdown("---")
    st.markdown(
        """
        <div class="sidebar-profile">
            <div class="profile-title">Karthik Siraparapu</div>
            <div class="profile-subtitle">Marwadi University</div>
            <div style="font-size:0.76rem;color:#475569;line-height:1.4;">
                <strong>IBM SkillsBuild | AICTE</strong><br>
                Internship 2026 (BharatCares)<br>
                ID: <code>IBMUEML004123</code>
            </div>
            <div style="margin-top:0.6rem;">
                <a href="https://github.com/Karthik-Siraparapu-1/AI-Shield" target="_blank"
                   style="text-decoration:none;font-size:0.78rem;font-weight:600;color:#0F62FE;">
                    📂 GitHub Repository ↗
                </a>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("Educational prototype. For real fraud, dial **1930** or visit cybercrime.gov.in.")


# ── PAGE 1: OVERVIEW ─────────────────────────────────────────────────────────
if page.startswith("🏠"):
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-badge">AICTE · IBM SkillsBuild 2026 Final Capstone</div>
            <h1 class="hero-title">AI Shield — Unified Cyber Fraud Defense</h1>
            <p class="hero-subtitle">
                An explainable, multi-vector threat intelligence platform to protect citizens from
                deceptive social-engineering scripts, malicious URLs, and fraudulent communications.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    M = load_metrics()
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        acc = f"{M['scam_text']['test']['accuracy']:.1%}" if M else "99.7%"
        st.markdown(f'<div class="metric-box"><div class="metric-label">Scam Text Accuracy</div><div class="metric-val">{acc}</div><div class="metric-sub">Linear SVM + Sigmoid Cal.</div></div>', unsafe_allow_html=True)
    with c2:
        auc = f"{M['url_uci']['test']['roc_auc']:.2f}" if M else "0.83"
        st.markdown(f'<div class="metric-box"><div class="metric-label">URL ROC-AUC</div><div class="metric-val">{auc}</div><div class="metric-sub">11k UCI Random Forest</div></div>', unsafe_allow_html=True)
    with c3:
        sel = M["email_text"]["selected"] if M else None
        f1 = f"{M['email_text']['comparison_holdout'][sel]['f1']:.2f}" if M and sel else "0.88"
        st.markdown(f'<div class="metric-box"><div class="metric-label">Email Holdout F1</div><div class="metric-val">{f1}</div><div class="metric-sub">Enron Baseline SVM</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown('<div class="metric-box"><div class="metric-label">Aligned SDG Goals</div><div class="metric-val">SDG 9 & 16</div><div class="metric-sub">Safe Digital Infrastructure</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### ⚡ Universal Quick Scanner")
    st.write("Paste any suspicious input — SMS, URL, or email excerpt — and AI Shield automatically routes it to the right model.")
    user_input = st.text_area(
        "Input text or URL to evaluate:",
        placeholder="e.g. 'Your SBI account is blocked. Update KYC at http://sbi-verify.xyz/login immediately'",
        height=120,
    )
    if st.button("🛡️ Run Security Analysis", type="primary") and user_input.strip():
        txt = user_input.strip()
        if txt.startswith("http://") or txt.startswith("https://") or (len(txt.split()) == 1 and "." in txt):
            st.info("🌐 URL detected — routed to **Lexical Ensemble URL Pipeline**.")
            render_verdict_card(predict_url(txt), "url")
        else:
            st.info("💬 Message detected — routed to **Calibrated NLP Scam Classifier**.")
            render_verdict_card(predict_text(txt), "text")


# ── PAGE 2: SCAM MESSAGE ANALYZER ────────────────────────────────────────────
elif page.startswith("💬"):
    st.title("💬 Scam Message Analyzer")
    st.write("Detect social-engineering, OTP demands, lottery baits, and digital arrest threats using a calibrated NLP pipeline.")
    col1, _ = st.columns([1, 1])
    with col1:
        sample_choice = st.selectbox("Load Test Vector", ["(Type or paste custom message)"] + list(EXAMPLES.keys()))
    preset_val = "" if sample_choice.startswith("(") else EXAMPLES[sample_choice]
    msg = st.text_area("Message Content", value=preset_val, height=170)
    if st.button("Analyze Scam Risk", type="primary") and msg.strip():
        render_verdict_card(predict_text(msg), "text")


# ── PAGE 3: PHISHING URL DETECTOR ────────────────────────────────────────────
elif page.startswith("🔗"):
    st.title("🔗 Phishing URL Detector")
    st.write("Inspect URLs for obfuscation, IP-based hosting, excessive subdomains, and credential harvesting — without making external network requests.")
    url_choice = st.selectbox("Benchmark URL", ["(Enter custom URL)"] + URL_EXAMPLES)
    target_url = st.text_input("Target URL", value="" if url_choice.startswith("(") else url_choice, placeholder="http://suspicious-domain.com/login")
    if st.button("Analyze URL", type="primary") and target_url.strip():
        render_verdict_card(predict_url(target_url), "url")


# ── PAGE 4: EMAIL FRAUD CHECKER ──────────────────────────────────────────────
elif page.startswith("📧"):
    st.title("📧 Email Fraud Checker")
    st.caption("⚠️ Experimental — see Model Report for generalization limits.")
    st.write("Extracts structural email signals (external links, urgency keywords) alongside TF-IDF content classification.")
    subj = st.text_input("Subject Line", placeholder="e.g. Critical Action Required: Update your credentials")
    body = st.text_area("Email Body", height=200, placeholder="Paste email content here...")
    if st.button("Analyze Email", type="primary") and (subj.strip() or body.strip()):
        r = predict_email(subj, body)
        render_verdict_card(r, "text")
        st.markdown("#### Structural Feature Extraction")
        st.json(r["structure"])


# ── PAGE 5: THREAT DASHBOARD ─────────────────────────────────────────────────
elif page.startswith("📊"):
    st.title("📊 Global Threat Intelligence Dashboard")
    st.write("Exploratory analysis of 3,000 cyber incidents across global critical infrastructure, finance, and tech (2015–2024).")

    f = RAW_DIR / RAW_FILES["cyber"]
    f = f if f.exists() else SAMPLE_DIR / RAW_FILES["cyber"]
    if f.exists():
        df = pd.read_csv(f)
        yr_range = st.slider("Filter Timeframe", int(df.Year.min()), int(df.Year.max()), (int(df.Year.min()), int(df.Year.max())))
        filtered = df[df.Year.between(*yr_range)]

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Filtered Incidents", f"{len(filtered):,}")
        k2.metric("Total Financial Loss", f"${filtered['Financial Loss (in Million $)'].sum():,.1f}M")
        k3.metric("Avg Loss / Incident", f"${filtered['Financial Loss (in Million $)'].mean():.2f}M")
        k4.metric("Leading Vulnerability", filtered["Security Vulnerability Type"].mode()[0] if not filtered.empty else "N/A")

        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Attacks by Vector")
            st.bar_chart(filtered["Attack Type"].value_counts(), color="#0F62FE")
        with c2:
            st.subheader("Financial Impact by Industry ($M)")
            st.bar_chart(filtered.groupby("Target Industry")["Financial Loss (in Million $)"].sum().sort_values(ascending=False), color="#fa4d56")

        c3, c4 = st.columns(2)
        with c3:
            st.subheader("Annual Incident Trajectory")
            st.line_chart(filtered.groupby("Year").size(), color="#10b981")
        with c4:
            st.subheader("Top 10 Countries by Loss ($M)")
            st.bar_chart(filtered.groupby("Country")["Financial Loss (in Million $)"].sum().nlargest(10), color="#8a3ffc")
    else:
        st.error("Dataset not found in data/sample/.")


# ── PAGE 6: MODEL AUDIT ───────────────────────────────────────────────────────
elif page.startswith("🧪"):
    st.title("🧪 Model Audit & Evaluation Benchmarks")
    st.write("Transparent reporting on model selection, data hygiene, and generalization testing.")
    M = load_metrics()
    t1, t2, t3 = st.tabs(["📊 Metrics", "🔬 Leakage Audit", "📈 Charts"])

    with t1:
        if M:
            for k, label in [("scam_text", "1. Scam Text — Linear SVM"), ("url_uci", "2. URL — RFC (UCI 11k)"), ("url_basic", "3. URL — RFC (Basic 1.3k)")]:
                if k in M:
                    with st.expander(label, expanded=True):
                        st.json(M[k]["test"])

    with t2:
        st.markdown("""
### Honest Data Science Audit
- **Composite Scam Leakage Purge:** Found 2,900 / 9,188 test texts duplicated in training. All purged.
- **GCT 100K Exclusion:** Label quality audit showed 49.9% accuracy (coin flip). Dataset excluded.
- **URL Features:** Only lexical features computable purely from the URL string — no external lookups.
        """)

    with t3:
        figures = sorted(FIG_DIR.glob("*.png"))
        if figures:
            for img in figures:
                st.image(str(img), caption=img.stem.replace("_", " ").title())
        else:
            st.info("Run `python eda.py` to generate figures in reports/figures/.")


# ── PAGE 7: ABOUT ─────────────────────────────────────────────────────────────
else:
    st.title("ℹ️ Project Architecture & Overview")
    st.markdown("""
### AI Shield
**A Unified, Explainable Fraud Detection Platform for Scam Messages, Phishing URLs, and Emails**

Developed as the final capstone for the **AICTE | IBM SkillsBuild Machine Learning & Applied AI Internship 2026** with **BharatCares (CSRBOX)**.

---

#### Key Highlights
- **Multi-Vector Defense:** Handles SMS fraud, malicious URLs, and deceptive emails in one console.
- **Explainable AI (XAI):** Surfaces the exact keywords, TF-IDF weights, and URL anomalies behind every decision.
- **Data Integrity First:** Rigorous audit, leak-free train/test splits, and honest metric reporting.
- **SDG 9 & SDG 16 Aligned:** Strengthening secure digital infrastructure and combating cyber-enabled fraud.

---

#### Author
- **Student:** Karthik Siraparapu | **Institution:** Marwadi University
- **Internship ID:** `IBMUEML004123` | **Mentor Org:** BharatCares & IBM SkillsBuild
    """)
