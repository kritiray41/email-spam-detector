import streamlit as st
import joblib

from src.preprocess import clean_text


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="AI Email Spam Detector",
    page_icon="📧",
    layout="centered"
)


# ==========================================
# LOAD MODEL AND TF-IDF
# ==========================================

@st.cache_resource
def load_artifacts():
    model = joblib.load("models/spam_model.pkl")
    tfidf = joblib.load("models/tfidf_vectorizer.pkl")

    return model, tfidf


try:
    model, tfidf = load_artifacts()
    artifacts_loaded = True

except Exception as e:
    artifacts_loaded = False
    error_message = str(e)


# ==========================================
# APPLICATION HEADER
# ==========================================

st.title("📧 Email Spam Detector ML Application")

st.markdown(
    "Enter any email text below to analyze whether it is "
    "**Legitimate (Ham)** or **Unsolicited (Spam)**."
)


# ==========================================
# CHECK MODEL
# ==========================================

if not artifacts_loaded:

    st.error(
        "Model files not found! "
        "Please run `python src/train.py` first."
    )

    st.stop()


# ==========================================
# EMAIL INPUT
# ==========================================

email_input = st.text_area(
    "Paste Email Content:",
    height=180,
    placeholder=(
        "e.g., Dear Customer, your account security "
        "needs urgent verification..."
    )
)


# ==========================================
# BUTTONS
# ==========================================

col1, col2 = st.columns([1, 4])

with col1:
    analyze_btn = st.button(
        "🔍 Analyze",
        type="primary"
    )

with col2:
    clear_btn = st.button("Reset Text")


if clear_btn:
    st.rerun()


# ==========================================
# ANALYZE EMAIL
# ==========================================

if analyze_btn:

    if not email_input.strip():

        st.warning(
            "Please enter email text before running analysis."
        )

    else:

        # ----------------------------------
        # PREPROCESS
        # ----------------------------------

        cleaned = clean_text(email_input)


        # ----------------------------------
        # TF-IDF VECTORIZE
        # ----------------------------------

        vectorized = tfidf.transform([cleaned])


        # ----------------------------------
        # PREDICTION
        # ----------------------------------

        prediction = model.predict(vectorized)[0]


        # ----------------------------------
        # PROBABILITY
        # ----------------------------------

        if hasattr(model, "predict_proba"):

            probs = model.predict_proba(vectorized)[0]

            ham_prob = probs[0] * 100
            spam_prob = probs[1] * 100

        else:

            spam_prob = 100.0 if prediction == 1 else 0.0
            ham_prob = 100.0 - spam_prob


        # ----------------------------------
        # DISPLAY RESULT
        # ----------------------------------

        st.divider()


        if prediction == 1:

            st.error(
                f"🚨 SPAM DETECTED "
                f"(Confidence: {spam_prob:.1f}%)"
            )

            st.write(
                "This email exhibits characteristics "
                "of unsolicited spam or phishing."
            )

        else:

            st.success(
                f"✅ CLEAN / HAM EMAIL "
                f"(Confidence: {ham_prob:.1f}%)"
            )

            st.write(
                "This email appears legitimate."
            )


        # ----------------------------------
        # TECHNICAL BREAKDOWN
        # ----------------------------------

        with st.expander(
            "Technical Preprocessing Breakdown"
        ):

            st.write(
                "**Cleaned & Stemmed Tokens:**",
                cleaned
            )

            st.write(
                "**Spam Probability Score:**",
                f"{spam_prob:.2f}%"
            )

            st.write(
                "**Ham Probability Score:**",
                f"{ham_prob:.2f}%"
            )