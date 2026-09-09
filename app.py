import streamlit as st
import torch
import xgboost as xgb
from transformers import AutoTokenizer, AutoModel
from src.preprocess import clean_text

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="AI Email Spam Detector",
    page_icon="",
    layout="centered"
)

# ==========================================
# LOAD MODELS (Hugging Face + XGBoost)
# ==========================================

@st.cache_resource
def load_artifacts():
    MODEL_PATH = "models/"
    
    # Load frozen DistilBERT
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModel.from_pretrained(MODEL_PATH)
    
    # Load trained XGBoost classifier
    xgb_clf = xgb.XGBClassifier()
    xgb_clf.load_model(f"{MODEL_PATH}/xgboost_spam.json")

    return tokenizer, model, xgb_clf

try:
    tokenizer, model, xgb_clf = load_artifacts()
    artifacts_loaded = True
except Exception as e:
    artifacts_loaded = False
    error_message = str(e)

# ==========================================
# APPLICATION HEADER
# ==========================================

st.title("Email Spam Detector ML Application")

st.markdown(
    "Enter any email text below to analyze whether it is "
    "**Legitimate (Ham)** or **Unsolicited (Spam/Phishing)**."
)

# ==========================================
# CHECK MODEL
# ==========================================

if not artifacts_loaded:
    st.error(
        f"Model files not found! Please ensure 'models/offline_hybrid_model' exists. Error: {error_message}"
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
    analyze_btn = st.button(" Analyze", type="primary")

with col2:
    clear_btn = st.button("Reset Text")

if clear_btn:
    st.rerun()

# ==========================================
# ANALYZE EMAIL
# ==========================================

if analyze_btn:
    if not email_input.strip():
        st.warning("Please enter email text before running analysis.")
    else:
        # 1. Defang URLs
        cleaned = clean_text(email_input)

        # 2. Extract Semantic Embeddings via DistilBERT
        inputs = tokenizer(cleaned, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            embedding = model(**inputs).last_hidden_state[:, 0, :].numpy()

        # 3. XGBoost Prediction & Probability
        prediction = int(xgb_clf.predict(embedding)[0])
        probs = xgb_clf.predict_proba(embedding)[0]

        ham_prob = probs[0] * 100
        spam_prob = probs[1] * 100

        # ==========================================
        # DISPLAY RESULT
        # ==========================================
        st.divider()

        if prediction == 1:
            st.error(
                f" SPAM DETECTED "
                f"(Confidence: {spam_prob:.1f}%)"
            )
            st.write(
                "This email exhibits characteristics "
                "of unsolicited spam or phishing."
            )
        else:
            st.success(
                f" CLEAN / HAM EMAIL "
                f"(Confidence: {ham_prob:.1f}%)"
            )
            st.write("This email appears legitimate.")

        # ==========================================
        # TECHNICAL BREAKDOWN
        # ==========================================
        with st.expander("Technical Preprocessing Breakdown"):
            st.write("**Defanged Text (Seen by Model):**")
            st.text(cleaned)
            
            st.write(
                "**Spam Probability Score:**",
                f"{spam_prob:.2f}%"
            )
            st.write(
                "**Ham Probability Score:**",
                f"{ham_prob:.2f}%"
            )
