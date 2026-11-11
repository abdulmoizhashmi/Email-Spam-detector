from pathlib import Path
import joblib
import streamlit as st

MODEL_PATH = Path("models/spam_model.joblib")
st.set_page_config(page_title="Email Spam Detector", page_icon="✉️", layout="centered")
st.title("✉️ Email Spam Detector")
st.caption("Machine Learning • TF-IDF + Logistic Regression")
if not MODEL_PATH.exists():
    st.warning("The model has not been trained yet. In the project terminal, run: `python train.py`")
    st.stop()

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()
subject = st.text_input("Email subject", placeholder="e.g. Congratulations! You won a prize")
body = st.text_area("Email body", height=220, placeholder="Paste the email content here...")
if st.button("Analyze email", type="primary", use_container_width=True):
    content = (subject + "\n" + body).strip()
    if len(content) < 5:
        st.error("Please enter an email subject or body.")
    else:
        prediction = model.predict([content])[0]
        probabilities = model.predict_proba([content])[0]
        classes = list(model.classes_)
        spam_probability = float(probabilities[classes.index("spam")])
        if prediction == "spam":
            st.error("Likely SPAM")
        else:
            st.success("Likely NOT SPAM (HAM)")
        st.metric("Estimated spam probability", f"{spam_probability:.1%}")
        st.progress(spam_probability)
        st.caption("Model estimate, not a guarantee. Review suspicious links and attachments carefully.")
st.divider()
st.caption("Model: TF-IDF word + character features and Logistic Regression.")
