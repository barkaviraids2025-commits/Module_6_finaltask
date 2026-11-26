"""
SMS Spam Detector - web app.   Run:  streamlit run app.py
"""

import joblib
import streamlit as st

st.set_page_config(page_title="SMS Spam Detector", page_icon="📩")


@st.cache_resource
def load_model():
    return joblib.load("model.joblib")


EXAMPLES = {
    "Prize scam": "Congratulations! You have won a FREE iPhone. Claim now by replying WIN to 87121. T&C apply",
    "Friend": "Hey, are we still meeting at 6? Let me know if you want me to bring anything",
    "Bank alert": "URGENT! Your account has been selected for a 500 cash reward. Call 09061701461 to claim",
    "Class update": "Sir, the lab record submission is moved to Monday. Please inform everyone in the group",
}

st.title("📩 SMS Spam Detector")
st.write("Paste a text message and the model will tell you if it looks like **spam** or a **normal (ham)** message.")

if "message" not in st.session_state:
    st.session_state["message"] = ""

st.caption("Try an example:")
cols = st.columns(len(EXAMPLES))
for col, (name, text) in zip(cols, EXAMPLES.items()):
    if col.button(name):
        st.session_state["message"] = text

message = st.text_area("Message", key="message", height=120, placeholder="Type or paste an SMS here...")

if st.button("Check message", type="primary"):
    if not message.strip():
        st.warning("Please enter a message first.")
    else:
        spam_prob = float(load_model().predict_proba([message])[0][1])
        if spam_prob >= 0.5:
            st.error(f"🚨 Looks like SPAM  ({spam_prob:.0%} spam probability)")
        else:
            st.success(f"✅ Looks like a normal message  ({1 - spam_prob:.0%} confidence)")
        st.progress(spam_prob, text=f"Spam probability: {spam_prob:.0%}")

with st.expander("About this model"):
    st.write("TF-IDF text features + Logistic Regression, trained on the SMS Spam Collection dataset "
             "(5,169 unique messages). It is a learning project and can make mistakes, "
             "so don't rely on it as your only protection.")
