import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------

st.markdown("""
<style>

.main {
    background-color: #0E1117;
    color: white;
}

.stTextArea textarea {
    background-color: #1E1E1E;
    color: white;
    border-radius: 15px;
    border: 2px solid #4A90E2;
    padding: 15px;
    font-size: 16px;
}

.stButton button {
    width: 100%;
    background: linear-gradient(90deg, #4A90E2, #6A5ACD);
    color: white;
    border-radius: 12px;
    height: 3em;
    font-size: 18px;
    font-weight: bold;
    border: none;
}

.stButton button:hover {
    background: linear-gradient(90deg, #6A5ACD, #4A90E2);
    color: white;
}

.result-box {
    padding: 25px;
    border-radius: 20px;
    text-align: center;
    font-size: 28px;
    font-weight: bold;
    margin-top: 20px;
}

.real {
    background: rgba(0,255,0,0.15);
    border: 2px solid #00FF99;
}

.fake {
    background: rgba(255,0,0,0.15);
    border: 2px solid #FF4B4B;
}

.title {
    text-align: center;
    font-size: 60px;
    font-weight: bold;
    background: -webkit-linear-gradient(#4A90E2, #6A5ACD);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.subtitle {
    text-align: center;
    font-size: 20px;
    color: #AAAAAA;
    margin-bottom: 40px;
}

</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------

model_path = "fake_news_model"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# ---------------- SIDEBAR ----------------

with st.sidebar:
    st.title("ℹ️ About")
    
    st.write("""
    This AI system detects whether news is:
    
    - ✅ Real
    - ❌ Fake
    
    using a fine-tuned BERT transformer model.
    """)

    st.markdown("---")

    st.subheader("⚙️ Tech Stack")
    st.write("""
    - Streamlit
    - BERT
    - Transformers
    - PyTorch
    """)

    st.markdown("---")

    st.subheader("📊 Model")
    st.write("Accuracy: ~98%")

# ---------------- HERO SECTION ----------------

st.markdown('<p class="title">📰 Fake News Detector</p>', unsafe_allow_html=True)

st.markdown(
    '<p class="subtitle">AI-powered fake news detection using BERT transformers</p>',
    unsafe_allow_html=True
)

# ---------------- INPUT ----------------

news = st.text_area(
    "Enter News Article",
    height=250,
    placeholder="Paste news article here..."
)

# ---------------- PREDICTION ----------------

if st.button("🔍 Analyze News"):

    if news.strip() == "":
        st.warning("Please enter news text.")
    else:

        with st.spinner("Analyzing with BERT..."):

            inputs = tokenizer(
                news,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=256
            )

            outputs = model(**inputs)

            prediction = torch.argmax(outputs.logits).item()

            probs = torch.softmax(outputs.logits, dim=1)
            confidence = torch.max(probs).item()

            st.markdown("---")

            if prediction == 1:

                st.markdown(
                    f"""
                    <div class="result-box real">
                    ✅ REAL NEWS<br><br>
                    Confidence: {confidence*100:.2f}%
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.progress(float(confidence))

            else:

                st.markdown(
                    f"""
                    <div class="result-box fake">
                    ❌ FAKE NEWS<br><br>
                    Confidence: {confidence*100:.2f}%
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.progress(float(confidence))