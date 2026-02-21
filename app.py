import streamlit as st
import pickle 
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from streamlit_lottie import st_lottie
import requests

import nltk

def _ensure_nltk():
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords")

_ensure_nltk()

# --- Setup ---
st.set_page_config(page_title="Spam Classifier", page_icon="📩", layout="centered")

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))
    return " ".join(y)

# Load vectorizer and model
tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

# --- Load Lottie animations ---
def load_lottie_url(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

spam_animation = load_lottie_url("https://assets7.lottiefiles.com/packages/lf20_jbrw3hcz.json")
safe_animation = load_lottie_url("https://assets2.lottiefiles.com/packages/lf20_touohxv0.json")

# --- Custom CSS ---
st.markdown("""
    <style>
    body {
        background: linear-gradient(135deg, #0F2027, #203A43, #416788, #81D2C7);
    }
    .main {
        background-color: #E0E0E2;
        padding: 25px;
        border-radius: 18px;
        box-shadow: 0px 6px 20px rgba(0,0,0,0.2);
    }
    .stTextArea textarea {
        border: 2px solid #81D2C7;
        border-radius: 12px;
        font-size: 16px;
    }
    .stButton button {
        background: linear-gradient(90deg, #81D2C7, #416788);
        color: white;
        border-radius: 10px;
        font-size: 18px;
        font-weight: bold;
        padding: 0.6em 1.2em;
        transition: 0.3s;
        border: none;
    }
    /* Fix white line in progress bar */
    .stProgress > div > div > div > div {
        background-color: #416788; /* bar color */
        border-radius: 10px;
    }
    .stProgress > div > div {
        background-color: rgba(255,255,255,0.1); /* faint transparent background */
        border-radius: 10px;    
    .stButton button:hover {
        transform: scale(1.05);
        background: linear-gradient(90deg, #416788, #81D2C7);
    }
    .result-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 16px;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.15);
        text-align: center;
        margin-top: 20px;
    }
    .result-title {
        font-size: 26px;
        font-weight: bold;
    }
            
    .safe { color: #1ABC9C; }
    .spam { color: #E74C3C; }
    </style>
""", unsafe_allow_html=True)

# --- Title ---
st.markdown("<h1 style='text-align:center; color:white;'>📩 Email / SMS Spam Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; font-size:18px; color:#f0f0f0;'>Detect spam messages instantly using Machine Learning ⚡</p>", unsafe_allow_html=True)

# --- Sidebar ---
st.sidebar.header("⚙️ Settings")
max_len = st.sidebar.slider("Max characters allowed in input", 50, 500, 250)
st.sidebar.markdown("---")
st.sidebar.header("ℹ️ About")
st.sidebar.success("Built with Streamlit + MultinomialNB for NLP-based spam detection.")
st.sidebar.info("Paste suspicious SMS/Email content and see how the model reacts 🚀")

# --- Input ---
input_sms = st.text_area("✉️ Enter your message below:", height=150, max_chars=max_len)

# --- Prediction ---
if st.button('⚡ Analyze Message'):
    if input_sms.strip() == "":
        st.warning("⚠️ Please enter a message before predicting.")
    else:
        # Preprocess
        transform_sms = transform_text(input_sms)
        vector_input = tfidf.transform([transform_sms])
        
        # Prediction + probability
        result = model.predict(vector_input)[0]
        proba = model.predict_proba(vector_input)[0]

        # Confidence (pick the right side)
        if result == 1:
            confidence = proba[1] * 100
        else:
            confidence = proba[0] * 100

        # Word & sentence count
        words = len(input_sms.split())
        sentences = input_sms.count(".") + 1

        # Result Card
        with st.container():
            st.markdown("<div class='result-card'>", unsafe_allow_html=True)
            
            if result == 1:
                st.markdown("<div class='result-title spam'>🚨 SPAM MESSAGE</div>", unsafe_allow_html=True)
                st.write("This message appears to be **spam** and potentially unsafe.")
                st.progress(int(confidence))
                st.write(f"**Confidence (Spam):** {confidence:.1f}%")
                st_lottie(spam_animation, height=120, key="spam")
            else:
                st.markdown("<div class='result-title safe'>✅ LEGITIMATE MESSAGE</div>", unsafe_allow_html=True)
                st.write("This message appears to be **safe and legitimate**.")
                st.progress(int(confidence))
                st.write(f"**Confidence (Legitimate):** {confidence:.1f}%")
                st_lottie(safe_animation, height=120, key="safe")
            
            st.info(f"📊 Message contains **{words} words** and **{sentences} sentence(s)**.")
            st.markdown("</div>", unsafe_allow_html=True)

# --- Footer ---
st.markdown("""
    <div style='text-align:center; color:#E0E0E2; font-size:15px; margin-top:40px;'>
        🚀 Developed by <b>Apratim K Jha</b>
    </div>
""", unsafe_allow_html=True)

st.markdown("<p style='text-align:center; color:#E0E0E2; font-size:13px;'>Powered by MultinomialNB Machine Learning • Protecting your digital communications</p>", unsafe_allow_html=True)
