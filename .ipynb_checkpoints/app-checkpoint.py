import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords')

# Load model and vectorizer
model = pickle.load(open('spam_model.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [ps.stem(w) for w in words if w not in stop_words]
    return " ".join(words)

def predict(text):
    cleaned = preprocess(text)
    vector = tfidf.transform([cleaned]).toarray()
    result = model.predict(vector)[0]
    return "SPAM 🚫" if result == 1 else "HAM ✅"

# UI
st.title("📩 Spam Message Classifier")

user_input = st.text_area("Enter your message:")

if st.button("Check"):
    if user_input.strip() != "":
        result = predict(user_input)
        st.subheader("Result:")
        st.write(result)
    else:
        st.warning("Please enter a message!")