import streamlit as st
import joblib
import re
import nltk
import numpy as np
from nltk.corpus import stopwords

nltk.download('stopwords')

model = joblib.load("best_sentiment_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.split()
    text = [w for w in text if w not in stop_words]
    return ' '.join(text)

st.title("💬 Sentiment Analysis Web App")

user_input = st.text_area("Enter text")

if st.button("Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter some text")
    else:
        cleaned = clean_text(user_input)
        vector = vectorizer.transform([cleaned])

        prediction = model.predict(vector)[0]
        decision_scores = model.decision_function(vector)

        confidence = np.max(decision_scores)
        confidence_pct = round(
            (confidence / np.sum(np.abs(decision_scores))) * 100, 2
        )

        if prediction == 0:
            st.error("😠 Negative")
        elif prediction == 1:
            st.info("😐 Neutral")
        else:
            st.success("😊 Positive")

