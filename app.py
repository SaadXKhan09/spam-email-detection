import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load dataset
df = pd.read_csv("emails.csv")

X = df["text"]
y = df["spam"]

# Train model
vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)

model = MultinomialNB()
model.fit(X_vec, y)

# Prediction function
def predict_spam(text):
    text_vec = vectorizer.transform([text])
    prediction = model.predict(text_vec)
    return "Spam" if prediction[0] == 1 else "Not Spam"

# UI
st.title("📧 Email Spam Detection")

msg = st.text_area("Enter your message")

if st.button("Predict"):
    result = predict_spam(msg)
    st.write("Result:", result)