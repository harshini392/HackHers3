import streamlit as st

st.title("BrandPulse AI - Sentiment Monitoring System")
from transformers import pipeline
import pandas as pd


# Load sentiment analysis model
sentiment_model = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

# User input box
review = st.text_input("Enter a review:")

# Predict sentiment
if review:
    result = sentiment_model(review)
    st.write("Sentiment Result:", result[0]["label"])
st.subheader("Dataset Sentiment Analysis")

df = pd.read_csv("reviews.csv", encoding="latin-1")


if st.button("Analyze Dataset"):
    results = sentiment_model(df["review"].tolist())

    sentiments = [r["label"] for r in results]
    df["sentiment"] = sentiments

    st.write(df)

    st.write("Sentiment Count:")
    st.write(df["sentiment"].value_counts())
    st.subheader("Dataset Sentiment Analysis")

df["sentiment"] = df["review"].apply(
    lambda x: sentiment_model(str(x))[0]["label"]
)

st.write(df.head())

