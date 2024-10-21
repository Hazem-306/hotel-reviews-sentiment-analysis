import streamlit as st
import joblib
import numpy as np

# Load the model and vectorizer
model = joblib.load('hotel_review_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Streamlit app
st.title("Hotel Review Sentiment Analysis")

# Input for the review
review_text = st.text_area("Enter the hotel review here:")

# Button to submit the review
if st.button("Predict"):
    # Vectorize the input review
    review_vectorized = vectorizer.transform([review_text])
    
    # Predict the sentiment
    prediction = model.predict(review_vectorized)
    prediction_proba = model.predict_proba(review_vectorized)

    # Display the results
    if prediction[0] == 1:
        st.success("This review is **Good**!")
        st.write(f"Probability of Good Review: {prediction_proba[0][1] * 100:.2f}%")
        st.write(f"Probability of Bad Review: {prediction_proba[0][0] * 100:.2f}%")
    else:
        st.error("This review is **Bad**!")
        st.write(f"Probability of Good Review: {prediction_proba[0][1] * 100:.2f}%")
        st.write(f"Probability of Bad Review: {prediction_proba[0][0] * 100:.2f}%")
