import streamlit as st
import pandas as pd
from mood_model import detect_mood
from ml_model_predictor import predict_mood

st.title("🎬 Movie Mood Recommender")

user_input = st.text_input("How are you feeling right now?")
use_custom_model = st.checkbox("Use Custom Trained Model")

if user_input:
    if use_custom_model:
        mood = predict_mood(user_input)
        st.write(f"Predicted mood (Custom ML): **{mood}**")
    else:
        mood = detect_mood(user_input)
        st.write(f"Predicted mood (Pre-trained NLP): **{mood}**")

    df = pd.read_csv("movies.csv")
    recs = df[df["mood"] == mood]

    if not recs.empty:
        st.subheader("🎥 Recommended Movies:")
        for title in recs["title"].values:
            st.write(f"✅ {title}")
    else:
        st.warning("No movies found for this mood.")
