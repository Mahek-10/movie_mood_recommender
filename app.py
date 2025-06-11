import streamlit as st
import pandas as pd
import json
import os
from mood_model import detect_mood
from ml_model_predictor import predict_mood

# Initialize session state
if "users" not in st.session_state:
    st.session_state.users = {}
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""
if "favorites" not in st.session_state:
    st.session_state.favorites = {}

# Load movie data
@st.cache_data
def load_movies():
    try:
        return pd.read_csv("movies.csv")
    except:
        return pd.DataFrame(columns=["title", "mood", "genre", "year"])

movies_df = load_movies()

# Styling
st.markdown("""
    <style>
    body {
        background-color: #f0f2f6;
    }
    .title {
        font-size: 36px;
        color: #FF4B4B;
        font-weight: bold;
    }
    .subtitle {
        font-size: 24px;
        font-weight: 600;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# --- Authentication ---
def signup():
    st.subheader("🔐 Create Account")
    user = st.text_input("New Username")
    pw = st.text_input("New Password", type="password")
    if st.button("Sign Up"):
        if user in st.session_state.users:
            st.error("User already exists.")
        else:
            st.session_state.users[user] = pw
            st.success("Account created! Please login.")

def login():
    st.subheader("🔓 Login")
    user = st.text_input("Username")
    pw = st.text_input("Password", type="password")
    if st.button("Login"):
        if user in st.session_state.users and st.session_state.users[user] == pw:
            st.session_state.logged_in = True
            st.session_state.username = user
            if user not in st.session_state.favorites:
                st.session_state.favorites[user] = []
            st.success(f"Welcome back, {user}!")
        else:
            st.error("Invalid credentials")

if not st.session_state.logged_in:
    st.title("🎬 Movie Mood Recommender")
    choice = st.radio("Select Option", ["Login", "Sign Up"])
    login() if choice == "Login" else signup()
    st.stop()

# --- Main App ---
st.title("🎬 Movie Mood Recommender")
st.write(f"👤 Logged in as: `{st.session_state.username}`")

user_input = st.text_input("💬 How are you feeling right now?")
use_custom_model = st.checkbox("Use Custom ML Mood Predictor")

if user_input:
    mood = predict_mood(user_input) if use_custom_model else detect_mood(user_input)
    st.success(f"🧠 Detected mood: **{mood}**")

    recs = movies_df[movies_df["mood"].str.lower() == mood.lower()]
    if not recs.empty:
        st.subheader("🎥 Recommended Movies for Your Mood:")
        for idx, row in recs.iterrows():
            col1, col2 = st.columns([5, 1])
            col1.markdown(f"✅ **{row['title']}** ({row['year']}) — *{row['genre']}*")
            if col2.button("❤️ Fav", key=f"fav_{idx}"):
                st.session_state.favorites[st.session_state.username].append(row['title'])
                st.success(f"Added '{row['title']}' to favorites.")
    else:
        st.warning("No movie found for this mood.")

# --- Favorite Section ---
st.markdown("---")
st.subheader("❤️ Your Favorites")
user_favs = st.session_state.favorites.get(st.session_state.username, [])
if user_favs:
    for fav in user_favs:
        st.write(f"⭐ {fav}")
else:
    st.info("You haven't added any favorites yet.")
