import streamlit as st
import pandas as pd
from mood_model import detect_mood
from ml_model_predictor import predict_mood

# In-memory user store for demo purposes
if "users" not in st.session_state:
    st.session_state.users = {}  # {username: password}
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""

# Sign-up function
def signup():
    st.subheader("🔐 Create New Account")
    new_user = st.text_input("New Username")
    new_pass = st.text_input("New Password", type="password")
    if st.button("Sign Up"):
        if new_user in st.session_state.users:
            st.error("User already exists.")
        else:
            st.session_state.users[new_user] = new_pass
            st.success("Account created! Please log in.")

# Login function
def login():
    st.subheader("🔓 Login")
    user = st.text_input("Username")
    passwd = st.text_input("Password", type="password")
    if st.button("Login"):
        if user in st.session_state.users and st.session_state.users[user] == passwd:
            st.session_state.logged_in = True
            st.session_state.username = user
            st.success(f"Welcome back, {user}!")
        else:
            st.error("Invalid username or password.")

# Auth system
if not st.session_state.logged_in:
    page = st.selectbox("Choose Option", ["Login", "Sign Up"])
    if page == "Login":
        login()
    else:
        signup()
    st.stop()

# Main App after login
st.title("🎬 Movie Mood Recommender")
st.write(f"👤 Logged in as: `{st.session_state.username}`")

user_input = st.text_input("How are you feeling right now?")
use_custom_model = st.checkbox("Use Custom Trained Model")

if user_input:
    if use_custom_model:
        mood = predict_mood(user_input)
        st.write(f"Predicted mood (Custom ML): **{mood}**")
    else:
        mood = detect_mood(user_input)
        st.write(f"Predicted mood (Pre-trained NLP): **{mood}**")

    try:
        df = pd.read_csv("movies.csv")
        recs = df[df["mood"].str.lower() == mood.lower()]
        if not recs.empty:
            st.subheader("🎥 Recommended Movies:")
            for title in recs["title"].values:
                st.write(f"✅ {title}")
        else:
            st.warning("No movies found for this mood.")
    except Exception as e:
        st.error(f"Error loading movie data: {e}")
