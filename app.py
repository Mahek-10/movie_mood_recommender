import streamlit as st
import pandas as pd
from mood_model import detect_mood
from ml_model_predictor import predict_mood

# Initialize app session state only once
if "users" not in st.session_state:
    st.session_state["users"] = {"admin": "admin"}  # optional default user
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
if "username" not in st.session_state:
    st.session_state["username"] = ""
if "favorites" not in st.session_state:
    st.session_state["favorites"] = {}

# --- Signup Function ---
def signup():
    st.subheader("🔐 Create New Account")
    new_user = st.text_input("Create Username")
    new_pass = st.text_input("Create Password", type="password")
    if st.button("Sign Up"):
        if new_user == "" or new_pass == "":
            st.warning("Please enter both username and password.")
        elif new_user in st.session_state["users"]:
            st.error("Username already exists!")
        else:
            st.session_state["users"][new_user] = new_pass
            st.success("✅ Account created. Please login.")
            st.experimental_rerun()

# --- Login Function ---
def login():
    st.subheader("🔓 Login")
    user = st.text_input("Username")
    pw = st.text_input("Password", type="password")
    if st.button("Login"):
        if user in st.session_state["users"] and st.session_state["users"][user] == pw:
            st.session_state["logged_in"] = True
            st.session_state["username"] = user
            if user not in st.session_state["favorites"]:
                st.session_state["favorites"][user] = []
            st.success(f"Welcome, {user}!")
            st.experimental_rerun()
        else:
            st.error("Invalid username or password.")

# --- Authentication Page ---
if not st.session_state["logged_in"]:
    st.title("🎬 Movie Mood Recommender")
    auth_mode = st.radio("Select", ["Login", "Sign Up"])
    login() if auth_mode == "Login" else signup()
    st.stop()

# --- Main App ---
st.title("🎬 Movie Mood Recommender")
st.markdown(f"👤 Logged in as: `{st.session_state['username']}`")

user_input = st.text_input("💬 How are you feeling right now?")
use_custom_model = st.checkbox("Use Custom Trained Model")

if user_input:
    mood = predict_mood(user_input) if use_custom_model else detect_mood(user_input)
    st.success(f"🧠 Detected mood: **{mood}**")

    try:
        df = pd.read_csv("movies.csv")
        recs = df[df["mood"].str.lower() == mood.lower()]
        if not recs.empty:
            st.subheader("🎥 Recommended Movies:")
            for idx, row in recs.iterrows():
                col1, col2 = st.columns([5, 1])
                col1.markdown(f"✅ **{row['title']}** ({row['year']}) — *{row['genre']}*")
                if col2.button("❤️ Fav", key=f"{row['title']}_{idx}"):
                    user = st.session_state["username"]
                    if row['title'] not in st.session_state["favorites"][user]:
                        st.session_state["favorites"][user].append(row['title'])
                        st.success(f"Added to favorites: {row['title']}")
        else:
            st.warning("No movies found for this mood.")
    except Exception as e:
        st.error(f"Error loading movie data: {e}")

# --- Favorite Section ---
st.markdown("---")
st.subheader("⭐ Your Favorite Movies")
favs = st.session_state["favorites"].get(st.session_state["username"], [])
if favs:
    for movie in favs:
        st.write(f"🌟 {movie}")
else:
    st.info("No favorites yet.")
