**🎥MOVIE MOOD RECOMMENDER**

Lights, camera, recommendation! 🎬 Craving the perfect film to match your current vibe? Look no further than the Movie Mood Recommender. This is your go-to source for discovering cinematic gems tailored to your specific mood. Whether you're feeling happy, sad, adventurous, or romantic, we'll help you find the ideal movie to enhance your experience.

Welcome to the Movie Mood Recommender – a beginner-friendly ML project that suggests movies based on your mood using both pre-trained NLP and a custom-trained ML model.

**INTRODUCTION**:
With so much content to watch on streaming platforms, selecting a movie to watch can be daunting. With so many options, viewers find it difficult to pick something that reflects their mood at the moment. This project addresses that by coming up with a smart and intuitive Movie Mood Recommender system.By leveraging the capability of Natural Language Processing (NLP) and Machine Learning (ML), this app identifies a user's mood from a basic sentence. It then suggests movies that resonate with that feeling — whether the user is feeling happy, sad, romantic, motivated, bored, or otherwise. The app also supports more than 80 distinct moods and uses a hand-picked corpus of mood-tagged movies to make very individual recommendations. An option to toggle between using a pre-trained TextBlob model and a custom-trained Naive Bayes classifier.This is a beginner-friendly project, deployable on Streamlit Cloud, and does not need any machine learning knowledge to execute.
It's perfect for students, developers, and movie lovers who want to dive into AI-fueled personalization. With a single click, your mood becomes your movie matchmaker.
Let your emotions be your guide in cinema! ????️✨

🧠 **Technologies and Tools Used**
This project combines modern machine learning, natural language processing, and web development tools to create a seamless, real-time movie recommendation experience:
**Technology Purpose**
🐍 Python:	Core programming language used throughout the project
🧾 Pandas:	Data loading, cleaning, and manipulation for movies.csv
🔢 Scikit-learn	:Training a Naive Bayes classifier and vectorizing text using CountVectorizer
🧠 TextBlob: Pre-trained sentiment analysis for rule-based mood detection
💻 Streamlit: Web framework for building and deploying the interactive UI
💾 Joblib: Saving and loading the ML model and vectorizer
📊 NumPy: Efficient handling of arrays and numerical operations (if needed)
📁 CSV Dataset: Contains 1000+ mood-tagged movie entries
These tools were selected to ensure that the system is:
Easy to use 💡
Fast to prototype ⚡
Beginner-friendly 🎓
Ready for deployment 🚀

**📊 Data Descriptions**
The movies.csv file is a structured dataset used to map movies to specific moods. It contains information such as movie titles, mood labels, genre, and release year. Each row in the dataset represents one movie.
Below is a breakdown of the key features:
Column Name	Description
🎞 Title:The name of the movie (e.g., La La Land, Inception)
🎭 Mood: The emotion associated with the movie (e.g., happy, sad, romantic, motivated)
🎬 Genre: The primary genre of the movie (e.g., Drama, Comedy, Action, Sci-Fi)
📅 Year: The release year of the movie (e.g., 2006, 2022)
🧠 These mood tags are used to filter and recommend relevant movies based on the predicted emotional tone of the user's input.

**✨ Features**
🎭 Mood Detection from Text: Enter how you feel in natural language (e.g., "I’m feeling excited!") and the app will detect your mood using either a pre-trained NLP model (TextBlob) or a custom-trained ML classifier (Naive Bayes).
🍿 Smart Movie Recommendations: Based on the detected mood, the app suggests movies from a curated dataset of 1000+ titles, tagged by genre, mood, and year.
🔄 Model Toggle:Easily switch between rule-based NLP and machine learning-based prediction for flexible mood detection.
🔐 User Login and Signup: Secure login system with session-based memory. Each user can log in or create an account without needing a database.
❤️ Favorites: System Logged-in users can save their favorite movies with a single click and view their personal movie list anytime during their session.
🧠 Supports 80+ Mood Categories: From "happy" and "romantic" to "philosophical", "witty", "introspective", and beyond — enjoy deep personalization.
🖥 Streamlit Web Interface: Clean, fast, and interactive user interface built with Streamlit, ideal for deployment on Streamlit Cloud.
📦 No ML Experience Needed:Beginner-friendly project — everything runs with a few clicks and includes pre-trained models.
🚀 Fast and Lightweight:Loads instantly, predicts instantly. No heavy dependencies, large models, or external APIs needed.
🔧 Easily Extendable: Want to add more moods? Expand the dataset? Add ratings or genres? The code is clean and modular for easy upgrades.

**Project Structure**
movie_mood_recommender/
│
├── app.py                  # 🎯 Main Streamlit app for UI and logic
├── mood_model.py           # 🧠 Rule-based mood detection using TextBlob
├── ml_model_predictor.py   # 🤖 Mood prediction using trained ML model
├── custom_model.py         # 🛠️ Script to train Naive Bayes model with 80+ moods
│
├── model.pkl               # 🧾 Saved Naive Bayes classifier
├── vectorizer.pkl          # 🧾 Saved CountVectorizer for text transformation
│
├── movies.csv              # 🎬 Movie dataset tagged with moods, genres, and years
├── requirements.txt        # 📦 List of Python dependencies
└── README.md               # 📘 Project documentation


**📦 Requirements**
streamlit>=1.30.0
textblob>=0.17.1
scikit-learn>=1.2.0
pandas>=2.0.0
joblib>=1.2.0
numpy>=1.23.0


**🧠 Tech Stack**
Layer,Technology Used,Description
👨‍💻 Frontend UI,Streamli,Interactive and fast web interface for input/output
🧠 Mood Detection,TextBlob (Rule-based),NLP sentiment analysis for basic mood classification
🧠 Mood Detection,Scikit-learn (ML-based),Custom-trained Naive Bayes classifier for advanced mood detection
🔤 Text Processing,CountVectorizer,Converts input text into numerical features
🗃️ Data Handling,Pandas,Reading and filtering movies.csv movie database
💾 Model Storage,Joblib,Efficient model and vectorizer serialization
📊 Core Libraries,NumPy,Underlying numerical computations (used by other packages)
This tech stack ensures that the app is:
Lightweight ⚡
Easy to deploy 🚀
Beginner-friendly 🎓
Fully customizable 🔧

**📦 Import Statements**
"import streamlit as st
import pandas as pd
from mood_model import detect_mood
from ml_model_predictor import predict_mood"

streamlit: Used to create the interactive web UI.
pandas: To read and process the movies.csv file.
detect_mood: Function using TextBlob for basic mood detection.
predict_mood: Function using a trained ML model (Naive Bayes) to detect mood.
**🧠 Session State Initialization**
if "users" not in st.session_state:
    st.session_state["users"] = {"admin": "admin"}  # default user
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
if "username" not in st.session_state:
    st.session_state["username"] = ""
if "favorites" not in st.session_state:
    st.session_state["favorites"] = {}
Streamlit doesn’t retain variables across page refreshes, so session_state keeps important user data:
users: Dictionary of username-password pairs.
logged_in: Tracks if a user is logged in.
username: Stores current session’s user.
favorites: Stores favorite movies per user.
**🔐 Sign-Up Function**
def signup():
UI for new users to register.
Prevents duplicate usernames.
Stores credentials in session_state["users"].
**🔓 Login Function**
def login():
UI for existing users to log in.
Verifies credentials and updates session state.
Initializes empty favorites list if first-time login.
**🔑 Authentication Logic**
if not st.session_state["logged_in"]:
Displays a radio button to toggle between "Login" and "Sign Up".
Calls the appropriate function.
st.stop() halts execution unless the user is logged in.
**🎬 Main App Content**
st.title("Movie Mood Recommender")
st.markdown(f"👤 Logged in as: `{st.session_state['username']}`")
**💬 Mood Input & Detection**
user_input = st.text_input("How are you feeling right now?")
use_custom_model = st.checkbox("Use Custom Trained Model")
**🧠 Predict Mood & Filter Recommendations**
mood = predict_mood(...) or detect_mood(...)
df = pd.read_csv("movies.csv")
recs = df[df["mood"].str.lower() == mood.lower()]
**🎥 Display Movie Recommendations**
for idx, row in recs.iterrows():
Allows user to click a ❤️ button to add to favorites.
**⭐ Favorites Section**
st.markdown("---")
st.subheader("⭐ Your Favorite Movies")
Shows all favorite movies saved by the logged-in user.
Pulls from session_state["favorites"][username].
**🛠 Error Handling**
except Exception as e:
    st.error(f"Error loading movie data: {e}")

**✅ Summary of custom_model.py**
moods:List of mood labels the model can recognize
sample_text:Basic sentence for each mood to simulate input
CountVectorizer:Converts text to numerical features
MultinomialNB: Trains a text classification model
joblib: Saves model and vectorizer for later use
model.pkl, vectorizer.pkl: Used in the Streamlit app to detect moods

**✅ Summary of ml_model_predictor.py**
predict_mood()	Combines both: preprocesses input and outputs the predicted mood
model.pkl: A Multinomial Naive Bayes classifier trained to classify text into one of 80+ moods.
vectorizer.pkl: A CountVectorizer trained on the same text used during model training — it converts input text into the correct numerical format (bag-of-words).
def predict_mood(text):
Defines a function that accepts one input — the user's text (e.g., "I'm feeling excited and motivated!").
X = vectorizer.transform([text])
Converts the input string into a format that the model understands:
vectorizer.transform() tokenizes the input based on the vocabulary learned during training.
[text] wraps the string in a list because the model expects a list or array of inputs.
return model.predict(X)[0]
Uses the trained model to predict the mood of the input.
.predict(X) returns a list/array of predictions, so [0] takes the first result.

**✅ Summary Table of mood_model.py**
Strategy: Method Used: Output Example
Keyword match:"furious", "rage":angry
Keyword match:"wholesome", "sweetheartwarming
Polarity > 0.6	(very positive)	happy
Polarity < -0.5	(very negative)	sad
No strong signal (neutral tone)	neutral

**Deployment**
On Streamlit
Link: https://moviemoodrecommender-27d7h5cstjye447eq3snvr.streamlit.app/

**Results**
After deployment on Streamlit

**First Page**
![Screenshot 2025-06-12 112121](https://github.com/user-attachments/assets/92e03738-a6d2-4624-84a4-f3806c69ff81)

**Sign Up: To create an account**
![Screenshot 2025-06-12 112304](https://github.com/user-attachments/assets/8beaef84-6146-4038-b1a1-b24c7ae588e8)

**Login**
![Screenshot 2025-06-12 112338](https://github.com/user-attachments/assets/a657af5f-9bae-42cf-acca-bced33cb76b1)

**Movie mood recommender page**
![Screenshot 2025-06-12 112400](https://github.com/user-attachments/assets/4af251a7-ec09-4c35-a378-108c7bab88fa)


**NOTE: ALL THE DATA PROVIDED IN CSV AND PKL FILES WAS TAKEN THROUGH CHATGPT AND THE CODES WERE EVEN ENHANCED USING CHATGPT**

**🛠 Future Improvements**
-Integrate real-time movie APIs like TMDb for dynamic recommendations 
=Expand mood detection dataset for better accuracy
-Add collaborative filtering for personalized suggestions
-Support multi-language inputs and voice-based mood input
