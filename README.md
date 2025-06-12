🎥 Movie Mood Recommender

Lights, camera, recommendation! 🎬 Craving the perfect film to match your current vibe? Look no further than the Movie Mood Recommender. This is your go-to source for discovering cinematic gems tailored to your specific mood. Whether you're feeling happy, sad, adventurous, or romantic, we'll help you find the ideal movie to enhance your experience.

Welcome to the Movie Mood Recommender – a beginner-friendly ML project that suggests movies based on your mood using both pre-trained NLP and a custom-trained ML model.

**INTRODUCTION**:
With so much content to watch on streaming platforms, selecting a movie to watch can be daunting. With so many options, viewers find it difficult to pick something that reflects their mood at the moment. This project addresses that by coming up with a smart and intuitive Movie Mood Recommender system.By leveraging the capability of Natural Language Processing (NLP) and Machine Learning (ML), this app identifies a user's mood from a basic sentence. It then suggests movies that resonate with that feeling — whether the user is feeling happy, sad, romantic, motivated, bored, or otherwise. The app also supports more than 80 distinct moods and uses a hand-picked corpus of mood-tagged movies to make very individual recommendations. An option to toggle between using a pre-trained TextBlob model and a custom-trained Naive Bayes classifier.This is a beginner-friendly project, deployable on Streamlit Cloud, and does not need any machine learning knowledge to execute.
It's perfect for students, developers, and movie lovers who want to dive into AI-fueled personalization. With a single click, your mood becomes your movie matchmaker.
Let your emotions be your guide in cinema! ????️✨

🧠 **Technologies and Tools Used**
This project combines modern machine learning, natural language processing, and web development tools to create a seamless, real-time movie recommendation experience:
**Technology	Purpose**
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

**Features**
🎭 Detects your mood from a sentence using TextBlob or a Naive Bayes classifier
🍿 Recommends movies based on mood (e.g., happy, sad, romantic, motivated)
🧠 Switch between pre-trained and custom ML models
🌐 Deployable on Streamlit Cloud
💡 No prior ML knowledge needed to run!

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
**🛠 Error Handling**
except Exception as e:
    st.error(f"Error loading movie data: {e}")


🛠 Future Improvements
Add real-time movie API (e.g., TMDb)
Personalize recommendations
Add more moods and a bigger dataset
