🎬 Movie Mood Recommender
Welcome to the Movie Mood Recommender – a beginner-friendly ML project that suggests movies based on your mood using both pre-trained NLP and a custom-trained ML model.

📌 Features
🎭 Detects your mood from a sentence using TextBlob or a Naive Bayes classifier

🍿 Recommends movies based on mood (e.g., happy, sad, romantic, motivated)

🧠 Switch between pre-trained and custom ML models

🌐 Deployable on Streamlit Cloud

💡 No prior ML knowledge needed to run!

📁 Project Structure
bash
Copy
Edit
movie_mood_recommender/
│
├── app.py                  # Main Streamlit app
├── mood_model.py           # Pre-trained TextBlob model
├── custom_model.py         # Script to train your own model (optional)
├── ml_model_predictor.py   # Uses custom ML model for prediction
├── model.pkl               # Trained Naive Bayes model
├── vectorizer.pkl          # Trained vectorizer
├── movies.csv              # Sample movie database
├── requirements.txt        # Python dependencies
└── README.md               # Project overview

📦 Requirements
txt
Copy
Edit
streamlit
textblob
scikit-learn
joblib
pandas

🤖 Tech Stack
Python
Streamlit
TextBlob (pre-trained sentiment analysis)
Scikit-learn (Naive Bayes classifier)

✨ Example Inputs
Mood Input	Detected Mood	Recommended Movie
"I’m feeling so happy today!"	happy	Frozen
"Everything feels sad and dull."	sad	Inside Out
"I want to feel inspired again."	motivated	The Pursuit of Happyness
"I'm in love!"	romantic	La La Land

🛠 Future Improvements
Add real-time movie API (e.g., TMDb)
Personalize recommendations
Add more moods and a bigger dataset
