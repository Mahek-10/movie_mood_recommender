import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib

# Massive mood list provided
moods = [
    "romantic", "sad", "motivated", "happy", "excited", "angry", "relaxed", "fearful", "bored",
    "heartwarming", "emotional", "nostalgic", "inspirational", "introspective", "intense", "tragic",
    "melancholic", "uplifting", "hopeful", "lighthearted", "touching", "dramatic", "powerful", "stylish",
    "quirky", "comedy", "rebellious", "provocative", "clever", "entertaining", "witty", "suspenseful",
    "creepy", "sensual", "thriller", "tense", "satirical", "reflective", "somber", "gritty", "disturbing",
    "psychological", "action", "cold", "playful", "adventurous", "charming", "humorous", "sentimental",
    "intellectual", "mysterious", "violent", "patriotic", "spiritual", "social", "wacky", "confused",
    "funny", "serious", "scary", "wild", "epic", "historical", "informative", "courageous", "magical",
    "inspired", "suspense", "joyful", "brave", "survival", "classic", "biographical", "sci-fi",
    "controversial", "experimental", "legal", "strong", "philosophical", "feelgood"
]

# Build sample training text dynamically (1 sentence per mood)
sample_text = [f"This moment feels very {mood.lower()}." for mood in moods]
df = pd.DataFrame({
    "text": sample_text,
    "mood": moods
})

# Vectorization
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df["text"])
y = df["mood"]

# Train model
model = MultinomialNB()
model.fit(X, y)

# Save model and vectorizer
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print(f"✅ Model trained and saved with {len(moods)} moods.")

