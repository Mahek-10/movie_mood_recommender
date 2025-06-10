import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib

# Training data
data = {
    "text": [
        "I feel so happy and excited!",
        "I'm feeling very low and sad",
        "This is such a romantic day",
        "I want to do something productive",
        "Feeling energetic and thrilled"
    ],
    "mood": ["happy", "sad", "romantic", "motivated", "excited"]
}

df = pd.DataFrame(data)

# Train model
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df["text"])
y = df["mood"]

model = MultinomialNB()
model.fit(X, y)

# Save model and vectorizer
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
