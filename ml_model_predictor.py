import joblib

model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

def predict_mood(text):
    X = vectorizer.transform([text])
    return model.predict(X)[0]
