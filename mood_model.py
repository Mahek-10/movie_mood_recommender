from textblob import TextBlob

def detect_mood(text):
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity
    if polarity > 0.5:
        return 'happy'
    elif polarity < -0.2:
        return 'sad'
    elif 'love' in text.lower():
        return 'romantic'
    elif 'motivate' in text.lower() or polarity > 0.1:
        return 'motivated'
    else:
        return 'neutral'
