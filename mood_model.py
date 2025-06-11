from textblob import TextBlob

def detect_mood(text):
    text_lower = text.lower()
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity

    mood_keywords = {
        "romantic": ["love", "romantic", "date", "passion"],
        "sad": ["sad", "cry", "lonely", "depress", "tears"],
        "motivated": ["motivate", "productive", "achieve", "goal", "focus", "determined"],
        "happy": ["happy", "joy", "great", "wonderful", "joyful"],
        "excited": ["excite", "thrill", "awesome", "can’t wait", "pumped"],
        "angry": ["angry", "mad", "furious", "rage"],
        "relaxed": ["relax", "calm", "peace", "soothing"],
        "fearful": ["fear", "scared", "terrified", "afraid"],
        "bored": ["bored", "dull", "yawn", "uninterested"],
        "heartwarming": ["heartwarming", "wholesome", "sweet"],
        "emotional": ["emotional", "moving", "touching"],
        "nostalgic": ["nostalgic", "remember", "childhood", "past"],
        "inspirational": ["inspire", "uplift", "empower", "encourage"],
        "introspective": ["introspective", "reflect", "think deeply"],
        "intense": ["intense", "powerful", "strong emotions"],
        "tragic": ["tragic", "heartbreak", "disaster"],
        "melancholic": ["melancholic", "blue", "gloomy"],
        "uplifting": ["uplifting", "hope", "positive"],
        "hopeful": ["hopeful", "faith", "believe"],
        "lighthearted": ["lighthearted", "breezy", "casual"],
        "touching": ["touching", "emotional", "sweet"],
        "dramatic": ["dramatic", "serious", "conflict"],
        "powerful": ["powerful", "impactful", "strong"],
        "stylish": ["stylish", "fashion", "cool"],
        "quirky": ["quirky", "weird", "odd", "offbeat"],
        "comedy": ["comedy", "laugh", "funny"],
        "rebellious": ["rebellious", "revolt", "defy", "resist"],
        "provocative": ["provocative", "controversial", "challenge"],
        "clever": ["clever", "smart", "intelligent"],
        "entertaining": ["entertaining", "fun", "engaging"],
        "witty": ["witty", "sharp", "sarcastic"],
        "suspenseful": ["suspense", "tense", "thrill"],
        "creepy": ["creepy", "chilling", "unsettling"],
        "sensual": ["sensual", "passionate", "intimate"],
        "thriller": ["thriller", "edge", "tense", "investigation"],
        "tense": ["tense", "nervous", "pressure"],
        "satirical": ["satirical", "mock", "parody"],
        "reflective": ["reflective", "thoughtful", "meditative"],
        "somber": ["somber", "dark", "serious"],
        "gritty": ["gritty", "raw", "realistic"],
        "disturbing": ["disturbing", "unsettling", "graphic"],
        "psychological": ["psychological", "mind", "mental", "twist"],
        "action": ["fight", "explosion", "chase", "action"],
        "cold": ["cold", "distant", "emotionless"],
        "playful": ["playful", "fun", "game"],
        "adventurous": ["adventure", "journey", "explore"],
        "charming": ["charming", "cute", "adorable"],
        "humorous": ["humor", "hilarious", "laugh"],
        "sentimental": ["sentimental", "memory", "heart"],
        "intellectual": ["intellectual", "deep", "philosophical"],
        "mysterious": ["mysterious", "mystery", "secret"],
        "violent": ["violent", "blood", "fight"],
        "patriotic": ["patriotic", "nation", "flag", "country"],
        "spiritual": ["spiritual", "divine", "soul"],
        "social": ["social", "issues", "justice"],
        "wacky": ["wacky", "crazy", "absurd"],
        "confused": ["confused", "lost", "unsure"],
        "funny": ["funny", "lol", "laughing"],
        "serious": ["serious", "important", "critical"],
        "scary": ["scary", "horror", "haunted"],
        "wild": ["wild", "crazy", "chaotic"],
        "epic": ["epic", "grand", "massive"],
        "historical": ["historical", "history", "past"],
        "informative": ["informative", "educational", "facts"],
        "courageous": ["brave", "courage", "fearless"],
        "magical": ["magical", "fantasy", "fairy"],
        "inspired": ["inspired", "driven", "ambition"],
        "joyful": ["joyful", "cheerful", "delighted"],
        "brave": ["brave", "hero", "fight"],
        "survival": ["survive", "survival", "struggle"],
        "classic": ["classic", "timeless", "old"],
        "biographical": ["biography", "real life", "true story"],
        "sci-fi": ["sci-fi", "space", "robot", "alien"],
        "controversial": ["controversial", "debated", "uncomfortable"],
        "experimental": ["experimental", "artsy", "unconventional"],
        "legal": ["court", "judge", "lawyer"],
        "strong": ["strong", "resilient", "bold"],
        "philosophical": ["philosophy", "meaning", "existential"],
        "feelgood": ["feelgood", "warm", "happy ending"]
    }

    # Keyword-based matching
    for mood, keywords in mood_keywords.items():
        if any(keyword in text_lower for keyword in keywords):
            return mood

    # Fallback: polarity based mood
    if polarity >= 0.6:
        return "happy"
    elif polarity >= 0.3:
        return "joyful"
    elif polarity <= -0.5:
        return "sad"
    elif polarity <= -0.3:
        return "serious"
    else:
        return "neutral"
