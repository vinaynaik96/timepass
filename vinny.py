import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

def get_sentiment(text):
    sid = SentimentIntensityAnalyzer()
    sentiment_scores = sid.polarity_scores(text)
    sentiment_label = 'Positive' if sentiment_scores['compound'] >= 0 else 'Negative'
    return sentiment_label

conversation = [
    "User: How are you?",
    "Bot: I'm doing great! How about you?",
    "User: I'm feeling happy today"
]

# Analyze sentiment for each message in the conversation
sentiments = []
for message in conversation:
    # Extract the text from the message (excluding the speaker prefix)
    text = message.split(":")[1].strip()
    sentiment = get_sentiment(text)
    sentiments.append(sentiment)

# Determine the overall sentiment of the conversation
overall_sentiment = max(set(sentiments), key=sentiments.count)

print("Overall sentiment:", overall_sentiment)
