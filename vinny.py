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
    # Extract the speaker and text from the message
    speaker, text = message.split(":")
    text = text.strip()
    
    # Determine the sentiment label for the message
    sentiment = get_sentiment(text)
    
    # Assign sentiment label to the message
    labeled_message = f"{speaker}: {text} ({sentiment})"
    sentiments.append(labeled_message)

# Print the sentiment labels for each message
for message in sentiments:
    print(message)

# Determine the overall sentiment of the conversation
positive_count = sum(1 for message in sentiments if 'Positive' in message)
negative_count = sum(1 for message in sentiments if 'Negative' in message)
overall_sentiment = 'Positive' if positive_count > negative_count else 'Negative'

print("Overall sentiment:", overall_sentiment)
