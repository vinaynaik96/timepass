import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

def get_sentiment_label(sentiment_score):
    if sentiment_score >= 0.05:
        return 'Positive'
    elif sentiment_score <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

def analyze_sentiment(text):
    # Create a SentimentIntensityAnalyzer object
    sid = SentimentIntensityAnalyzer()
    
    # Analyze the sentiment of the text
    sentiment_scores = sid.polarity_scores(text)
    
    # Return the compound sentiment score
    return sentiment_scores['compound']

# Example conversation between helpdesk and client
conversation = [
    {'speaker': 'helpdesk', 'text': 'How can I assist you today?'},
    {'speaker': 'client', 'text': 'I'm having trouble with your product.'},
    {'speaker': 'helpdesk', 'text': 'I apologize for the inconvenience. Please provide more details about the issue.'},
    {'speaker': 'client', 'text': 'The product keeps crashing whenever I try to open it.'},
    {'speaker': 'helpdesk', 'text': 'I understand your frustration. We'll do our best to resolve this issue for you.'},
]

# Analyze the sentiment of each message in the conversation
sentiment_scores = []
sentiment_labels = []
for message in conversation:
    sentiment_score = analyze_sentiment(message['text'])
    sentiment_scores.append(sentiment_score)
    sentiment_label = get_sentiment_label(sentiment_score)
    sentiment_labels.append(sentiment_label)

# Print the sentiment labels
for i, message in enumerate(conversation):
    print(f"Message {i+1} ({message['speaker']}): '{message['text']}' - Sentiment: {sentiment_labels[i]}")
