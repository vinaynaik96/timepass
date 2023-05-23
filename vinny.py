from nltk.sentiment.vader import SentimentIntensityAnalyzer

def analyze_sentiment(text):
    sid = SentimentIntensityAnalyzer()
    sentiment_scores = sid.polarity_scores(text)
    compound_score = sentiment_scores['compound']
    
    if compound_score >= 0.05:
        return 'Positive'
    elif compound_score <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

# Example conversation
conversation = [
    "Helpdesk: Hi there! How can I assist you today?",
    "Client: Hi, I'm having trouble accessing my account. I keep getting an error message.",
    "Helpdesk: I'm sorry to hear that. Could you please provide me with your username so I can look into it?",
    "Client: Sure, my username is johndoe123.",
    "Helpdesk: Thank you, John. Let me check our system. Please bear with me for a moment.",
    "Helpdesk: John, it seems that there was a temporary glitch in our system. I have resolved the issue for you.",
    "Client: That's great! Thank you so much for your help.",
    "Helpdesk: You're welcome, John! If you need any further assistance, feel free to ask. Have a great day!"
]

# Break conversation into three separate conversations
convo1 = conversation[0:2]  # Helpdesk's first response and client's first message
convo2 = conversation[2:4]  # Client's response and Helpdesk's second response
convo3 = conversation[4:6]  # Helpdesk's third response and client's second message

# Analyze sentiment of conversation 1
print("Conversation 1:")
for message in convo1:
    sentiment = analyze_sentiment(message.split(":")[1].strip())
    print(f"{message} (Sentiment: {sentiment})")

print()

# Analyze sentiment of conversation 2
print("Conversation 2:")
for message in convo2:
    sentiment = analyze_sentiment(message.split(":")[1].strip())
    print(f"{message} (Sentiment: {sentiment})")

print()

# Analyze sentiment of conversation 3
print("Conversation 3:")
for message in convo3:
    sentiment = analyze_sentiment(message.split(":")[1].strip())
    print(f"{message} (Sentiment: {sentiment})")
