import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class EmotionSentimentAnalyzer:
    def __init__(self, emotion_model_name, sentiment_model_name):
        self.emotion_tokenizer = AutoTokenizer.from_pretrained(emotion_model_name)
        self.emotion_model = AutoModelForSequenceClassification.from_pretrained(emotion_model_name)
        self.sentiment_tokenizer = AutoTokenizer.from_pretrained(sentiment_model_name)
        self.sentiment_model = AutoModelForSequenceClassification.from_pretrained(sentiment_model_name)
        self.emotion_labels = ["joy", "sadness", "anger", "fear", "love", "surprise"]
        self.sentiment_labels = ["Negative", "Neutral", "Positive"]

    def analyze(self, conversation):
        emotions = self._analyze_emotions(conversation)
        sentiments = self._analyze_sentiments(conversation)
        overall_sentiment = self._determine_overall_sentiment(sentiments)
        return emotions, sentiments, overall_sentiment

    def _analyze_emotions(self, conversation):
        emotions = []
        for message in conversation:
            text = message.split(":")[1].strip()
            encoded_input = self.emotion_tokenizer.encode_plus(text, padding=True, truncation=True, return_tensors="pt")
            input_ids = encoded_input["input_ids"]
            attention_mask = encoded_input["attention_mask"]

            with torch.no_grad():
                logits = self.emotion_model(input_ids, attention_mask=attention_mask).logits

            emotion_idx = torch.argmax(logits, dim=1).item()
            emotion_label = self.emotion_labels[emotion_idx]
            emotions.append(emotion_label)

        return emotions

    def _analyze_sentiments(self, conversation):
        sentiments = []
        for message in conversation:
            text = message.split(":")[1].strip()
            encoded_input = self.sentiment_tokenizer.encode_plus(text, padding=True, truncation=True, return_tensors="pt")
            input_ids = encoded_input["input_ids"]
            attention_mask = encoded_input["attention_mask"]

            with torch.no_grad():
                logits = self.sentiment_model(input_ids, attention_mask=attention_mask).logits

            sentiment_idx = torch.argmax(logits, dim=1).item()
            sentiment_label = self.sentiment_labels[sentiment_idx]
            sentiments.append(sentiment_label)

        return sentiments

    def _determine_overall_sentiment(self, sentiments):
        if all(sent == 'Positive' for sent in sentiments):
            return 'Positive'
        elif all(sent == 'Negative' for sent in sentiments):
            return 'Negative'
        else:
            return 'Mixed'


# Example usage
emotion_model_name = "cardiffnlp/twitter-roberta-base-emoji"
sentiment_model_name = "bert-base-uncased"

analyzer = EmotionSentimentAnalyzer(emotion_model_name, sentiment_model_name)

conversation = [
    "User: How are you?",
    "Bot: I'm doing great! How about you?",
    "User: I'm feeling happy today"
]

emotions, sentiments, overall_sentiment = analyzer.analyze(conversation)

# Print results
for message, emotion, sentiment in zip(conversation, emotions, sentiments):
    print(f"{message} (Emotion: {emotion}, Sentiment: {sentiment})")

print(f"\nOverall Sentiment: {overall_sentiment}")
