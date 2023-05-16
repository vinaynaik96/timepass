from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

class ConversationalSentimentAnalyzer:
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def analyze_sentiment(self, conversation):
        try:
            # Tokenize the conversation
            inputs = self.tokenizer.encode_plus(
                conversation,
                add_special_tokens=True,
                padding="longest",
                truncation=True,
                return_tensors="pt"
            )
            inputs = inputs.to(self.device)

            # Perform sentiment analysis
            with torch.no_grad():
                logits = self.model(**inputs).logits
                predicted_labels = torch.argmax(logits, dim=1)
                sentiment_label = "Positive" if predicted_labels.item() == 1 else "Negative"
            
            return sentiment_label

        except Exception as e:
            print("Error analyzing sentiment:", e)
            return None

# Example usage
conversation = "User: How are you?\nBot: I'm doing great! How about you?\nUser: I'm feeling happy today."
analyzer = ConversationalSentimentAnalyzer("bert-base-uncased")
sentiment = analyzer.analyze_sentiment(conversation)
print("Sentiment:", sentiment)
