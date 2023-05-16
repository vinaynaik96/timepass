from transformers import AutoModelForSequenceClassification, AutoTokenizer

class ConversationalSentimentAnalyzer:
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def analyze_sentiment(self, conversation):
        try:
            inputs = self.tokenizer.encode_plus(
                conversation,
                add_special_tokens=True,
                padding="longest",
                truncation=True,
                return_tensors="pt"
            )
            inputs = inputs.to(self.device)

            with torch.no_grad():
                logits = self.model(**inputs).logits
                predicted_labels = torch.argmax(logits, dim=1)
                sentiment_label = "Positive" if predicted_labels.item() == 1 else "Negative"

            # Get word-level sentiment scores
            tokens = self.tokenizer.tokenize(conversation)
            scores = torch.nn.functional.softmax(logits, dim=1).squeeze().tolist()
            word_sentiment_scores = {token: score for token, score in zip(tokens, scores)}

            return sentiment_label, word_sentiment_scores

        except Exception as e:
            print("Error analyzing sentiment:", e)
            return None, {}

# Example usage
conversation = "User: How are you?\nBot: I'm doing great! How about you?\nUser: I'm feeling happy today."
analyzer = ConversationalSentimentAnalyzer("roberta-base")
sentiment, word_sentiment_scores = analyzer.analyze_sentiment(conversation)

if sentiment is not None:
    print("Sentiment:", sentiment)

if word_sentiment_scores:
    print("Word-level sentiment scores:")
    for token, score in word_sentiment_scores.items():
        print(token, ":", score)
