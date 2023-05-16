import torch
from transformers import BertTokenizer, BertForSequenceClassification
from deepmoji.model_def import deepmoji_transfer

class EmotionSentimentAnalyzer:
    def __init__(self, deepmoji_model_path, deepmoji_vocab_path, bert_model_name):
        # Load pre-trained DeepMoji model and tokenizer for emotion detection
        self.deepmoji_model = deepmoji_transfer.DeepMojiModel(deepmoji_model_path, deepmoji_vocab_path)
        self.deepmoji_tokenizer = deepmoji_transfer.get_tokenizer(deepmoji_vocab_path)
        
        # Load pre-trained BERT model and tokenizer for sentiment analysis
        self.bert_tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        self.bert_model = BertForSequenceClassification.from_pretrained(bert_model_name, num_labels=3)
        
        # Set device (GPU if available, else CPU)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.deepmoji_model.to(self.device)
        self.bert_model.to(self.device)
        
        self.emotion_labels = deepmoji_transfer.get_labels()
        self.sentiment_labels = ["Negative", "Neutral", "Positive"]

    def analyze(self, conversation):
        inputs = []
        for message in conversation:
            text = message.split(":")[1].strip()
            inputs.append(text)
        
        emotions = self._detect_emotions(inputs)
        sentiments = self._analyze_sentiments(inputs)
        overall_sentiment = self._determine_overall_sentiment(sentiments)
        
        # Return results
        return emotions, sentiments, overall_sentiment
    
    def _detect_emotions(self, inputs):
        emotion_predictions = self.deepmoji_model(self.deepmoji_tokenizer.encode(inputs, return_tensors='pt').to(self.device))
        emotion_predicted_labels = torch.argmax(emotion_predictions, dim=1).cpu().numpy()
        emotions = [self.emotion_labels[label] for label in emotion_predicted_labels]
        return emotions
    
    def _analyze_sentiments(self, inputs):
        encoded_input = self.bert_tokenizer.batch_encode_plus(
            inputs,
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors='pt'
        )
        input_ids = encoded_input['input_ids'].to(self.device)
        attention_mask = encoded_input['attention_mask'].to(self.device)
        outputs = self.bert_model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predicted_labels = torch.argmax(logits, dim=1).cpu().numpy()
        sentiments = [self.sentiment_labels[label] for label in predicted_labels]
        return sentiments
    
    def _determine_overall_sentiment(self, sentiments):
        if all(sent == 'Positive' for sent in sentiments):
            return 'Positive'
        elif all(sent == 'Negative' for sent in sentiments):
            return 'Negative'
        else:
            return 'Mixed'

# Example usage
deepmoji_model_path = 'path_to_pretrained_deepmoji_model'
deepmoji_vocab_path = 'path_to_deepmoji_vocab'
bert_model_name = 'bert-base-uncased'

analyzer = EmotionSentimentAnalyzer(deepmoji_model_path, deepmoji_vocab_path, bert_model_name)

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

