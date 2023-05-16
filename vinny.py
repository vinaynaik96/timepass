import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Load pre-trained BERT model and tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=3)  # Set num_labels to 3 for positive, negative, neutral

# Set device (GPU if available, else CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Example conversation data
conversation = [
    "User: How are you?",
    "Bot: I'm doing great! How about you?",
    "User: I'm feeling happy today"
]

# Process conversation data and prepare input for model
inputs = []
for message in conversation:
    # Extract the text from the message (excluding the speaker prefix)
    text = message.split(":")[1].strip()
    inputs.append(text)

# Tokenize and encode the input
encoded_input = tokenizer.batch_encode_plus(
    inputs,
    truncation=True,
    padding=True,
    max_length=128,
    return_tensors='pt'
)

# Move input tensors to the device
input_ids = encoded_input['input_ids'].to(device)
attention_mask = encoded_input['attention_mask'].to(device)

# Perform inference
model.eval()
with torch.no_grad():
    outputs = model(input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    predicted_labels = torch.argmax(logits, dim=1).cpu().numpy()

# Map predicted labels to sentiment labels
sentiment_labels = ["Negative", "Neutral", "Positive"]
sentiments = [sentiment_labels[label] for label in predicted_labels]

# Print sentiment labels for each message in the conversation
for message, sentiment in zip(conversation, sentiments):
    print(f"{message} ({sentiment})")
