import torch
from transformers import BertTokenizer, BertForSequenceClassification

import random
import numpy as np
import torch

# Load pre-trained BERT model and tokenizer
model_name = 'bert-large-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=3)  # Set num_labels to 3 for positive, negative, neutral

# Set device (GPU if available, else CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Example conversation data
conversation = [
"SPEAKER 1 : Thank you for contacting GDSD. My name is Risha Hamar.  I'm going to lock myself out of my account. I didn't realize my keyboard had turned to  US, so I was typing in for US keyboard when she was in the UK. Sorry, could you please  just repeat it once again?  Can I have my password reset on my computer, please?  Yes, password reset for your account, right?  Sorry? I know my password. It just says my account's locked. I just see my account  locked.  Are you working from office or from home?  From home.  I would surely help you. Are you connected to the VPN?  Yes.  Yes, okay. Please tell me your full name.  Lindsay Hall. ",
"SPEAKER 2 : Could you please spell it?",
"SPEAKER 1 : Lindsay, your last name, please?  Hall, H-A-L-L.  H for hotel, is it?  Yes.",
"SPEAKER 2 : A for alpha? ",
"SPEAKER 1 : Yes.",
"SPEAKER 2 : your system is broken worst service?",
"SPEAKER 1 : Yes.  Lindsay, could you please confirm your email address? Because here I can see your  email address seems to be lindsay.benes.op.com, right?  Yes, that's right. That's my email address, but that's not my main link to my account.  Okay, and could you please confirm your manager's name? ",
"SPEAKER 2 : Charles Sparkman.",
"SPEAKER 1 : Thank you.  I would surely reset the password for you.",
"SPEAKER 2 : Lindsay, I have reset the password for you. So it's welcome at two, digit two, where  W is in uppercase, the rest in smallercase, at, as in at OUP.com, at digit two.  Okay. ",
"SPEAKER 1 : Okay.  Thank you.  Thank you, Lindsay. Thank you for contacting GTSD. Have a great day ahead. Goodbye.",
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
