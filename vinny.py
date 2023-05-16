from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load pre-trained model and tokenizer
model_name = "microsoft/DialoGPT-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Define the sentiment analysis function
def get_conversation_sentiment(conversation):
    # Encode the conversation
    encoded_input = tokenizer.encode(conversation, return_tensors="pt")

    # Generate response from the model
    with torch.no_grad():
        generated = model.generate(
            encoded_input.input_ids,
            max_length=100,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7,
        )

    # Decode and extract sentiment
    generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
    sentiment = get_sentiment(generated_text)  # You need to implement the sentiment analysis function

    return sentiment

# Example conversation
conversation = "User: How are you?\nBot: I'm doing great! How about you?\nUser: I'm feeling happy today."

# Get sentiment of the conversation
sentiment = get_conversation_sentiment(conversation)
print("Sentiment:", sentiment)
