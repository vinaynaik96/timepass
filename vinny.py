import pandas as pd
from collections import Counter
from transformers import pipeline, BertTokenizer, BartForConditionalGeneration
import re

class ConversationAnalyzer:
    def __init__(self, conversation_data):
        self.conversation_data = conversation_data
        self.speaker1_entities = []
        self.speaker2_entities = []
        self.speaker1_counts = Counter()
        self.speaker2_counts = Counter()
        self.summary = ""

    def extract_entities(self):
        # Group conversations by speaker
        speaker1_conversations = self.conversation_data[self.conversation_data["Speaker"] == "SPEAKER 1"]
        speaker2_conversations = self.conversation_data[self.conversation_data["Speaker"] == "SPEAKER 2"]

        # Concatenate conversation text for each speaker
        speaker1_text = " ".join(speaker1_conversations["Text"])
        speaker2_text = " ".join(speaker2_conversations["Text"])

        # Load the pre-trained NER model
        ner_model = "dslim/bert-base-NER"
        ner = pipeline("ner", model=ner_model)

        # Extract person entities using NER model
        self.speaker1_entities = [entity["word"] for entity in ner(speaker1_text) if entity["entity"] == "B-PER"]
        self.speaker2_entities = [entity["word"] for entity in ner(speaker2_text) if entity["entity"] == "B-PER"]

    def merge_entities(self, entities):
        merged_entities = []
        for entity in entities:
            found = False
            for i, merged_entity in enumerate(merged_entities):
                if re.search(r'\b' + re.escape(entity) + r'\b', merged_entity, flags=re.IGNORECASE):
                    merged_entities[i] = re.sub(r'\b' + re.escape(entity) + r'\b', entity, merged_entity, flags=re.IGNORECASE)
                    found = True
                    break
            if not found:
                merged_entities.append(entity)
        return merged_entities

    def count_entities(self):
        # Merge similar person entities
        merged_speaker1_entities = self.merge_entities(self.speaker1_entities)
        merged_speaker2_entities = self.merge_entities(self.speaker2_entities)

        # Count occurrences of person names
        self.speaker1_counts = Counter(merged_speaker1_entities)
        self.speaker2_counts = Counter(merged_speaker2_entities)

    def summarize_conversation(self):
        # Prepare input for summarization
        input_text = " ".join(self.conversation_data["Text"])

        # Load the pre-trained BART model for text summarization
        bart_model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
        bart_tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")

        # Tokenize input text
        input_ids = bart_tokenizer.encode(input_text, truncation=True, max_length=1024, return_tensors="pt")

        # Generate summary
        summary_ids = bart_model.generate(input_ids, num_beams=4, length_penalty=2.0, max_length=150, min_length=40, no_repeat_ngram_size=3)

        # Decode the summary
        self.summary = bart_tokenizer.decode(summary_ids.squeeze(), skip_special_tokens=True)

    def analyze_conversation(self):
        self.extract_entities()
        self.count_entities()
        self.summarize_conversation()

        # Print the count of person names
        print("Speaker 1:")
        if len(self.speaker1_counts) == 0:
            print("No person entities found.")
        else:
            for name, count in self.speaker1_counts.items():
                print(f"{name}: {count} times")

        print("Speaker 2:")
        if len(self.speaker2_counts) == 0:
            print("No person entities found.")
        else:
            for name, count in self.speaker2_counts.items():
                print(f"{name}: {count} times")

        print("\nConversation Summary:")
        print(self.summary)

# Define the conversation data
data = {
    "Start": {
        "0": "0:00:00",
        "1": "0:00:15",
        "2": "0:00:17",
        "3": "0:00:20",
        "4": "0:00:24",
        "5": "0:00:58",
        "6": "0:01:05",
        "7": "0:01:50",
        "8": "0:01:54",
        "9": "0:02:07"
    },
    "End": {
        "0": "0:00:15",
        "1": "0:00:17",
        "2": "0:00:20",
        "3": "0:00:23",
        "4": "0:00:58",
        "5": "0:01:05",
        "6": "0:01:50",
        "7": "0:01:54",
        "8": "0:02:07",
        "9": "0:02:33"
    },
    "Speaker": {
        "0": "SPEAKER 1",
        "1": "SPEAKER 2",
        "2": "SPEAKER 1",
        "3": "SPEAKER 2",
        "4": "SPEAKER 1",
        "5": "SPEAKER 2",
        "6": "SPEAKER 1",
        "7": "SPEAKER 2",
        "8": "SPEAKER 1",
        "9": "SPEAKER 2"
    },
    "Text": {
        "0": " Hi Mark, this is Priya from GTSD. Hi there. Mark, are you working from home or office? ",
        "1": " From home. ",
        "2": " Are you connected to VPN? ",
        "3": " Yes, I am at the moment. ",
        "4": " Okay. The password which I have given, please try within a couple of minutes or you have a grace time of 20 minutes. Okay. If it's not working, could you please contact us back? Or are you using Teams on your phone? Sorry, say that again. Do you use Teams on your phone so that I can message you on Teams? ",
        "5": " I can't log in now because my password has changed. I can't use it on my phone either. ",
        "6": " Okay. Is the password you are trying, is it welcome at 123? Correct? The first letter W would be in capital. Is that right? Yes. I will tell you infinitively. Please correct me if I'm wrong. It's W in capital which is W as in whiskey, E as in nickel, L as in Lima, C as in Charlie, O as in Oscar, M as in Mary, E as in Edward. The at symbol number 123. Is it right? ",
        "7": " Yes, that's correct. ",
        "8": " Okay. I will check your account to see what's going on. It seems like you are entering the wrong password. ",
        "9": " Okay. Let me try again. "
    }
}

# Create a DataFrame from the conversation data
conversation_df = pd.DataFrame(data)

# Create an instance of ConversationAnalyzer
analyzer = ConversationAnalyzer(conversation_df)

# Analyze the conversation
analyzer.analyze_conversation()
