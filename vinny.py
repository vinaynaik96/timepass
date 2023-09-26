import openai
import streamlit as st

# Load your Azure OpenAI API key
openai.api_key = 'your-api-key'

def generate_code(prompt):
    response = openai.Completion.create(
        engine="davinci-codex",  # Use the codex model
        prompt=prompt,
        max_tokens=150,
        temperature=0.5
    )
    code = response['choices'][0]['text'].strip()
    return code

st.title('NLP to Python Code Generator')
user_input = st.text_area("Enter your natural language prompt:", "e.g., Write a function to calculate the factorial of a number.")

if st.button('Generate Code'):
    with st.spinner('Generating Python Code...'):
        generated_code = generate_code(user_input)
        st.code(generated_code, language='python')
