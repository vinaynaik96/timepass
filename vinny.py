from langchain.chat_models import AzureChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
import streamlit as st
 
BASE_URL = "https://azureopenaicoe.openai.azure.com/"
API_KEY = "ceea3c2ec4814ed89507f4ee06b907a2"
DEPLOYMENT_NAME = "chat"
 
model = AzureChatOpenAI(
    openai_api_base=BASE_URL,
    openai_api_version="2023-05-15",
    deployment_name=DEPLOYMENT_NAME,
    openai_api_key=API_KEY,
    openai_api_type="azure",
)


import streamlit as st
import os

# App title
st.set_page_config(page_title="💬 Knowledge Base Chatbot")

st.title('💬 Knowledge Base Chatbot')

# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

# Display or clear chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]


def generate_response(prompt_input,user_input):
    system_message = SystemMessage(content=f"create a technical knowledge article create objective as Heading,procedure and conclusion inside that it should keep these content without much change and at end add created by whom and date {prompt_input}")
    user_message = HumanMessage(content=user_input)
    messages = [system_message, user_message]
    response = model(messages)
    return response.content

custom_css = '''
    <style>
        textarea.stTextArea {
            width: 800px !important;
            height: 400px !important;
        }
    </style>
    '''

st.write(custom_css, unsafe_allow_html=True)

with st.sidebar:
    st.title("Service Now Description")
    user_input = st.text_area("Type your text here:")
    st.subheader("Your Conversation :")
    st.write(user_input)
    
# User-provided prompt
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = generate_response(prompt,user_input)
            st.write(response)
    message = {"role": "assistant", "content": response}
    st.session_state.messages.append(message)
