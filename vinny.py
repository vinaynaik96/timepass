import streamlit as st
from langchain.llms import AzureOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import LLMChain
from langchain.chat_models import AzureChatOpenAI
# LLM
from langchain.memory import ConversationBufferMemory
 
@st.cache_resource
def create_llm():
    llm = AzureChatOpenAI(deployment_name='chat')
    prompt = ChatPromptTemplate(
        messages=[
            SystemMessagePromptTemplate.from_template(
                  """You are a briliant code assistant give only valid code.
                  Please give code with import library and no output required.
                  If you don't know the answer, just say that you don't know, don't try to make up an answer."""),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{question}"),
        ]
    )
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation = LLMChain(llm=llm, prompt=prompt, verbose=True, memory=memory)
    return conversation
 
def run_code_bot():
    st.title("Code Assistance")
    conversation = create_llm()
 
    if "messages" not in st.session_state:
        st.session_state.messages = []
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
 
    if prompt := st.chat_input("What is up?"):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        response = conversation({"question": prompt})
        with st.chat_message("assistant"):
            st.markdown(response["text"])
        st.session_state.messages.append({"role": "assistant", "content": response["text"]})
        st.download_button('Download the code', response["text"])
