import streamlit as st
import os
from langchain.chat_models import AzureChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.chains import LLMChain
 
 
 
@st.cache_resource
def create_llm():
    llm = AzureChatOpenAI(deployment_name='chat')
    prompt = ChatPromptTemplate(
        messages=[
            SystemMessagePromptTemplate.from_template("You are a nice chatbot having a conversation with a human."),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{question}"),
        ]
    )
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation = LLMChain(llm=llm, prompt=prompt, verbose=True, memory=memory)
    return conversation
 
def main():
    st.title("Code Assitance")
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
 
if __name__ == '__main__':
    main()
