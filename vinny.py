from langchain.agents import AgentType
#from langchain.agents import create_pandas_dataframe_agent
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chat_models import ChatOpenAI
import streamlit as st
import pandas as pd
from langchain.chains import RetrievalQA
from langchain.document_loaders import CSVLoader ,DataFrameLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import AzureOpenAI
from langchain.vectorstores import Chroma
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import LLMChain
from langchain.chat_models import AzureChatOpenAI

from langchain.memory import ConversationBufferMemory
import os
import openai
os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["OPENAI_API_VERSION"] = "2023-05-15"
os.environ["OPENAI_API_BASE"] = "https://azureopenaicoe.openai.azure.com/"
os.environ["OPENAI_API_KEY"] = "ceea3c2ec4814ed89507f4ee06b907a2"
llm=AzureOpenAI(deployment_name='langchain')

def clear_submit():
    """
    Clear the Submit Button State
    Returns:

    """
    st.session_state["submit"] = False

    
def render_clickable_link(lst):
    i=0
    for dictionary in lst:
        st.write(f"{i}:")
        for key, value in dictionary.items():
            if key == "BotURL":
                st.markdown(f"{key}: [{value}]({value})")
            else:
                st.write(f"{key}: {value}")
        st.write(" ")
        i=i+1


memory = ConversationBufferMemory()
chat_llm = AzureChatOpenAI(deployment_name='chat')
# Prompt 
prompt = ChatPromptTemplate(
    messages=[
        SystemMessagePromptTemplate.from_template(
            """You are a briliant code assistant give only valid code.
            If you don't know the answer, just say that you don't know, don't try to make up an answer. 
            
            {question}
            """
        ),
        # The `variable_name` here is what must align with memory
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{question}")
    ]
)
memory = ConversationBufferMemory(sequence_length=5)  # Adjust the sequence_length as needed

conversation = LLMChain(
    llm=chat_llm,
    prompt=prompt,
    verbose=True,
    memory=memory
)

st.set_page_config(page_title="Search The Bot Or Generate Your Own", page_icon="🦜")
st.title("Search The Bot Or Generate Your Own")


embeddings = OpenAIEmbeddings(deployment_name='text-embedding-ada-002')
llm=AzureOpenAI(deployment_name='langchain')
# load from disk
retriever = Chroma(persist_directory="./Botstabledata_db", embedding_function=embeddings).as_retriever()

compressor = LLMChainExtractor.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=retriever)

if "messages" not in st.session_state or st.sidebar.button("Clear conversation history"):
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if "visibility" not in st.session_state:
    st.session_state.visibility = "visible"
    st.session_state.disabled = False
col1, col2 = st.columns(2)
with col2:
    option = st.selectbox(
        "How would you like to be contacted?",
        ("Get BotName", "Code Assistant"),
        label_visibility=st.session_state.visibility,
        disabled=st.session_state.disabled,
    )

if prompt := st.chat_input(placeholder="Please Type Your Query"):
    prompt_engg = prompt + " remember give me unique 3 bot names"
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    
    with st.chat_message("assistant"):
        if option == "Get BotName":
            # Your code for bot name retrieval
            
        elif option == "Code Assistant":
            # Get the current conversation history from the memory
            conversation_history = memory.get_conversation()
            
            # Construct a message with the current user input
            user_message = {"role": "user", "content": prompt}
            
            # Append the user message to the conversation history
            conversation_history.append(user_message)
            
            # Use the conversation with the updated history
            chat_result = conversation({"question": prompt, "chat_history": conversation_history})
            
            # Append the assistant's response to the conversation history
            assistant_message = {"role": "assistant", "content": chat_result['text']}
            conversation_history.append(assistant_message)
            
            # Store the updated conversation history in the memory
            memory.set_conversation(conversation_history)
            
            st.session_state.messages.append(assistant_message)
            
            # To see the full conversation history in the Streamlit app
            full_conversation = memory.get_conversation()
            st.write(full_conversation)

    prompt_engg = prompt + " remember give me unique 3 bot names"
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    
    with st.chat_message("assistant"):
        if option == "Get BotName":
            # Your code for bot name retrieval
            
        elif option == "Code Assistant":
            # Get the current conversation history from the memory
            conversation_history = memory.get_conversation()
            
            # Construct a message with the current user input
            user_message = {"role": "user", "content": prompt}
            
            # Append the user message to the conversation history
            conversation_history.append(user_message)
            
            # Use the conversation with the updated history
            chat_result = conversation({"question": prompt, "chat_history": conversation_history})
            
            # Append the assistant's response to the conversation history
            assistant_message = {"role": "assistant", "content": chat_result['text']}
            conversation_history.append(assistant_message)
            
            # Store the updated conversation history in the memory
            memory.set_conversation(conversation_history)
            
            st.session_state.messages.append(assistant_message)
            
            # To see the full conversation history in the Streamlit app
            full_conversation = memory.get_conversation()
            st.write(full_conversation)

    prompt_engg = prompt + " remember give me unique 3 bot names"
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    
    with st.chat_message("assistant"):
        if option == "Get BotName":
            # Your code for bot name retrieval
            
        elif option == "Code Assistant":
            # Get the current conversation history from the memory
            conversation_history = memory.get_conversation()
            
            # Construct a message with the current user input
            user_message = {"role": "user", "content": prompt}
            
            # Append the user message to the conversation history
            conversation_history.append(user_message)
            
            # Use the conversation with the updated history
            chat_result = conversation({"question": prompt, "chat_history": conversation_history})
            
            # Append the assistant's response to the conversation history
            assistant_message = {"role": "assistant", "content": chat_result['text']}
            conversation_history.append(assistant_message)
            
            # Store the updated conversation history in the memory
            memory.set_conversation(conversation_history)
            
            st.session_state.messages.append(assistant_message)
            
            # To see the full conversation history in the Streamlit app
            full_conversation = memory.get_conversation()
            st.write(full_conversation)

    prompt_engg = prompt + " remember give me unique 3 bot names"
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    
    with st.chat_message("assistant"):
         if option == "Get BotName":

            # Your code for bot name retrieval
            
            if option == "Code Assistant":
            # Get the current conversation history from the memory
                conversation_history = memory.get_conversation()
            
            # Construct a message with the current user input
                user_message = {"role": "user", "content": prompt}
            
            # Append the user message to the conversation history
                conversation_history.append(user_message)
            
            # Use the conversation with the updated history
                chat_result = conversation({"question": prompt, "chat_history": conversation_history})
            
            # Append the assistant's response to the conversation history
                assistant_message = {"role": "assistant", "content": chat_result['text']}
                conversation_history.append(assistant_message)
            
            # Store the updated conversation history in the memory
                memory.set_conversation(conversation_history)
            
                st.session_state.messages.append(assistant_message)
            
            # To see the full conversation history in the Streamlit app
                full_conversation = memory.get_conversation()
                st.write(full_conversation)

