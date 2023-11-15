from langchain.agents import AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chat_models import ChatOpenAI
import streamlit as st
import pandas as pd
from langchain.chains import RetrievalQA
from langchain.document_loaders import CSVLoader, DataFrameLoader
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
from langchain.chains import LLMChain, ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import os

os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["OPENAI_API_VERSION"] = "2023-05-15"
os.environ["OPENAI_API_BASE"] = "https://azureopenaicoe.openai.azure.com/"
os.environ["OPENAI_API_KEY"] = "ceea3c2ec4814ed89507f4ee06b907a2"

llm = AzureOpenAI(deployment_name='langchain')


def clear_submit():
    st.session_state["submit"] = False


def render_clickable_link(lst):
    i = 0
    for dictionary in lst:
        st.write(f"{i}:")
        for key, value in dictionary.items():
            if key == "BotURL":
                st.markdown(f"{key}: [{value}]({value})")
            else:
                st.write(f"{key}: {value}")
        st.write(" ")
        i = i + 1


memory = ConversationBufferMemory()
chat_llm = AzureChatOpenAI(deployment_name='chat')
# Prompt
prompt = ChatPromptTemplate(
    messages=[
        SystemMessagePromptTemplate.from_template(
            """You are a brilliant code assistant. Give only valid code.
            If you don't know the answer, just say that you don't know, don't try to make up an answer. 
            
            {question}
            """
        ),
        # The `variable_name` here is what must align with memory
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{question}")
    ]
)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

conversation = LLMChain(
    llm=chat_llm,
    prompt=prompt,
    verbose=True,
    memory=memory
)

st.set_page_config(page_title="Search The Bot Or Generate Your Own", page_icon="🦜")
st.title("Search The Bot Or Generate Your Own")

embeddings = OpenAIEmbeddings(deployment_name='text-embedding-ada-002')
llm = AzureOpenAI(deployment_name='langchain')

retriever = Chroma(persist_directory="./Botstabledata_db", embedding_function=embeddings).as_retriever()
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
qa = ConversationalRetrievalChain.from_llm(llm, retriever, memory=memory)

if "messages" not in st.session_state or st.sidebar.button("Clear conversation history"):
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if "visibility" not in st.session_state:
    st.session_state.visibility = "visible"
    st.session_state.disabled = False


if prompt := st.chat_input(placeholder="Please Type Your Query"):
    prompt_engg = prompt + " remember give me unique 3 bot names"
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        response = qa({"question": prompt_engg})
        res = []
        res1 = []
        print("===============")
        print(type(response["answer"]))
        print("===============")
        for k, v in response.items():
            print(k, v)
            print("===============")
        print(response["answer"])
        dct = {}
        res.append(response["answer"])
        res1.append(dct)
        if len(res) == 0:
            res = "No Result Found , Please Give Valid Description"
            st.write(res)
        st.session_state.messages.append({"role": "assistant", "content": res})
        st.write(res)
