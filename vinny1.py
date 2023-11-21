from langchain.callbacks import StreamlitCallbackHandler
import streamlit as st
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
import os

os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["OPENAI_API_VERSION"] = "2023-05-15"
os.environ["OPENAI_API_BASE"] = "https://azureopenaicoe.openai.azure.com/"
os.environ["OPENAI_API_KEY"] = "ceea3c2ec4814ed89507f4ee06b907a2"

def clear_submit():
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
        
st.set_page_config(page_title="Search The Bot Or Generate Your Own", page_icon="🦜")
st.title("Search The Bot Or Generate Your Own")

embeddings = OpenAIEmbeddings(deployment_name='text-embedding-ada-002')
llm = AzureOpenAI(deployment_name='langchain')

retriever = Chroma(persist_directory="./Botstabledata_db", embedding_function=embeddings).as_retriever()

compressor = LLMChainExtractor.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=retriever)

if "messages" not in st.session_state or st.sidebar.button("Clear conversation history"):
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]
else:
    context = "\n".join([msg["content"] for msg in st.session_state["messages"]])
    llm.add_training_data(context)

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
        response = compression_retriever.get_relevant_documents(prompt_engg)
        res=[]
        res1=[]
        for data in response:
            dct={}
            bot_name = data.metadata['bot_name']
            bot_name = bot_name.replace(" ", "")
            bot_url = f"https://www.Botstore.com/botname={bot_name}"                
            dct["Description"] = data.page_content
            dct["BotName"] = data.metadata['bot_name']
            dct["BotURL"] = bot_url              
            output = f"Description : {data.page_content} \n BotName : {data.metadata['bot_name']}  \n BotURL : {bot_url}"
            res.append(output)
            res1.append(dct)
        if len(res) == 0:
            res = "No Result Found, Please Give Valid Description"
            st.write(res)
        st.session_state.messages.append({"role": "assistant", "content": res})
        render_clickable_link(res1)
