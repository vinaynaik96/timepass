from langchain.chat_models import AzureChatOpenAI
from langchain.callbacks import StreamlitCallbackHandler
import streamlit as st
from langchain.callbacks.manager import CallbackManager
import os
from langchain.vectorstores import Chroma
from langchain.embeddings import AzureOpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.memory import ConversationTokenBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["OPENAI_API_VERSION"] = "2023-05-15"
os.environ["OPENAI_API_KEY"] = "22c2ca1a04134562be1ab848aaba7d9c"
os.environ["OPENAI_API_BASE"] = "https://coeaoai.openai.azure.com/"

def clear_submit():
    st.session_state["submit"] = False
        
embeddings = AzureOpenAIEmbeddings(azure_deployment='text-embedding-ada-002')

callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
llm=AzureChatOpenAI(deployment_name='chat', callback_manager=callback_manager, verbose=True)

def create_chain(llm, prompt, CONDENSE_QUESTION_PROMPT, db):
    memory = ConversationTokenBufferMemory(llm=llm, memory_key="chat_history", return_messages=True, input_key='question',      
                                           output_key='answer')
    chain = ConversationalRetrievalChain.from_llm(llm=llm,chain_type="stuff",retriever=db.as_retriever(search_kwargs={"k": 3}),
                                                  return_source_documents=True, max_tokens_limit=256,
                                                  combine_docs_chain_kwargs={"prompt": prompt},
                                                  condense_question_prompt=CONDENSE_QUESTION_PROMPT, memory=memory,)
    return chain

def set_custom_prompt_condense():
    _template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.
    Chat History:
    {chat_history}
    Follow Up Input: {question}
    Standalone question:"""
    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)
    return CONDENSE_QUESTION_PROMPT

def set_custom_prompt():
    prompt_template = """Use the following pieces of information to answer the user's question.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Context: {context}
    Question: {question}
    Only return the helpful answer below and nothing else.
    Helpful answer:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return prompt

prompt = set_custom_prompt()
CONDENSE_QUESTION_PROMPT = set_custom_prompt_condense()

# load from disk
db = Chroma(persist_directory="Botstabledata_db", embedding_function=embeddings)
qa = create_chain(llm=llm, prompt=prompt,CONDENSE_QUESTION_PROMPT=CONDENSE_QUESTION_PROMPT, db=db)


def run_qa_bot():
    st.title("Q&A ChatBot")
    
    if "messages" not in st.session_state or st.sidebar.button("Clear conversation history"):
        st.session_state["messages"] = [{"role": "assistant", "content":
    "How can I help you?"}]
        
    if prompt := st.chat_input(placeholder="Please Type Your Query"):
        query= prompt+" remember give me unique 3 bot names"
        st.chat_message("user").write(prompt)
        
        with st.chat_message("assistant"):
                response = bot_response = qa({"question": query})
                count = 1
                for data in response["source_documents"]:
                    dct={}
                    bot_name=data.metadata['bot_name']
                    bot_name=bot_name.replace(" ", "")
                    bot_url=f"https://www.Botstore.com/botname={bot_name}"                
                    dct["Description"]=data.page_content
                    dct["BotName"]=data.metadata['bot_name']
                    dct["BotURL"]=bot_url
                    output=f"{count}) BotName : {data.metadata['bot_name']} \n\n Description : {data.page_content} \n\n BotURL : {bot_url}\n\n\n"
                    st.write(output)
                    count += 1
                if len(bot_response["source_documents"]) == 0:
                    st.write("No Result Found , Please Give Valid Description")
