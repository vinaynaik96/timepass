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
llm = AzureChatOpenAI(deployment_name='chat', callback_manager=callback_manager, verbose=True)
        
def create_chain(llm, prompt, CONDENSE_QUESTION_PROMPT, db):
    memory = ConversationTokenBufferMemory(llm=llm, memory_key="chat_history", return_messages=True, input_key='question', output_key='answer')
    chain = ConversationalRetrievalChain.from_llm(llm=llm, chain_type="stuff", retriever=db.as_retriever(search_kwargs={"k": 3}), return_source_documents=True, max_tokens_limit=256, combine_docs_chain_kwargs={"prompt": prompt}, condense_question_prompt=CONDENSE_QUESTION_PROMPT, memory=memory)
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

# Load from disk
db = Chroma(persist_directory="Botstabledata_db", embedding_function=embeddings)
qa = create_chain(llm=llm, prompt=prompt, CONDENSE_QUESTION_PROMPT=CONDENSE_QUESTION_PROMPT, db=db)

# Check session_state for existing conversation history or create a new one
if "messages" not in st.session_state or st.sidebar.button("Clear conversation history"):
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]
    st.session_state["context"] = ""  # Initialize context

st.title('🦜🔗 Q&A ChatBot')

while True:
    with st.form("user_input"):
        user_query = st.text_input("You:", key="user_input")
        submit_button = st.form_submit_button(label='Send')

    if submit_button:
        query = user_query + " remember give me unique 3 bot names"
        st.session_state["messages"].append({"role": "user", "content": user_query})

        context = st.session_state["context"]
        if context:
            query = f"{context} {query}"  # Include previous context

        with st.spinner("Thinking..."):
            response = qa({"question": query})

            st.session_state["messages"].append({"role": "assistant", "content": response})
            st.session_state["context"] = response.get("context", "")

        for msg in st.session_state["messages"]:
            if msg["role"] == "user":
                st.text_input("You:", value=msg["content"], disabled=True)
            else:
                st.text_input("Assistant:", value=msg["content"], disabled=True)
