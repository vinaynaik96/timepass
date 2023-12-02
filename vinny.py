import streamlit as st
import tempfile
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import LanguageParser
from langchain.embeddings import AzureOpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chat_models import AzureChatOpenAI
from langchain.memory import ConversationSummaryMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import Language
from langchain.document_loaders import NotebookLoader, PythonLoader

os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["OPENAI_API_VERSION"] = "2023-05-15"
os.environ["OPENAI_API_KEY"] = "22c2ca1a04134562be1ab848aaba7d9c"
os.environ["OPENAI_API_BASE"] = "https://coeaoai.openai.azure.com/"

FILE_LOADER_MAPPING = {
    "ipynb": (NotebookLoader, {}),
    "py": (PythonLoader, {}),
}

def create_vector_database(documents):
    # Function to create vector database (unchanged from your code)
    # ...

def run_code_rag():
    st.title("Code Description")
    
    uploaded_files = st.file_uploader("Upload your documents", type=["py", "ipynb"], accept_multiple_files=True)
    loaded_documents = []

    if uploaded_files:
        with tempfile.TemporaryDirectory() as td:
            for uploaded_file in uploaded_files:
                ext = os.path.splitext(uploaded_file.name)[-1][1:].lower()
                if ext in FILE_LOADER_MAPPING:
                    loader_class, loader_args = FILE_LOADER_MAPPING[ext]
                    file_path = os.path.join(td, uploaded_file.name)
                    with open(file_path, 'wb') as temp_file:
                        temp_file.write(uploaded_file.read())
                    
                    loader = loader_class(file_path, **loader_args)
                    documents = loader.load()
                    loaded_documents.extend(loader.load())
                else:
                    st.warning(f"Unsupported file extension: {ext}")
                        
        persist_directory, embeddings = create_vector_database(loaded_documents)
        db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
        retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": 8})
        
        callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
        llm = AzureChatOpenAI(deployment_name='chat', callback_manager=callback_manager, verbose=True)
        
        memory = ConversationSummaryMemory(llm=llm, memory_key="chat_history", return_messages=True)
        qa = ConversationalRetrievalChain.from_llm(llm, retriever=retriever, memory=memory)
    
        query = st.text_input("Ask a question:")
        if st.button("Get Answer"):
            if query:
                result = qa(query)
                st.write(result['answer'])
                
        # Conversational chat interface
        st.subheader("Conversational Chat Interface")
        user_input = st.text_input("You:")
        if st.button("Send"):
            if user_input:
                response = qa(user_input)
                st.text_area("Chatbot:", value=response['answer'], height=100)
