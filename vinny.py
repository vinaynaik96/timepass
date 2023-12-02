import streamlit as st
import tempfile
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

import os

os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["OPENAI_API_VERSION"] = "2023-05-15"
os.environ["OPENAI_API_KEY"] = "22c2ca1a04134562be1ab848aaba7d9c"
os.environ["OPENAI_API_BASE"] = "https://coeaoai.openai.azure.com/"

FILE_LOADER_MAPPING = {
    "ipynb": (NotebookLoader, {}),
    "py": (PythonLoader, {}),
}

def create_vector_database(documents):
    python_splitter = RecursiveCharacterTextSplitter.from_language(language=Language.PYTHON,
                                                               chunk_size=2000,
                                                               chunk_overlap=200)
    texts = python_splitter.split_documents(documents)
    embeddings = AzureOpenAIEmbeddings(azure_deployment='text-embedding-ada-002')
    persist_directory = 'code_db'
    # Create and persist a Chroma vector database from the chunked documents
    db = Chroma.from_documents(
        documents=texts,
        embedding=embeddings,
        persist_directory=persist_directory
        # persist_directory=DB_DIR,
    )
    db.persist()

    return persist_directory, embeddings

def run_code_rag():
    st.title("Code Description")
     # Upload files
    uploaded_files = st.file_uploader("Upload your documents", type=[ "py", "ipynb"], accept_multiple_files=True)
    loaded_documents = []

    if uploaded_files:
            # Create a temporary directory
        with tempfile.TemporaryDirectory() as td:
                # Move the uploaded files to the temporary directory and process them
                for uploaded_file in uploaded_files:
                    st.write(f"Uploaded: {uploaded_file.name}")
                    ext = os.path.splitext(uploaded_file.name)[-1][1:].lower()
                    st.write(f"Uploaded: {ext}")
    
                    # Check if the extension is in FILE_LOADER_MAPPING
                    if ext in FILE_LOADER_MAPPING:
                        loader_class, loader_args = FILE_LOADER_MAPPING[ext]
                        # st.write(f"loader_class: {loader_class}")
    
                        # Save the uploaded file to the temporary directory
                        file_path = os.path.join(td, uploaded_file.name)
                        with open(file_path, 'wb') as temp_file:
                            temp_file.write(uploaded_file.read())
    
                        # Use Langchain loader to process the file
                        loader = loader_class(file_path, **loader_args)
                        documents = loader.load()
                        loaded_documents.extend(loader.load())
                    else:
                        st.warning(f"Unsupported file extension: {ext}")
                        
        persist_directory, embeddings = create_vector_database(loaded_documents)
        db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
        
        retriever = db.as_retriever( search_type="mmr", search_kwargs={"k": 8})
        
        callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
        llm=AzureChatOpenAI(deployment_name='chat', callback_manager=callback_manager, verbose=True)
        
        memory = ConversationSummaryMemory(llm=llm,memory_key="chat_history",return_messages=True)
        qa = ConversationalRetrievalChain.from_llm(llm, retriever=retriever, memory=memory)
    
        query = st.text_input("Ask a question:")
        if st.button("Get Answer"):
            if query:
                result = qa(query)
                st.write(result['answer'])
