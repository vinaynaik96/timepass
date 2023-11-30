from langchain.text_splitter import Language
from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import LanguageParser
from langchain.chat_models import AzureChatOpenAI
from langchain.callbacks.manager import CallbackManager
from langchain.memory import ConversationSummaryMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from langchain.embeddings import AzureOpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

import os

os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["OPENAI_API_VERSION"] = "2023-05-15"
os.environ["OPENAI_API_KEY"] = "22c2ca1a04134562be1ab848aaba7d9c"
os.environ["OPENAI_API_BASE"] = "https://coeaoai.openai.azure.com/"

loader = GenericLoader.from_filesystem(
    "./test_repo/langchain/libs/langchain",
    glob="*",
    suffixes=[".py", ".js"],
    parser=LanguageParser(),
)
documents = loader.load()

python_splitter = RecursiveCharacterTextSplitter.from_language(language=Language.PYTHON,
                                                               chunk_size=2000,
                                                               chunk_overlap=200)
texts = python_splitter.split_documents(documents)

embeddings = AzureOpenAIEmbeddings(azure_deployment='text-embedding-ada-002')

db = Chroma.from_documents(texts, embedding = embeddings)
retriever = db.as_retriever( search_type="mmr", search_kwargs={"k": 8})

callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
llm=AzureChatOpenAI(deployment_name='chat', callback_manager=callback_manager, verbose=True)

memory = ConversationSummaryMemory(llm=llm,memory_key="chat_history",return_messages=True)
qa = ConversationalRetrievalChain.from_llm(llm, retriever=retriever, memory=memory)

question = "What is code all about?"
result = qa(question)
result['answer']
