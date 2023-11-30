import streamlit as st
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

# Set up Langchain
loader = GenericLoader.from_filesystem(
    "./test_repo/langchain/libs/langchain",
    glob="*",
    suffixes=[".py", ".js"],
    parser=LanguageParser(),
)
documents = loader.load()

python_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON, chunk_size=2000, chunk_overlap=200
)
texts = python_splitter.split_documents(documents)

embeddings = AzureOpenAIEmbeddings(azure_deployment='text-embedding-ada-002')

db = Chroma.from_documents(texts, embedding=embeddings)
retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": 8})

callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
llm = AzureChatOpenAI(deployment_name='chat', callback_manager=callback_manager, verbose=True)

memory = ConversationSummaryMemory(llm=llm, memory_key="chat_history", return_messages=True)
qa = ConversationalRetrievalChain.from_llm(llm, retriever=retriever, memory=memory)


# Streamlit app
def main():
    st.title("Code Description & QA App")

    uploaded_file = st.file_uploader("Upload a Python file (.py)", type="py")

    if uploaded_file is not None:
        file_contents = uploaded_file.read().decode("utf-8")  # Get text content of uploaded file

        # Process the uploaded code file
        question = "What is this code about?"  # Define a default question for description generation
        result = qa(file_contents, user_question=question)

        # Display description and QA results
        st.subheader("Description:")
        st.write(result['description'])

        st.subheader("QA Results:")
        for qa_result in result['qa_results']:
            st.write(f"Question: {qa_result['question']}")
            st.write(f"Answer: {qa_result['answer']}")
            st.write("-----")


if __name__ == "__main__":
    main()
