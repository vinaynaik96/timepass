#Langchain libraries
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
from langchain.embeddings import AzureOpenAIEmbeddings

#AZURE Credentials
import os
os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["OPENAI_API_VERSION"] = "2023-05-15"
os.environ["OPENAI_API_BASE"] = "https://azureopenaicoe.openai.azure.com/"
os.environ["OPENAI_API_KEY"] = "ceea3c2ec4814ed89507f4ee06b907a2"

def render_clickable_link(lst):
    i=0
    for dictionary in lst:
        print(f"{i}:")
        for key, value in dictionary.items():
            if key == "BotURL":
                print(f"{key}: [{value}]({value})")
            else:
                print(f"{key}: {value}")
        print(" ")
        i=i+1 
        
embeddings = AzureOpenAIEmbeddings(azure_deployment='text-embedding-ada-002')
llm=AzureOpenAI(deployment_name='langchain')
# load from disk
retriever = Chroma(persist_directory="./Botstabledata_db", embedding_function=embeddings).as_retriever()

compressor = LLMChainExtractor.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=retriever)

prompt = "give me a linux image in AWS"
prompt_engg= prompt+" remember give me unique 3 bot names"

response = compression_retriever.get_relevant_documents(prompt_engg)
response

res=[]
res1=[]

for data in response:
    dct={}
    bot_name=data.metadata['bot_name']
    bot_name=bot_name.replace(" ", "")
    bot_url=f"https://www.Botstore.com/botname={bot_name}"                
    dct["Description"]=data.page_content
    dct["BotName"]=data.metadata['bot_name']
    dct["BotURL"]=bot_url              
    output=f"Description : {data.page_content} \n BotName : {data.metadata['bot_name']}  \n BotURL : {bot_url}"
    res.append(output)
    res1.append(dct)
if len(res) ==0:
    res="No Result Found , Please Give Valid Description"

render_clickable_link(res1)
