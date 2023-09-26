from langchain.chat_models import AzureChatOpenAI
from langchain.schema import HumanMessage

BASE_URL = "https://azureopenaicoe.openai.azure.com/"
API_KEY = "ceea3c2ec4814ed89507f4ee06b907a2"
DEPLOYMENT_NAME = "chat"
model = AzureChatOpenAI(
    openai_api_base=BASE_URL,
    openai_api_version="2023-05-15",
    deployment_name=DEPLOYMENT_NAME,
    openai_api_key=API_KEY,
    openai_api_type="azure",
)

model(
    [
        HumanMessage(
            content="create a python code which sum two argument"
        )
    ]
)
