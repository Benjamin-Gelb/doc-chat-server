from langchain.llms.base import BaseLLM
from langchain.embeddings.base import Embeddings
from langchain.chat_models.base import BaseChatModel
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI

load_dotenv()

class LLMConfig:
    llm : BaseLLM = None
    embeddings_model : Embeddings = OpenAIEmbeddings()
    chat_model : BaseChatModel= ChatOpenAI(temperature=0)

    # Runtime Checks
    if llm:
        assert(issubclass(llm.__class__, BaseLLM))

    if chat_model:
        assert(issubclass(chat_model.__class__, BaseChatModel))

    assert(chat_model or llm)
    assert(issubclass(embeddings_model.__class__, Embeddings))
    # def __init__(self,  embeddings_model: Embeddings, llm : BaseLLM = None, chat_model: BaseChatModel = None) -> None:



    #     self.llm = llm
    #     self.chat_model = chat_model
    #     self.embeddings = embeddings_model



# config = Config(OpenAIEmbeddings(), chat_model=ChatOpenAI(temperature=0))