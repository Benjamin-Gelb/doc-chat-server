from langchain.llms.base import BaseLLM
from langchain.embeddings.base import Embeddings
from langchain.chat_models.base import BaseChatModel
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain, base
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from mongoengine import StringField, EnumField, Document
from enum import Enum



load_dotenv()

class MessageType(Enum):
    AIMessage = 'AIMessage'
    HumanMessage = 'HumanMessage'

class ChatMessage(Document):
    type = EnumField(enum=MessageType)
    content = StringField(required=True)

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

    @classmethod
    def create_chain(self, conversation_history: list[ChatMessage]) -> base.Chain:
        prompt = PromptTemplate.from_template("""You are a legal assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know.
        Previous conversation:
        {chat_history}
        Question:
        {user_input}
        Legal context:
        {context}
        """
        )
        memory = ConversationBufferMemory(memory_key="chat_history", input_key="user_input")
        for message in conversation_history:
            if message.type == 'AIMessage':
                memory.chat_memory.add_ai_message(message.content)
            if message.type == 'HumanMessage':
                memory.chat_memory.add_user_message(message.content)

        return LLMChain(llm=self.chat_model or self.llm, prompt=prompt, memory=memory)
    # def __init__(self,  embeddings_model: Embeddings, llm : BaseLLM = None, chat_model: BaseChatModel = None) -> None:



    #     self.llm = llm
    #     self.chat_model = chat_model
    #     self.embeddings = embeddings_model



# config = Config(OpenAIEmbeddings(), chat_model=ChatOpenAI(temperature=0))