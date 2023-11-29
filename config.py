from uuid import UUID
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
from typing import Any, Dict, Generator, Optional, List
from chromadb import ClientAPI, Collection

from langchain.callbacks.base import BaseCallbackManager, BaseCallbackHandler

load_dotenv()

class MyHandler(BaseCallbackHandler):
    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], *, run_id: UUID, parent_run_id: UUID | None = None, tags: List[str] | None = None, metadata: Dict[str, Any] | None = None, **kwargs: Any) -> Any:
        print(serialized, prompts)
class MessageType(Enum):
    AIMessage = 'AIMessage'
    HumanMessage = 'HumanMessage'

class ChatMessage(Document):
    type = EnumField(enum=MessageType)
    content = StringField(required=True)

class LLMConfig:
    llm : BaseLLM = None
    embeddings_model : Embeddings = OpenAIEmbeddings()
    chat_model : BaseChatModel= ChatOpenAI(
        temperature=0,
        callback_manager=BaseCallbackManager([MyHandler()])
        #    model_name='gpt-4',
)

    # Runtime Checks
    if llm:
        assert(issubclass(llm.__class__, BaseLLM))

    if chat_model:
        assert(issubclass(chat_model.__class__, BaseChatModel))

    assert(chat_model or llm)
    assert(issubclass(embeddings_model.__class__, Embeddings))

    @classmethod
    def create_chain(self, conversation_history: list[ChatMessage]) -> base.Chain:
        prompt = PromptTemplate.from_template("""You are a legal assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, disclose that your knowledge is incomplete but try to answer to the best of your ability.
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

def data_generator():
    while True:
        received_data = yield
        print(received_data)
        yield f"Received: {received_data}"

def send_data_to_generator(generator: Generator[str | None, Any, None], data):
    try:
        print(data)
        generated_data = generator.send(data)
    except StopIteration:
        print("Generator has completed.")
    
class MyStreamingHandler(BaseCallbackHandler):
    def __init__(self, gen: Generator) -> None:
        super().__init__()
        self.gen : Generator = gen
    def on_llm_new_token(self, token: str, **kwargs):
        print(f"Received new token: {token}")
        send_data_to_generator(generator=self.gen,data= token)
    # async def on_llm_end(
    #     self) -> None:
    #     """Run when LLM ends running."""
    #     self.gen.close()

def stream_response(docs: Collection, embeddings,  user_message: str):
    gen = data_generator()
    next(gen)

    llm = ChatOpenAI(temperature=0.2, model_name='gpt-4', streaming=True, callbacks=[MyStreamingHandler(gen)])
    prompt = PromptTemplate.from_template("""You are a legal assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, disclose that your knowledge is incomplete but try to answer to the best of your ability.
        Question:
        {user_input}
        Legal context:
        {context}
        """
    )
    chain = LLMChain(prompt=prompt, llm=llm )
    

    
    context = docs.query(query_embeddings=embeddings, n_results=2)
    chain.run(user_input=user_message, context=context)
    return gen
        

    # def __init__(self,  embeddings_model: Embeddings, llm : BaseLLM = None, chat_model: BaseChatModel = None) -> None:



    #     self.llm = llm
    #     self.chat_model = chat_model
    #     self.embeddings = embeddings_model



# config = Config(OpenAIEmbeddings(), chat_model=ChatOpenAI(temperature=0))