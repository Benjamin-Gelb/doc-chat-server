from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain, base
from config import LLMConfig
from langchain.vectorstores import Chroma
from chromadb import Collection 
from mongoengine import StringField, EnumField, Document
from utils import create_embeddings


load_dotenv()

class MessageType:
    AIMessage = 'AIMessage'
    HumanMessage = 'HumanMessage'

class ChatMessage(Document):
    type =  EnumField(MessageType, required=True)
    content = StringField(required=True)

def create_chain(message_history: list[ChatMessage]) -> base.Chain:
    prompt =PromptTemplate.from_template("""You are a legal assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know.
    Previous conversation:
    {chat_history}
    Question:
    {user_input}
    Legal context:
    {context}
    """
    )
    memory = ConversationBufferMemory(memory_key="chat_history", input_key="user_input")
    for message in message_history:
        if message.type == 'AIMessage':
            memory.chat_memory.add_ai_message(message.content)
        if message.type == 'HumanMessage':
            memory.chat_memory.add_user_message(message.content)

    return LLMChain(llm=LLMConfig.chat_model or LLMConfig.llm, prompt=prompt, memory=memory)



    