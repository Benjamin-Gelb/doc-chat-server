from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
from chromadb import Collection
from langchain.schema import StrOutputParser
import enum
from mongoengine import EnumField, StringField, DateTimeField, IntField, EmbeddedDocument
from datetime import datetime
from operator import itemgetter
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough



class MessageType(enum.Enum):
    AIMessage = 'AIMessage'
    HumanMessage = 'HumanMessage'

class ChatMessage(EmbeddedDocument):
    type = EnumField(enum=MessageType)
    content = StringField(required=True)
    timestamp = DateTimeField(default=datetime.now())

    ## Below are additional fields that are unique to {type: AIMessage}
    @property
    def message_validation(self):
        return self.type == MessageType.AIMessage

    context = StringField(required=message_validation)
    pagenumber = IntField(required=message_validation)
    docname = StringField(required=message_validation)

load_dotenv()

restate_template = """Re-state the following question, given the context of the prior conversation and a summary of the relevant text.
Question:
{question}
Current Conversation:
{history}
Summary
{summary}
Re-stated Question:
"""

def provide_memory_context(memory: ConversationBufferMemory, chat_history: list[ChatMessage]):
    """Updates memory with chat history."""
    chat_memory = memory.chat_memory
    for message in chat_history:
        if message.type == MessageType.HumanMessage:
            chat_memory.add_user_message(message.content)
            continue
        if message.type == MessageType.AIMessage:
            chat_memory.add_ai_message(message.content)
            continue
    return memory    

question_template = """""You are a legal assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, disclose that your knowledge is incomplete but answer to the best of your ability.
Question:
{question}
Legal context:
{context}
"""

def setup(chat_history: list[ChatMessage], summary: str, context: str) -> LLMChain:

    memory = ConversationBufferMemory(memory_key='history')
    memory = provide_memory_context(memory, chat_history)
    restate_prompt =PromptTemplate(template=restate_template, input_variables=['question'], partial_variables={
            'summary': summary,
        })
    # restate_chain = LLMChain(prompt=prompt, memory=memory, llm=ChatOpenAI(temperature=0), output_key='question')

    question_prompt = PromptTemplate(template=question_template, input_variables=['question'], partial_variables={
        'context': context
    })
    llm = ChatOpenAI(temperature=0)
    # question_chain = LLMChain(prompt=prompt, llm=llm)

    return {"question" : RunnablePassthrough.assign(
      history=RunnableLambda(memory.load_memory_variables) | itemgetter("history")
  ) | restate_prompt | llm | StrOutputParser()} | question_prompt | llm | StrOutputParser()










