from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
from main import ChatMessage
from main import MessageType

load_dotenv()

template = """Re-state the following question, given the context of the prior conversation and a summary of the relevant text.
Question:
{question}
Current Conversation:
{memory}
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
        

def setup(**kwargs) -> LLMChain:
    chat_history: list[ChatMessage] = kwargs.get('chat_history')
    memory = ConversationBufferMemory(memory_key='memory')
    memory = provide_memory_context(memory, chat_history)
    def restate_chain(question: str, summary: str):
        chain = LLMChain(prompt=PromptTemplate.from_template(template), memory=memory, llm=ChatOpenAI(temperature=0))
        output = chain.predict(
            question=question,
            summary=summary
        )
        return output
    return restate_chain










