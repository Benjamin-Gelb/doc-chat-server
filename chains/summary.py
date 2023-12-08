from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain, StuffDocumentsChain
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

# max string length for summary

from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.docstore.document import Document

def stuff(docs: list[Document]):
    template = """Write a concise summary of the following:
    "{text}"
    CONCISE SUMMARY:"""
    prompt = PromptTemplate.from_template(template)

    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k")
    llm_chain = LLMChain(llm=llm, prompt=prompt)

    stuff_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="text")

    return stuff_chain.run(docs)


# from langchain.document_loaders import PyPDFLoader

# loader = PyPDFLoader('chains/p_brief_priv.pdf')
# docs = loader.load()
# print(stuff(docs))

# def setup(**kwargs):
#     docs = kwargs.get('docs', None)
#     if not docs:
#         raise KeyError("Didn't pass 'docs' as keyword argument.")
#     return stuff(docs)
    


# from langchain.chains.combine_documents.stuff import StuffDocumentsChain
# from langchain.chains.llm import LLMChain
# from langchain.prompts import PromptTemplate

# # Define prompt
# prompt_template = """Write a concise summary of the following document:
# "{text}"
# CONCISE SUMMARY:"""
# prompt = PromptTemplate.from_template(prompt_template)

# # Define LLM chain
# llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k")
# llm_chain = LLMChain(llm=llm, prompt=prompt)

# # Define StuffDocumentsChain
# stuff_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="text")

# docs = loader.load()
# print(stuff_chain.run(docs))
