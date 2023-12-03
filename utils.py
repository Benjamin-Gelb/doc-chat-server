import random, string
from uuid import UUID
from mongoengine import Document
from typing import Dict, Any, List
from chromadb import ClientAPI, Collection
from config import LLMConfig
import PyPDF2
from werkzeug.datastructures import FileStorage
from langchain.text_splitter import CharacterTextSplitter
import hashlib
from langchain.chains import base

def random_string(n: int):
    return ''.join([random.choice(string.ascii_letters + string.digits) for _ in range(n)])

def doc_to_dict(document: Document, include:List[str] = [], exclude:List[str] = []):
    document_data = document.to_mongo().to_dict()

    if len(include) != 0:
        document_copy = {}
        for key in include:
            document_copy[key] = document_data[key]
        return document_copy

    for key in exclude:
        del document_data[key]

    return document_data

def create_collection(chroma_client: ClientAPI) -> UUID:
    collection_name = random_string(24)
    collection = chroma_client.create_collection(name=collection_name)
    return (collection.id, collection_name)

def pdf_file_to_text(file : FileStorage):
    reader = PyPDF2.PdfReader(file.stream)
    text = ''
    for page, i in enumerate(reader.pages):
        
        text += page.extract_text()
    return text

def create_embeddings(full_text: str) -> (List[str],List[str], List[List[float]]):
    text_splitter = CharacterTextSplitter(        
        separator = "\n",
        chunk_size = 1000,
        chunk_overlap  = 200, #striding over the text
        length_function = len,
    )
    documents = text_splitter.split_text(full_text)

    ids  = []
    for doc in documents:
        hash_object = hashlib.sha256()
        hash_object.update(doc.encode())
        ids.append(hash_object.hexdigest())

    return (ids, documents, LLMConfig.embeddings_model.embed_documents(documents))
    
def talk_to_doc(docs: Collection, user_message: str, chain: base.Chain):
    context = docs.query(query_embeddings=create_embeddings(user_message)[2], n_results=2)
    return chain.run(user_input=user_message, context=context)
