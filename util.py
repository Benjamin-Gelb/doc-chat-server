import os, string, random, hashlib
from werkzeug.datastructures import FileStorage
from typing import Any


from PyPDF2 import PdfReader, PageObject
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import base, OpenAIEmbeddings
from langchain.chains.base import Chain

from chromadb import api, Collection, EmbeddingFunction




# def pdf_summary(file: FileStorage):
#     loader = PyPDFLoader(file)
#     docs = loader.load()
#     summary = stuff(docs)


def to_dict(self, include : list[str] = None, exclude : list[str] = None):
    visitor_dict = self.to_mongo().to_dict()
    if include:
        copy = dict()
        for key in include:
            copy[key] = visitor_dict.get(key, None)
        return copy
    if exclude:
        try:
            for key in exclude:
                del visitor_dict[key]
        except KeyError as e:
            print(e)
            pass
    return visitor_dict

def generate_unique_char_string(length=64):
    """ Generate a random string of fixed length """
    letters = string.ascii_letters + string.digits
    return ''.join(random.choice(letters) for _ in range(length))

# import jwt
# from chromadb import api

# class CustomEmbeddingFunction(EmbeddingFunction):

#     embedding_model = OpenAIEmbeddings()
#     def __call__(self, input: Any) -> Embeddings:
#         return self.embedding_model.embed_documents(input)

# def create_collection(client: api.ClientAPI) -> Collection:
#     name = generate_unique_char_string(32)
#     return client.create_collection(name=name, embedding_function=CustomEmbeddingFunction())

# def retrieve_collection(name: str, client: api.ClientAPI):
#     return client.get_collection(name=name, embedding_function=CustomEmbeddingFunction())

from typing import Callable

def extract_text(reader: PdfReader) -> (str, Callable[[int], int]):
    full_text = ""
    char_len = 0
    page_map = []

    for page in reader.pages:
        page_text = page.extract_text()
        char_len += len(page_text)
        full_text += page_text
        page_map.append(char_len)

    def get_page_number(char_progress: int):
        for i, char_len in enumerate(page_map):
            if char_progress < char_len:
                return i+1
        return -1
    
    return (full_text, get_page_number)


def reader_to_summary(reader : PdfReader, summary_chain: Chain):
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return summary_chain.invoke({
        'document': text
    })

import jwt
from chromadb import api


class CustomEmbeddingFunction(EmbeddingFunction):

    embedding_model = OpenAIEmbeddings()
    def __call__(self, input: Any) -> base.Embeddings:
        return self.embedding_model.embed_documents(input)

def create_collection(client: api.ClientAPI) -> Collection:
    name = generate_unique_char_string(32)
    return client.create_collection(name=name, embedding_function=CustomEmbeddingFunction())

def retrieve_collection(name: str, client: api.ClientAPI):
    return client.get_collection(name=name, embedding_function=CustomEmbeddingFunction())

def gen_token(key: str, field: str):
    payload = {
        field: key
    }
    secret = os.getenv('JWT_SECRET')
    algorithm=os.getenv('JWT_ALGORITHM')
    encoded_payload = jwt.encode(payload, secret, algorithm=algorithm)
    return encoded_payload

def retrieve_token(encoded_payload: str, field: str):
    secret = os.getenv('JWT_SECRET')
    algorithm=os.getenv('JWT_ALGORITHM')
    payload = jwt.decode(encoded_payload, secret, algorithms=[algorithm])
    return payload[field]
