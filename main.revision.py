from mongoengine import (
    Document, StringField, IntField, DateTimeField, ListField,
    ReferenceField, EnumField, EmbeddedDocument,
    EmbeddedDocumentListField,
)
from chromadb import Collection
from werkzeug.datastructures import FileStorage
from PyPDF2 import PdfReader, PageObject
from langchain.chains.base import Chain 
import hashlib, enum, random, string, os
from langchain.text_splitter import CharacterTextSplitter

def generate_unique_char_string(length=64):
    """ Generate a random string of fixed length """
    letters = string.ascii_letters + string.digits
    return ''.join(random.choice(letters) for i in range(length))

class Doc(Document):
    ### Document Data
    filename = StringField()
    mimetype = StringField()
    pagelength = IntField(min_value=0)
    uploadDate = DateTimeField()
    summary = StringField()


    # Retrival Data
    contenthash = StringField()
    vs_ref = StringField()

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
        for i, char_len in page_map:
            if char_progress < char_len:
                return i+1
        return -1
    
    return (full_text, get_page_number)




def pdf_to_collection(file: FileStorage, collection: Collection) -> (str, int, str):
    """Inserts File content into chromadb collection and returns relevant metadata generated from insertion"""

    ## In its current iteration this is not fault tolerant for pdf files where the page size is massive.

    vs_ref = generate_unique_char_string(64) ## Unique value that ties the Database to Chromadb

    reader = PdfReader(file.stream)
    ids = []
    documents = []
    metadatas = []
    document_hash = hashlib.sha256()

    text_splitter = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separator='\n'
    )

    full_text, get_page_number = extract_text(reader)

    char_prog = 0
    for doc in text_splitter.split_text(full_text):
        char_prog += len(doc)
        hash = hashlib.sha256()
        hash.update(doc)

        documents.append(doc)
        ids.append(hash.hexdigest())
        metadatas.append({
            'page_number': get_page_number(char_prog),
            'char_length': len(doc),
            # 'filename': file.filename,
            # 'mimetype': file.mimetype,
            'vs_ref': vs_ref
        })
        document_hash.update(doc)
    
    collection.add(ids, None, metadatas, documents)
    return (vs_ref, reader.numPages, document_hash.digest())

def reader_to_summary(reader : PdfReader, summary_chain: Chain):
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return summary_chain.invoke({
        'document': text
    })

from datetime import datetime

def upload_document(file: FileStorage, collection: Collection):

    if not file.filename.lower().endswith('.pdf'):
        raise ValueError("File is not acceptable format\n(Acceptable Formats:\n\t'.pdf'\n)")
    vs_ref, page_length, doc_hash = pdf_to_collection(file, collection)

    document = Doc(
        filename = file.filename,
        mimetype= file.mimetype,
        pagelength = page_length,
        uploadDate = datetime.now(),
        contenthash = doc_hash,
        vs_ref = vs_ref
    )
    document.save()
    return document

class MessageType(enum.Enum):
    AIMessage = 'AIMessage'
    HumanMessage = 'HumanMessage'

class ChatMessage(EmbeddedDocument):
    type = EnumField(enum=MessageType)
    content = StringField(required=True)
    timestamp = DateTimeField(default=datetime.now())

    ## Below are additional fields that are unique to {type: AIMessage}
    def message_validation(self):
        return self.type == MessageType.AIMessage

    context = StringField(required=message_validation())
    pagenumber = IntField(required=message_validation())
    docname = StringField(required=message_validation())

class Chat(Document):
    name = StringField()
    docId = ReferenceField(Doc)
    memory = ListField(EmbeddedDocumentListField(ChatMessage))

class Visitor(Document):
    identitycookie = StringField(unique=True)
    sessiontoken = StringField(unique=True)
    documents = ListField(ReferenceField(Doc))
    conversations= ListField(ReferenceField(Chat))
    # lastActive= DateTimeField()


import chromadb
chroma_client = chromadb.PersistentClient(path=os.getenv('CHROMA_PATH'))
        

