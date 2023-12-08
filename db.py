from mongoengine import (
    Document, StringField, IntField, DateTimeField, ListField,
    ReferenceField, EnumField, EmbeddedDocument,
    EmbeddedDocumentListField, UUIDField, BinaryField
)
from chromadb import Collection
from werkzeug.datastructures import FileStorage
from PyPDF2 import PdfReader, PageObject
from langchain.chains.base import Chain 
import hashlib, enum, random, string, os
from langchain.text_splitter import CharacterTextSplitter
from chromadb import api
from dotenv import load_dotenv


from langchain import schema

from util import to_dict, generate_unique_char_string,extract_text, CustomEmbeddingFunction


load_dotenv()

Document.to_dict = to_dict

class Doc(Document):
    ### Document Data
    filename = StringField()
    mimetype = StringField()
    pagelength = IntField(min_value=0)
    uploadDate = DateTimeField()
    summary = StringField()
    doc_id = StringField()


    # Retrival Data
    # priveledge = StringField() # maybe hash of session-token (when I create one) and content-hash
    contenthash = BinaryField()
    vs_ref = StringField()


    chunk_size = 1000
    chunk_overlap = 100

    @classmethod
    def split_by(self):
        return (self.chunk_size, self.chunk_overlap)

    def stitch(self, collection: Collection):

        doc_chunks = collection.get(where={
            'vs_ref': self.vs_ref
        })

        start = self.chunk_overlap
        end = self.chunk_size-self.chunk_overlap

        full_text = ""
        for i, chunk in enumerate(doc_chunks['documents']):
            if i == 0: start = 0
            if i == len(doc_chunks['documents'])-1: end = end + self.chunk_overlap
            full_text += chunk[start :end]

        return full_text
    
    def stitch_as_docs(self, collection: Collection):
        doc_chunks = collection.get(where={
            'vs_ref': self.vs_ref
        })

        start = self.chunk_overlap
        end = self.chunk_size-self.chunk_overlap

        document_list : list[schema.Document] = []
        for i, chunk in enumerate(doc_chunks['documents']):
            if i == 0: start = 0
            if i == len(doc_chunks['documents'])-1: end = end + self.chunk_overlap
            document_list.append(schema.Document(page_content=chunk[start :end]))
        return document_list
    
    def retrieve_context(self, collection: Collection, **kwargs : Collection.query):
        result = collection.query(
            where={
                'vs_ref': self.vs_ref
            },
            **kwargs
        )
        return result

def pdf_to_collection(file: FileStorage, collection: Collection) -> (str, int, str):
    """Inserts File content into chromadb collection and returns relevant metadata generated from insertion"""

    ## In its current iteration this is not fault tolerant for pdf files where the page size is massive.

    vs_ref = generate_unique_char_string(64) ## Unique value that ties the Database to Chromadb

    reader = PdfReader(file.stream)
    ids = []
    documents = []
    metadatas = []
    document_hash = hashlib.sha256()

    chunk_size, chunk_overlap = Doc.split_by()
    
    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separator='\n'
    )

    full_text, get_page_number = extract_text(reader)

    char_prog = 0
    docs = text_splitter.split_text(full_text)
    for iter, doc in enumerate(docs):

        char_prog += len(doc)
        hash = hashlib.sha256()
        string_encoding = doc.encode()

        hash.update(string_encoding)
        documents.append(doc)
        ids.append(hash.hexdigest())
        metadatas.append({
            'page_number': get_page_number(char_prog),
            'char_length': len(doc),
            # 'filename': file.filename,
            # 'mimetype': file.mimetype,
            'chunk' : iter,
            'vs_ref': vs_ref
        })
        document_hash.update(string_encoding)
    
    collection.add(ids, None, metadatas, documents)
    return (vs_ref, len(reader.pages), document_hash.digest())


def upload_pdf(file: FileStorage, collection: Collection) -> Doc:

    if not file.filename.lower().endswith('.pdf'):
        raise ValueError("File is not acceptable format\n(Acceptable Formats:\n\t'.pdf'\n)")
    vs_ref, page_length, doc_hash = pdf_to_collection(file, collection)

    document = Doc(
        filename = file.filename,
        doc_id = generate_unique_char_string(18),
        mimetype= file.mimetype,
        pagelength = page_length,
        uploadDate = datetime.now(),
        contenthash = doc_hash,
        vs_ref = vs_ref
    )
    document.save()
    return document

from datetime import datetime

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

class Chat(Document):
    title = StringField()
    doc_id = ReferenceField(Doc)
    chat_id = StringField(default=generate_unique_char_string(18))
    memory = ListField(EmbeddedDocumentListField(ChatMessage), default=[])

class Visitor(Document):
    identitykey = StringField(unique=True)
    documents = ListField(ReferenceField(Doc), default=[])
    conversations= ListField(ReferenceField(Chat), default=[])
    collectionName= StringField()
    collectionId = UUIDField()
    # lastActive= DateTimeField()
            

    def update_conversation(self, messages: list[ChatMessage]):
        self.conversations.extend(messages)
        self.save()

    def get_document(self, id: str) -> Doc:
        print(self.documents)
        for document in self.documents:
            if document.doc_id == id:
                return document
    
    def get_conversation(self, id: str) -> Chat:
        for chat in self.conversations:
            if chat.chat_id == id:
                return chat
            

    def get_collection(self, client: api.ClientAPI)-> Collection:
        return client.get_collection(name=self.collectionName, embedding_function=CustomEmbeddingFunction())

    @staticmethod
    def cookie_field():
        return 'identity-key'


