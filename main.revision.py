from mongoengine import (
    Document, StringField, IntField, DateTimeField, ListField,
    ReferenceField, EnumField, EmbeddedDocument,
    EmbeddedDocumentListField, UUIDField
)
from chromadb import Collection
from werkzeug.datastructures import FileStorage
from PyPDF2 import PdfReader, PageObject
from langchain.chains.base import Chain 
import hashlib, enum, random, string, os
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from chromadb import api
from dotenv import load_dotenv


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

Document.to_dict = to_dict

load_dotenv()

def generate_unique_char_string(length=64):
    """ Generate a random string of fixed length """
    letters = string.ascii_letters + string.digits
    return ''.join(random.choice(letters) for _ in range(length))

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
    contenthash = StringField()
    vs_ref = StringField()


    chunk_size = 1000
    chunk_overlap = 100

    @classmethod
    def split_by(self):
        return (self.chunk_size, self.chunk_overlap)

    def stitch(self, collection: Collection):
        doc_chunks = collection.query(where={
            'vs_ref': {
                "$eq": self.vs_ref
            }
        })
        print(doc_chunks)

        





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
    return (vs_ref, reader.numPages, document_hash.digest())

def reader_to_summary(reader : PdfReader, summary_chain: Chain):
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return summary_chain.invoke({
        'document': text
    })

from datetime import datetime

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
    name = StringField()
    docId = ReferenceField(Doc)
    memory = ListField(EmbeddedDocumentListField(ChatMessage))

import jwt
from chromadb import api


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
        self.documents.filter(doc_id=id)

    def get_collection(self, client: api.ClientAPI)-> Collection:
        return client.get_collection(name=self.collectionName)

    @staticmethod
    def cookie_field():
        return 'identity-key'


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

from chromadb import Embeddings, PersistentClient, EmbeddingFunction
from typing import Any

class CustomEmbeddingFunction(EmbeddingFunction):

    embedding_model = OpenAIEmbeddings()
    def __call__(self, input: Any) -> Embeddings:
        return self.embedding_model.embed_documents(input)

def create_collection(client: api.ClientAPI) -> Collection:
    name = generate_unique_char_string(32)
    return client.create_collection(name=name, embedding_function=CustomEmbeddingFunction())

def retrieve_collection(name: str, client: api.ClientAPI):
    return client.get_collection(name=name, embedding_function=CustomEmbeddingFunction())



client = PersistentClient(path=os.getenv('CHROMA_PATH'))


from flask import Response, Request, request, Flask, make_response, abort
from flask_cors import CORS
from mongoengine import connect

app = Flask(__name__)

CORS(app, supports_credentials=True, ) # origins=os.getenv('CLIENT_HOST')
db = connect(db=os.getenv('MONGO_URI'))

def missing_resource(message: str):
    return Response(
            response=message,
            status=404
        )

def find_visitor(request: Request) -> Visitor:
    encoded_payload = request.cookies.get(Visitor.cookie_field(), None)
    if not encoded_payload:
        return None
    
    identity_key = retrieve_token(encoded_payload, Visitor.cookie_field())
    return Visitor.objects(identitykey=identity_key).first()

@app.route('/visitor', methods=['GET'])
def get_visitor():
    visitor = find_visitor(request)
    if not visitor:
        return missing_resource('Missing identity-key make POST to /visitor to receive one on client.')
    return make_response(visitor.to_dict(exclude=['identitykey', '_id'])), 200

@app.route('/visitor', methods=['POST'])
def create_visitor():

    key = generate_unique_char_string(64)
    collection : Collection = create_collection(client)
    encoded_key = gen_token(key, Visitor.cookie_field())
    visitor = Visitor(
        identitykey=key,
        collectionName=collection.name,
        collectionId=collection.id
    )
    visitor.save()

    response = make_response(visitor.to_dict(exclude=['identitykey', '_id']))
    response.set_cookie(Visitor.cookie_field(), value=encoded_key)
    return response, 201

@app.route('/document', methods=['GET'])
def retrieve_document_list():
    """Returns paginated documents."""
    page = request.query_string or 1
    visitor = find_visitor(request)
    if not visitor:
        return missing_resource('Missing identity-key make POST to /visitor to receive one on client.')
    documents = visitor.documents[:10*page]
    return make_response(documents), 200

@app.route('/documents/<id>', methods=['GET'])
def get_document(id: str):
    visitor = find_visitor(request)
    return make_response(visitor.documents.filter(doc_id=id).first()), 200

@app.route('/document', methods=['POST'])
def post_doc():
    visitor = find_visitor(request)

    files = request.files.getlist('files')
    collection = visitor.get_collection(client)

    doc_list = []
    for file in files:
        uploaded_doc = upload_pdf(file, collection)
        doc_list.append(uploaded_doc.to_dict(exclude=['_id']))
    return make_response(doc_list), 201

@app.route('/document/<id>/summary', methods=['PUT'])
def generate_summary(id: str):
    visitor = find_visitor(request)

    collection : Collection = visitor.get_collection()
    document : Doc = visitor.get_document(id)
    document.stitch(collection)


if __name__ == '__main__':
    app.run(debug=True)

        
    
    # def evaluate_token(self, named_field: str, key: str):
    #     try:
    #         secret = os.getenv('JWT_SECRET')
    #         decoded_payload = jwt.decode(key, secret, algorithms=os.getenv('JWT_ALGORITHM'))
    #         key = decoded_payload[named_field]
    #     except jwt.ExpiredSignatureError:
    #         # Handle expired token
    #         pass
    #     except jwt.InvalidTokenError:
    #         # Handle invalid token




import chromadb
chroma_client = chromadb.PersistentClient(path=os.getenv('CHROMA_PATH'))
        

