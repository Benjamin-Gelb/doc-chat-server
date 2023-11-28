from dotenv import load_dotenv
from utils import random_string, doc_to_dict, create_collection, pdf_file_to_text, create_embeddings
import os, enum

from tempfile import mkdtemp
load_dotenv()

### ChromaDB ###
import chromadb
chroma_client = chromadb.PersistentClient(path=os.getenv('CHROMA_PATH'))



### MongoDB ###
from mongoengine import connect, Document, EmbeddedDocumentListField, EmbeddedDocument, ReferenceField, StringField, ListField, DateTimeField, QueryFieldList, UUIDField, EnumField
from datetime import datetime, timedelta


class MessageType(enum.Enum):
    AIMessage = 'AIMessage'
    HumanMessage = 'HumanMessage'

class ChatMessage(EmbeddedDocument):
    type = EnumField(enum=MessageType)
    content = StringField(required=True)

class Session(Document):
    sessionCookie = StringField()
    expiresAt = DateTimeField(default=lambda _ : datetime.now() + timedelta(days=1))
    chromaId = UUIDField()
    chromaName= StringField()
    documents = ListField(StringField(), default = [])
    conversation = EmbeddedDocumentListField(ChatMessage, default = [])

class Visitor(Document):
    persistentCookie = StringField()
    expiresAt= DateTimeField()
    sessions = ListField(ReferenceField(Session), default=[])

def create_new_session():
    cookie_value = random_string(24)
    cookie_expiration = datetime.now() + timedelta(days=1)
    id, name = create_collection(chroma_client)

    # This can be async
    return Session(sessionCookie=cookie_value, expiresAt=cookie_expiration, chromaId=id, chromaName=name)
    # session.save()
    # response.set_cookie('session-cookie', cookie_value, expires=cookie_expiration, httponly=True)



def get_session(session_cookie: str) -> Session:
    query_results = Session.objects(sessionCookie=session_cookie)
    if len(query_results) < 1:
        return None
    session = query_results[0]
    return session

def update_missing_field(session: Session, missing_field: str):
    session[missing_field] = Session[missing_field].default 

def add_docs(session: Session, documents: list[str]):
    try:
        docs = session.documents
        print(docs)
    except Exception as e:
        session.documents = []
        session.save()
    docs = session.documents
    docs.append(documents)
    session.documents = documents
    session.save()

def sessions_from_visitor(visitor: Visitor):
    return [doc_to_dict(session, include=['conversation', 'documents', 'sessionCookie']) for session in visitor.sessions]


### Flask ###

from flask import Flask, make_response, Response, request
from flask_cors import CORS, cross_origin
from typing import List, Any

app = Flask(__name__)

CORS(app, supports_credentials=True, origins="http://localhost:5173")
db = connect(db=os.getenv('MONGO_URI'))

class SessionData:
    activeDocuments: List[str]
    conversation: List[Any]

class DefaultAPIResponse:
    activeSession: bool
    sessionData: SessionData

@app.route('/session', methods=['GET'])
def retrieve_cookie():
    default_response= make_response({
            'exists': False,
            'data': None
        })
    cookie_value = request.cookies.get('session-cookie', None)
    if not cookie_value:
        return default_response
    
    sessions = Session.objects(sessionCookie=cookie_value)
    if len(sessions) < 1:
        return default_response
    
    session_data = [doc_to_dict(session, exclude=['sessionCookie','expiresAt','chromaId','chromaName', '_id']) for session in sessions]
    return make_response({
        'exists': True,
        'sessions': session_data
    })

@app.route('/visitor', methods=['GET'])
def get_visitor():
    persistent_cookie = request.cookies.get('persistent-cookie', None)
    if not persistent_cookie:
        return make_response({
            'exists': False,
            'data': None 
        })
    visitor : Visitor = Visitor.objects(persistentCookie=persistent_cookie).first()

    if not visitor:
        return make_response({
            'exists': False,
            'data': None 
        })
    response = make_response({
        'exists' : True,
        'sessions': [doc_to_dict(session, include=['conversation', 'documents', 'sessionCookie'])  for session in visitor.sessions]
    })
    return response
    

@app.route('/visitor', methods=['POST'])
def create_visitor():
    session = create_new_session()
    session.save()

    persistent_cookie = random_string(64)
    cookie_expiration = datetime.now() + timedelta(days=30)
    visitor = Visitor(persistentCookie=persistent_cookie, expiresAt=cookie_expiration, sessions=[session.id])
    visitor.save()

    visitor : Visitor = Visitor.objects(persistentCookie=persistent_cookie).first()
    sessions = [doc_to_dict(session, include=['conversation', 'documents', 'sessionCookie']) for session in visitor.sessions]
    response = make_response({
        'sessions': sessions
    })
    response.set_cookie('session-cookie', session.sessionCookie, expires=session.expiresAt)
    response.set_cookie('persistent-cookie', visitor.persistentCookie, expires=visitor.expiresAt, httponly=True)

    return response

@app.route('/session', methods=['POST'])
def create_session():

    persistent_cookie = request.cookies.get('persistent-cookie', None)
    if not persistent_cookie:
        return make_response({
            'exists': False,
            'data': None 
        })

    # This can be async
    session = create_new_session()
    session.save()
    visitor : Visitor = Visitor.objects(persistentCookie=persistent_cookie).first()
    visitor.sessions.append(session.id)
    visitor.save()

    visitor : Visitor = Visitor.objects(persistentCookie=persistent_cookie).first()
    response = make_response({
        'sessions': [doc_to_dict(session, include=['conversation', 'documents', 'sessionCookie']) for session in visitor.sessions]
    })

    response.set_cookie('session-cookie', session.sessionCookie, expires=session.expiresAt)
    return response

# @app.route('/session', methods=['PUT'])
# def set_cookie():
#     persistent_cookie = request.cookies.get('persistent-cookie', None)
#     visitor : Visitor = Visitor.objects(persistentCookie=persistent_cookie).first()


#     body = request.json
#     if body['sessionCookie']:
#         #validate session cookie
#         lambda x : x.sessionCookie ==
#         if visitor.sessions
#         return make_response({'sessions': sessions_from_visitor(visitor)}).set_cookie('session-cookie', )
#     return 



@app.route('/document', methods=['POST'])
def upload_document():
    session_cookie = request.cookies.get('session-cookie', None)
    session = get_session(session_cookie)
    if not session:
        return Response("Missing or outdated session-cookie.", status=404)
    uploaded_documents = []
    try:
        collection = chroma_client.get_collection(name=session.chromaName)
        items = request.files.getlist('files')
        for file in items:
            if file.filename.lower().endswith('.pdf'):
                full_text = pdf_file_to_text(file)
                ids, documents, embeddings = create_embeddings(full_text)
                collection.add(
                    ids=ids,
                    documents=documents,
                    embeddings=embeddings,
                    metadatas=[{
                        'filename' : file.filename,
                        'section' : n
                    } for n in range(len(ids))]
                )
                uploaded_documents.append(file.filename)
        add_docs(session, uploaded_documents)

    except Exception as e:
        print(e)
        return Response("There was an internal issue with your request.", status=500)
    return make_response({
        'documents': uploaded_documents
    })
        
@app.route('/document', methods=['DELETE'])
def clear_collection():
    session_cookie = request.cookies.get('session-cookie')
    session = get_session(session_cookie)

    try:
        collection = chroma_client.get_collection(name=session.chromaName)
        all_docs = collection.get()['ids']
        if all_docs:
            collection.delete(ids=all_docs)
        session.update(set__documents=[])
    except Exception as e:
        print(e)

    return Response("Resources were deleted from collection.", status=200)

from config import LLMConfig
from utils import talk_to_doc

@app.route('/chat', methods=['POST'])
def chat_with_doc():

    user_message = request.json['content']

    session_cookie = request.cookies.get('session-cookie')
    session = get_session(session_cookie)
    conversation_history = session.conversation

    docs= chroma_client.get_collection(name=session.chromaName)
    chain = LLMConfig.create_chain(conversation_history=conversation_history)

    response_content: str = talk_to_doc(docs=docs, user_message=user_message, chain=chain)

    user = ChatMessage(type=MessageType.HumanMessage, content=user_message)
    ai =ChatMessage(content=response_content, type=MessageType.AIMessage)

    conversation_history.extend([user,ai])
    session.conversation = conversation_history
    session.save()
    

    return make_response(ai.to_mongo().to_dict())

from config import stream_response
    
@app.route('/stream', methods=['POST'])
def stream_chat():
    user_message = request.json['content']

    session_cookie = request.cookies.get('session-cookie')
    session = get_session(session_cookie)
    embeddings= create_embeddings(user_message)[2]
    # conversation_history = session.conversation

    docs= chroma_client.get_collection(name=session.chromaName)
    # chain = LLMConfig.create_chain(conversation_history=conversation_history)
    gen = stream_response(user_message=user_message, embeddings=embeddings, docs=docs)

    return Response(gen, content_type='text/plain')


# @app.route('/chat', methods=['PUT'])
# def store_chat():
#     session_cookie = request.cookies.get('session-cookie', None)
#     if not session_cookie:
#         return make_response
    
#     sessions = Session.objects(sessionCookie=session_cookie)

if __name__ == '__main__':
    app.run(debug=True)
