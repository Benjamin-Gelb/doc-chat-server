from dotenv import load_dotenv
from utils import random_string, doc_to_dict, create_collection, pdf_file_to_text, create_embeddings
import os

from tempfile import mkdtemp
load_dotenv()

### ChromaDB ###
import chromadb
chroma_client = chromadb.PersistentClient(path=os.getenv('CHROMA_PATH'))



### MongoDB ###
from mongoengine import connect, Document, StringField, ListField, DateTimeField, QueryFieldList, UUIDField
from datetime import datetime, timedelta


class Session(Document):
    sessionCookie = StringField()
    expiresAt = DateTimeField(default=lambda _ : datetime.now() + timedelta(days=1))
    chromaId = UUIDField()
    chromaName= StringField()
    documents = ListField(StringField(), default=[])
    conversation = ListField(StringField(), default = [])





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
    print(docs)
    docs.append(documents)
    session.documents = documents
    session.save()

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
    session_data = doc_to_dict(sessions[0], exclude=['sessionCookie','expiresAt','chromaId','chromaName', '_id'])
    return make_response({
        'exists': True,
        'data': session_data
    })

@app.route('/session', methods=['POST'])
def set_cookie(status_code: int = 200):
    response = make_response({
        'message': 'Cookie sent!'
    }, status_code=status_code)

    cookie_value = random_string(24)
    cookie_expiration = datetime.now() + timedelta(days=1)
    id, name = create_collection(chroma_client)

    # This can be async
    session = Session(sessionCookie= cookie_value, expiresAt=cookie_expiration, chromaId=id, chromaName=name)
    session.save()
    # End block

    response.set_cookie('session-cookie', cookie_value, expires=cookie_expiration, httponly=True)
    return response

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
                        'filename' : file.filename
                    } for _ in range(len(ids))]
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

# @app.route('/chat', methods=['PUT'])
# def store_chat():
#     session_cookie = request.cookies.get('session-cookie', None)
#     if not session_cookie:
#         return make_response
    
#     sessions = Session.objects(sessionCookie=session_cookie)

if __name__ == '__main__':
    app.run(debug=True)
