from chromadb import Collection
import os
from dotenv import load_dotenv
from chromadb import PersistentClient

from flask import Response, Request, request, Flask, make_response, abort
from flask_cors import CORS
from mongoengine import connect
from chains.summary import stuff 

from db import Visitor, Doc, Chat, ChatMessage, MessageType, upload_pdf
from util import retrieve_token, generate_unique_char_string, create_collection, gen_token
from langchain import schema

load_dotenv()


client = PersistentClient(path=os.getenv('CHROMA_PATH'))

app = Flask(__name__)

CORS(app, supports_credentials=True, ) # origins=os.getenv('CLIENT_HOST')
db = connect(db=os.getenv('MONGO_URI'))

def missing_resource(message: str):
    return Response(
            response=message,
            status=404
        )

def unauthorized(message: str):
    return Response(
        response=message,
        status=401
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
    return make_response([doc.to_dict(exclude=['_id', 'contenthash']) for doc in documents]), 200

@app.route('/documents/<id>', methods=['GET'])
def get_document(id: str):
    visitor = find_visitor(request)
    document : Doc = visitor.get_document(id)
    if document.summary:
        return make_response(document.to_dict(exclude=['_id', 'contenthash'])), 200
    return missing_resource("Document summary not generated yet, try /id/summary.")

@app.route('/document', methods=['POST'])
def post_doc():
    visitor = find_visitor(request)

    files = request.files.getlist('files')
    collection = visitor.get_collection(client)

    doc_list = []
    for file in files:
        doc = upload_pdf(file, collection)
        doc_list.append(doc.to_dict(exclude=['_id', 'contenthash']))
        visitor.documents.append(doc.id)
    visitor.save()
    return make_response(doc_list), 201

from chains.summary import stuff 

@app.route('/document/<id>/summary', methods=['PUT'])
def generate_summary(id: str):

    visitor = find_visitor(request)
    collection : Collection = visitor.get_collection(client)
    document : Doc = visitor.get_document(id)
    if document.summary:
        return make_response(document.to_dict(exclude=['_id', 'contenthash'])), 200

    docs = document.stitch_as_docs(collection)

    summary = stuff(docs)
    document.summary = summary
    document.save()

    return make_response(document.to_dict(exclude=['_id', 'contenthash'])), 201

@app.route('/chat/<doc_id>', methods=['POST'])
def chat_from_doc(doc_id: str):
    visitor = find_visitor(request)
    if not visitor:
        return missing_resource('Missing identity-key make POST to /visitor to receive one on client.')
    chat = Chat(doc_id= doc_id)
    chat.save()
    visitor.conversations.append(chat.id)
    visitor.save()
    return visitor.get_conversation(chat.chat_id).to_dict(exclude=['_id'])

@app.route('/chat', methods=['GET'])
def get_chats():
    """Returns paginated documents."""
    page = request.query_string or 1
    visitor = find_visitor(request)
    if not visitor:
        return missing_resource('Missing identity-key make POST to /visitor to receive one on client.')
    chats = visitor.conversations[:page*10]
    return make_response([chat.to_dict(exclude=['_id']) for chat in chats])

from .chains.restate import setup

@app.route('/chat/<chat_id>', methods=['PUT'])
def send_message(chat_id: str):
    visitor = find_visitor(request)

    body = dict(request.json)
    message_content = body.get('content')

    if not visitor:
        return missing_resource('Missing identity-key make POST to /visitor to receive one on client.')
    collection = visitor.get_collection(client)
    conversation: Chat = visitor.get_conversation(chat_id)
    document = visitor.get_document(conversation.doc_id)
    context  = document.retrieve_context(collection, query_texts=schema.Document(message_content), k=3)
    setup(chat_history=conversation.memory, summary=document.summary, context=context)







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
        

