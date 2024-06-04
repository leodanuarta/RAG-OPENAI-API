from flask import Flask, request, jsonify
from flask_caching import Cache
from concurrent.futures import ThreadPoolExecutor
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone as PineconeClient, ServerlessSpec
from fitz import open as open_pdf
import os
import time

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './uploads'
app.config['CACHE_TYPE'] = 'simple'  # Simpel caching menggunakan dictionary
cache = Cache(app)
executor = ThreadPoolExecutor()

messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="Hi AI, how are you today ?"),
    AIMessage(content="I'm great, thank you. How can I help you ?"),
    HumanMessage(content="I'd like to understand string theory.")
]

class Chatbot:
    def __init__(self):
        self.chat = ChatOpenAI(openai_api_key=os.environ["OPENAI_API_KEY"], model="gpt-3.5-turbo")
        self.knowledge_base = KnowledgeBase()
        self.vector_store = self.knowledge_base.vector_store
        self.conversation_memory = {}  # In-memory storage for conversation history

    def generate_response(self, user_id, messages):
        if user_id in self.conversation_memory:
            messages = self.conversation_memory[user_id] + messages
        response = self.chat(messages)
        if response:
            self.update_memory(user_id, messages, response)
        return response.content if response else "Tidak ada respons"
    
    def generate_response_with_knowledge(self, user_id, messages):
        query = messages[-1].content
        query_processor = QueryProcessor(self.vector_store)
        top_results = query_processor.get_top_results([query])
        
        knowledge_content = " ".join([res.metadata.get('content', '') for res in top_results])
        combined_messages = messages + [SystemMessage(content=f"Informasi tambahan: {knowledge_content}")]
        
        if user_id in self.conversation_memory:
            combined_messages = self.conversation_memory[user_id] + combined_messages

        final_response = self.chat(combined_messages)
        if final_response:
            self.update_memory(user_id, combined_messages, final_response)
        return final_response.content if final_response else "Tidak ada respons"

    def update_memory(self, user_id, messages, response):
        if user_id not in self.conversation_memory:
            self.conversation_memory[user_id] = []
        self.conversation_memory[user_id].extend(messages + [AIMessage(content=response.content)])

class QueryProcessor:
    def __init__(self, vector_store):
        self.vector_store = vector_store

    def get_top_results(self, queries, k=3):
        try:
            query_str = " ".join(queries)
            results = self.vector_store.similarity_search(query_str, k=k)
            return results
        except ValueError as ve:
            print("Kesalahan dalam memproses query:", ve)
            return []
        except Exception as e:
            print("Terjadi kesalahan yang tidak terduga:", e)
            return []

class KnowledgeBase:
    def __init__(self):
        self.api_key = os.environ.get('PINECONE_API_KEY')
        self.pc = PineconeClient(api_key=self.api_key)
        self.index_name = "llama-2-rag-python"
        self.vector_store = self.create_vector_store()

    def create_index(self):
        existing_indexes = self.pc.list_indexes()
        if len(existing_indexes) >= 5:
            self.pc.delete_index(existing_indexes[0]['name'])
        if self.index_name not in [index['name'] for index in existing_indexes]:
            spec = ServerlessSpec(cloud="aws", region="us-east-1")
            self.pc.create_index(self.index_name, dimension=1536, metric='dotproduct', spec=spec)
            while not self.pc.describe_index(self.index_name).status['ready']:
                time.sleep(1)
            print("Indeks telah dibuat.")
        else:
            print("Indeks sudah ada.")
        return PineconeVectorStore(index=self.pc.Index(self.index_name), embedding=OpenAIEmbeddings(model="text-embedding-ada-002"))

    def create_vector_store(self):
        self.create_index()
        return PineconeVectorStore(index=self.pc.Index(self.index_name), embedding=OpenAIEmbeddings(model="text-embedding-ada-002"))

    def add_document(self, document_id, content):
        embedding = OpenAIEmbeddings(model="text-embedding-ada-002").embed_text(content)
        self.pc.upsert(index=self.index_name, vectors=[{'id': document_id, 'values': embedding, 'metadata': {'content': content}}])

def extract_text_from_pdf(pdf_path):
    with open_pdf(pdf_path) as doc:
        text = ""
        for page in doc:
            text += page.get_text()
    return text

@app.route('/tanya', methods=['POST'])
def ask_question():
    data = request.get_json()
    user_id = data.get('user_id')
    question = data.get('question')
    if not user_id or not question:
        return jsonify({'error': 'User ID dan pertanyaan diperlukan'}), 400

    chatbot = Chatbot()
    messages = [
        SystemMessage(content="Anda adalah asisten yang membantu."),
        HumanMessage(content=question)
    ]

    response = cache.get(f"{user_id}_response")
    if response is None:
        future = executor.submit(chatbot.generate_response_with_knowledge, user_id, messages)
        response = future.result()
        cache.set(f"{user_id}_response", response, timeout=300)  # Cache response for 5 minutes

    query_processor = QueryProcessor(chatbot.vector_store)
    top_results = query_processor.get_top_results([question])
    serializable_results = [{'metadata': res.metadata} for res in top_results]

    return jsonify({'response': response, 'top_results': serializable_results})

@app.route('/sambutan', methods=['GET'])
def welcome_message():
    chatbot = Chatbot()
    messages = [
        SystemMessage(content="Anda adalah asisten yang membantu."),
        HumanMessage(content="Berikan pesan sambutan.")
    ]
    
    response = chatbot.generate_response('welcome', messages)
    
    return jsonify({'response': response})

@app.route('/upload_pdf', methods=['POST'])
def upload_pdf():
    if 'file' not in request.files:
        return jsonify({'error': 'Tidak ada file yang diunggah'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Nama file kosong'}), 400
    
    if file and file.filename.endswith('.pdf'):
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        text = extract_text_from_pdf(file_path)
        
        knowledge_base = KnowledgeBase()
        document_id = file.filename
        knowledge_base.add_document(document_id, text)
        
        return jsonify({'text': text})
    else:
        return jsonify({'error': 'Format file tidak didukung'}), 400

if __name__ == "__main__":
    app.run(debug=True)
