from itertools import chain
import os
import re
import time
import uuid
import PyPDF2
from flask import Flask, request, jsonify
from flask_cors import CORS # Import Flask-CORS
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from pinecone import Pinecone, ServerlessSpec
from tqdm.auto import tqdm
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone as VectorPinecone
from datasets import load_dataset
import openai
from PyPDF2 import PdfReader
from flask_cors import CORS
from dotenv import load_dotenv
import redis
import functools

load_dotenv()

# Inisialisasi aplikasi Flask
app = Flask(__name__)
# CORS(app)  # Tambahkan ini untuk mengaktifkan CORS
CORS(app)

# Direktori untuk menyimpan file PDF
UPLOAD_FOLDER = 'uploaded_files'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Inisialisasi model chat
chat = ChatOpenAI(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    model="gpt-3.5-turbo"
)

# Pesan awal
initial_messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="Hi AI, how are you today ?"),
    AIMessage(content="I'm great, thank you. How can I help you ?"),
    HumanMessage(content="I'd like to understand string theory.")
]

# Embedding model
embed_model = OpenAIEmbeddings(model="text-embedding-ada-002")

# Inisialisasi Redis client dengan connection pool
redis_client =redis.ConnectionPool(
    host='redis-19055.c252.ap-southeast-1-1.ec2.redns.redis-cloud.com',
    port=19055,
    password='Csrzh5wQdIzv3aHol7gIye9C7hD7ZwSO'
)

def get_redis_client():
    return redis.StrictRedis(connection_pool=redis_client, decode_responses=True)


def create_index_knowledge(indexName: str):
    # Konfigurasi klien Pinecone
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    spec = ServerlessSpec(cloud="aws", region="us-east-1")

    # index_name = "llama-2-rag-python-tespdf"
    index_name = indexName
    existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

    # Koneksi ke indeks jika sudah ada
    if index_name in existing_indexes:
        index = pc.Index(index_name)
    else:
        pc.create_index(index_name, dimension=1536, metric='dotproduct', spec=spec)
        # Tunggu hingga indeks siap
        while not pc.describe_index(index_name).status['ready']:
            time.sleep(1)
        index = pc.Index(index_name)
        time.sleep(1)
    
    index.describe_index_stats()
    return index

@app.route("/trainlabira", methods=["GET"])
def upsert_knowledge():
    index = create_index_knowledge()
    # Dummy data

    dataset = load_dataset(
        "jamescalam/llama-2-arxiv-papers-chunked",
        split="train"
    )

    data = dataset.to_pandas() 
    batch_size = 100
    for i in tqdm(range(0, len(data), batch_size)):
        i_end = min(len(data), i + batch_size)
        batch = data.iloc[i:i_end]
        ids = [f"{x['doi']}-{x['chunk-id']}" for _, x in batch.iterrows()]
        texts = [x['chunk'] for _, x in batch.iterrows()]
        embeds = embed_model.embed_documents(texts)
        metadata = [{'text': x['chunk'], 'source': x['source'], 'title': x['title']} for _, x in batch.iterrows()]

        index.upsert(vectors=zip(ids, embeds, metadata))
    
    return jsonify({"text": "Berhasil memasukkan data ke vectordb"}), 200

def extract_text_from_pdf(file):
    # pdf_reader = PyPDF2.PdfReader(file)
    # document_text = ""
    # for page in pdf_reader.pages:
    #     document_text += page.extract_text()
    # return document_text
    pdf_reader = PyPDF2.PdfReader(file)
    return "".join([page.extract_text() for page in pdf_reader.pages])

# Fungsi untuk mendapatkan metadata dari PDF
def get_pdf_metadata(pdf_path):
    reader = PdfReader(pdf_path)
    info = reader.metadata
    # title = info.title if info.title else 'Unknown Title'
    # author = info.author if info.author else 'Unknown Author'
    # return title, author
    return info.title or "Unknown Title", info.author or "Unknown Author"


# Fungsi untuk membersihkan teks
def clean_text(text):
    # Menghapus karakter-karakter aneh menggunakan regex
    cleaned_text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # Menghapus karakter non-ASCII
    cleaned_text = re.sub(r'[^a-zA-Z0-9\s\.,;?!-]', '', cleaned_text)  # Menghapus karakter khusus kecuali beberapa tanda baca
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)  # Mengganti spasi berlebih dengan satu spasi
    return cleaned_text.strip()


def chunk_text(text, max_tokens):
    words = text.split()
    chunks, current_chunk = []

    # for word in words:
    #     if len(current_chunk) + len(word) + 1 > max_tokens:
    #         chunks.append(' '.join(current_chunk))
    #         current_chunk = [word]
    #     else:
    #         current_chunk.append(word)

    for word in words:
        if len(" ".join(current_chunk)) + len(word) + 1 > max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
        else:
            current_chunk.append(word)
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks

def recursive_chunk(segment, embed_model, max_payload_size=40960):
    chunks = chunk_text(segment, max_tokens=500)  # Initial chunk size estimate
    embeddings = embed_model.embed_documents(chunks)
    final_chunks, final_embeddings = []

    for chunk, embedding in zip(chunks, embeddings):
        embedding_size = len(embedding) * 8  # Each float is 8 bytes

        if embedding_size > max_payload_size:
            sub_chunks, sub_embeddings = recursive_chunk(chunk, embed_model, max_payload_size)
            final_chunks.extend(sub_chunks)
            final_embeddings.extend(sub_embeddings)
        else:
            final_chunks.append(chunk)
            final_embeddings.append(embedding)

    return final_chunks, final_embeddings

@app.route('/kasihlabira', methods=["POST"])
def upsert_knowledge_pdf():
    openai.api_key = os.getenv('OPENAI_API_KEY')
    
    index_name, namespace = request.args.get("index_name"), request.args.get("namespace")

    if not index_name or not namespace or "file" not in request.files:
        return jsonify({"error" : "[ERROR] Missing `index_name` or `namespace` or Body `file`"}), 400

    file = request.files['file']

    if not file.filename or not file.filename.endswith(".pdf"):
        return jsonify({"error " : "[ERROR] Invalid file format; must be .pdf"}), 400

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    pdf_content = extract_text_from_pdf(file)
    title, author = get_pdf_metadata(file_path)
    index = create_index_knowledge(index_name)

    text = clean_text(pdf_content)

    # Memecah teks menjadi paragraf atau bagian yang lebih kecil (opsional)
    text_segments = text.split("\n\n")  # Misalnya, memecah berdasarkan dua baris baru

    embeddings = embed_model.embed_documents(text_segments)

    document_id = str(uuid.uuid4())

    # Ensure we have the same number of embeddings and text segments
    assert len(embeddings) == len(text_segments), "Mismatch between embeddings and text segments"

    for segment in tqdm(text_segments, desc="Upserting to Pinecone"):
        sub_chunks, sub_embeddings = recursive_chunk(segment, embed_model)
    
    for i, (embedding, sub_chunk) in enumerate(zip(sub_embeddings, sub_chunks)):
        metadata = {
            'document_id': document_id,
            'page_number': i,
            'paragraph_number': i,
            'source': file.filename,
            'title': title,
            'author': author,
            'text': sub_chunk
        }
        # Ensure embedding is a flat list of floats
        try:
            embedding = [float(value) for value in embedding]
            index.upsert(vectors=[(f'{document_id}_{i}', embedding, metadata)], namespace=namespace)
        except ValueError as e:
            print(f"Error converting embedding to float: {e}")
            print(f"Embedding: {embedding}")
            continue  # Skip this embedding if it cannot be converted

    return jsonify({"text": "Berhasil mempelajari data pdf " + file.filename}), 200

# Fungsi untuk caching prompt uang sudah di augment
@functools.lru_cache(maxsize=128)
def augment_prompt(query: str, indexname: str, namespace: str):
    index = create_index_knowledge(indexname)
    text_field = "text"
    vectorstore = VectorPinecone(index, embed_model.embed_query, text_field)
    results = vectorstore.similarity_search(query, k=3, namespace=namespace)
    source_knowledge = "\n".join([x.page_content for x in results])
    augmented_prompt = f"""You are a helpful assistant. If the question below requires specific knowledge, use the context provided. Otherwise, answer the question directly.

    Contexts:
    {source_knowledge}

    Query: {query}"""
    return augmented_prompt


def get_short_term_memory(session_id):
    return get_redis_client().get(session_id) or ""

def update_short_term_memory(session_id, new_memory):
    get_redis_client().set(session_id, new_memory)


@app.route("/tanyalabira", methods=["POST"])
def querying_question():
    body = request.get_json()
    query, indexname, namespace, session_id = body.get("question"), body.get("index_name"), body.get("namespace"), body.get("session_id")

    if not query or not indexname or not namespace or not session_id:
        return jsonify({"error": "[ERROR] Missing Body `question`, `index_name`, `namespace`, or `session_id`"}), 400
    
    short_term_memory = get_short_term_memory(session_id)
    # Hybrid prompting approach
    prompt = HumanMessage(content=augment_prompt(query, indexname, namespace))
    session_messages = initial_messages + [HumanMessage(content=short_term_memory)] if short_term_memory else initial_messages
    response = chat(session_messages + [prompt])

    new_memory = str(short_term_memory, encoding="utf-8") + "\n" + response.content
    update_short_term_memory(session_id, new_memory)

    return jsonify({'text': response.content}), 200


if __name__ == "__main__":
    app.run(debug=True)
