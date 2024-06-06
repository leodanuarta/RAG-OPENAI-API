from itertools import chain
import os
import re
import time
import uuid
import PyPDF2
from flask import Flask, request, jsonify
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from pinecone import Pinecone, ServerlessSpec
from tqdm.auto import tqdm
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone as VectorPinecone
from datasets import load_dataset
import fitz  # PyMuPDF
import openai
from PyPDF2 import PdfReader


# Inisialisasi aplikasi Flask
app = Flask(__name__)

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

def create_index_knowledge():
    # Konfigurasi klien Pinecone
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    spec = ServerlessSpec(cloud="aws", region="us-east-1")

    index_name = "llama-2-rag-python-tespdf"
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
    pdf_reader = PyPDF2.PdfReader(file)
    document_text = ""
    for page in pdf_reader.pages:
        document_text += page.extract_text()
    return document_text

# Fungsi untuk mendapatkan metadata dari PDF
def get_pdf_metadata(pdf_path):
    reader = PdfReader(pdf_path)
    info = reader.metadata
    title = info.title if info.title else 'Unknown Title'
    author = info.author if info.author else 'Unknown Author'
    return title, author

# Fungsi untuk membersihkan teks
def clean_text(text):
    # Menghapus karakter-karakter aneh menggunakan regex
    cleaned_text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # Menghapus karakter non-ASCII
    cleaned_text = re.sub(r'[^a-zA-Z0-9\s\.,;?!-]', '', cleaned_text)  # Menghapus karakter khusus kecuali beberapa tanda baca
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)  # Mengganti spasi berlebih dengan satu spasi
    return cleaned_text.strip()

@app.route('/kasihlabira', methods=["POST"])
def upsert_knowledge_pdf():
    openai.api_key = os.getenv('OPENAI_API_KEY')

    if 'file' not in request.files:
        return jsonify({"error": "Body `file` tidak ditemukan" }), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "File harus diinput"}), 400

    if not file.filename.endswith('.pdf'):
        return jsonify({"error": "File harus berbentuk .pdf"}), 400
    
    if file and file.filename.endswith('.pdf'):
        # Simpan file ke server
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        pdf_content = extract_text_from_pdf(file)
        title, author = get_pdf_metadata(file_path)
        index = create_index_knowledge()

        text = clean_text(pdf_content)

        # Memecah teks menjadi paragraf atau bagian yang lebih kecil (opsional)
        text_segments = text.split("\n\n")  # Misalnya, memecah berdasarkan dua baris baru

        # response = openai.Embedding.create(
        #     model="text-embedding-ada-002",
        #     input=text
        # )

        embeddings = embed_model.embed_documents(text_segments)

        

        document_id = str(uuid.uuid4())

        # Ensure we have the same number of embeddings and text segments
        assert len(embeddings) == len(text_segments), "Mismatch between embeddings and text segments"

        # embeddings = [item['embedding'] for item in response.data]

        for i, (embedding, segment) in enumerate(tqdm(zip(embeddings, text_segments), desc="Upserting to Pinecone", total=len(text_segments))):
            metadata = {
                'document_id': document_id,
                'page_number': i,
                'paragraph_number': i,
                'source': file.filename,
                'title': title,
                'author': author,
                'text': segment
            }
            # Ensure embedding is a flat list of floats
            if any(isinstance(item, list) for item in embedding):
                embedding = list(chain.from_iterable(embedding))
            try:
                embedding = [float(value) for value in embedding]
            except ValueError as e:
                print(f"Error converting embedding to float: {e}")
                print(f"Embedding: {embedding}")
                continue  # Skip this embedding if it cannot be converted
            index.upsert(vectors=[(f'{document_id}_{i}', embedding, metadata)])
    
    return jsonify({"text": "Berhasil mempelajari data pdf " + file.filename}), 200

def augment_prompt(query: str):
    index = create_index_knowledge()
    text_field = "text"
    vectorstore = VectorPinecone(index, embed_model.embed_query, text_field)
    results = vectorstore.similarity_search(query, k=3)
    source_knowledge = "\n".join([x.page_content for x in results])
    augmented_prompt = f"""You are a helpful assistant. If the question below requires specific knowledge, use the context provided. Otherwise, answer the question directly.

    Contexts:
    {source_knowledge}

    Query: {query}"""
    return augmented_prompt

@app.route("/tanyalabira", methods=["POST"])
def querying_question():
    body = request.get_json()
    query = body.get("question")

    if not query:
        return jsonify({'error': '[ERROR] Question Needed'}), 400

    # Hybrid prompting approach
    prompt = HumanMessage(content=augment_prompt(query))

    response = chat(initial_messages + [prompt])
    return jsonify({'text': response.content}), 200

if __name__ == "__main__":
    app.run(debug=True)
