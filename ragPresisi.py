import os
import time
from flask import Flask, request, jsonify
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from pinecone import Pinecone, ServerlessSpec
from tqdm.auto import tqdm
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone as VectorPinecone

# Inisialisasi aplikasi Flask
app = Flask(__name__)

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

    index_name = "llama-2-rag-python"
    existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

    # Cek apakah indeks sudah ada
    if index_name not in existing_indexes:
        pc.create_index(index_name, dimension=1536, metric='dotproduct', spec=spec)
        # Tunggu hingga indeks siap
        while not pc.describe_index(index_name).status['ready']:
            time.sleep(1)

    # Koneksi ke indeks
    index = pc.Index(index_name)
    time.sleep(1)
    index.describe_index_stats()
    return index

def upsert_knowledge():
    index = create_index_knowledge()
    data = ""  # Dummy data

    batch_size = 100
    for i in tqdm(range(0, len(data), batch_size)):
        i_end = min(len(data), i + batch_size)
        batch = data.iloc[i:i_end]
        ids = [f"{x['doi']}-{x['chunk-id']}" for _, x in batch.iterrows()]
        texts = [x['chunk'] for _, x in batch.iterrows()]
        embeds = embed_model.embed_documents(texts)
        metadata = [{'text': x['chunk'], 'source': x['source'], 'title': x['title']} for _, x in batch.iterrows()]

        index.upsert(vectors=zip(ids, embeds, metadata))

def augment_prompt(query: str):
    index = create_index_knowledge()
    text_field = "text"
    vectorstore = VectorPinecone(index, embed_model.embed_query, text_field)
    results = vectorstore.similarity_search(query, k=3)
    source_knowledge = "\n".join([x.page_content for x in results])
    augmented_prompt = f"""Using the context below, answer the query.

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

    prompt = HumanMessage(content=augment_prompt(query))
    response = chat(initial_messages + [prompt])
    return jsonify({'text': response.content}), 200

if __name__ == "__main__":
    app.run(debug=True)
