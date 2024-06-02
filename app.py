import os
import time
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from datasets import load_dataset
import pinecone
from tqdm.auto import tqdm
from pinecone import Pinecone as PineconeClient, ServerlessSpec
from flask import Flask, request, jsonify

app = Flask(__name__)

class Chatbot:
    def __init__(self):
        os.environ["OPENAI_API_KEY"] = "sk-proj-dUFywzoCHYLXpcrGgDwwT3BlbkFJDKhmcIsnRdqNadSE2Fow"
        self.chat = ChatOpenAI(openai_api_key=os.environ["OPENAI_API_KEY"], model="gpt-3.5-turbo")

    def generate_response(self, messages):
        response = self.chat(messages)
        return response

class DataProcessor:
    def __init__(self, dataset_name):
        try:
            self.dataset = load_dataset(dataset_name, split="train")
        except KeyError:
            raise ValueError(f"Dataset '{dataset_name}' tidak ditemukan. Mohon periksa nama dataset dan coba lagi.")

    def process_data(self):
        return self.dataset.to_pandas()

class KnowledgeBase:
    def __init__(self):
        self.api_key = os.environ.get('PINECONE_API_KEY')
        self.pc = PineconeClient(api_key=self.api_key)
        self.spec = ServerlessSpec(cloud="aws", region="us-east-1")
        self.index_name = "llama-2-rag-python"

    def create_index(self):
        existing_indexes = self.pc.list_indexes()
        if len(existing_indexes) >= 5:
            self.pc.delete_index(existing_indexes[0]['name'])
        if self.index_name not in [index['name'] for index in existing_indexes]:
            self.pc.create_index(self.index_name, dimension=1536, metric='dotproduct', spec=self.spec)
            while not self.pc.describe_index(self.index_name).status['ready']:
                time.sleep(1)
        
        index_description = self.pc.describe_index(self.index_name)
        print("Deskripsi indeks:", index_description)

    def delete_index(self):
        self.pc.delete_index(self.index_name)

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
        except Exception as e:
            print("Terjadi kesalahan yang tidak terduga:", e)

def store_data_in_pinecone(data, embed_model, index):
    batch_size = 100

    for i in tqdm(range(0, len(data), batch_size)):
        i_end = min(len(data), i+batch_size)
        batch = data.iloc[i:i_end]
        ids = [f"{x['doi']}-{x['chunk-id']}" for i, x in batch.iterrows()]
        texts = [x['chunk'] for _, x in batch.iterrows()]
        embeds = embed_model.embed_documents(texts)
        metadata = [
            {'text': x['chunk'],
            'source': x['source'],
            'title': x['title']} for i, x in batch.iterrows()
        ]
        index.upsert(vectors=zip(ids, embeds, metadata))

@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.get_json()
    question = data.get('question')
    if not question:
        return jsonify({'error': 'Pertanyaan diperlukan'}), 400
    
    chatbot = Chatbot()
    messages = [
        SystemMessage(content="Anda adalah asisten yang membantu."),
        HumanMessage(content=question)
    ]
    
    response = chatbot.generate_response(messages)
    response_content = response.content if response else "Tidak ada respons"
    
    return jsonify({'response': response_content})

@app.route('/welcome', methods=['GET'])
def welcome_message():
    chatbot = Chatbot()
    messages = [
        SystemMessage(content="Anda adalah asisten yang membantu."),
        HumanMessage(content="Apa yang bisa kamu bantu ?")
    ]
    
    response = chatbot.generate_response(messages)
    response_content = response.content if response else "Tidak ada respons"
    
    return jsonify({'response': response_content})

def main():
    chatbot = Chatbot()
    dataset_name = "jamescalam/llama-2-arxiv-papers-chunked"
    
    try:
        data_processor = DataProcessor(dataset_name)
        data = data_processor.process_data()
        print(data.head())
        print(data.columns)
    except ValueError as e:
        print(e)
        return

    knowledge_base = KnowledgeBase()
    knowledge_base.create_index()

    embed_model = OpenAIEmbeddings(model="text-embedding-ada-002")

    index = knowledge_base.pc.Index(knowledge_base.index_name)
    
    vector_store = PineconeVectorStore(index=index, embedding=embed_model)

    query_processor = QueryProcessor(vector_store)

    queries = ["Apa yang istimewa dari Llama 2?", "Bagaimana cara kerjanya?", "Ceritakan tentang nanas"]
    for query in queries:
        top_results = query_processor.get_top_results([query])
        print(f"Hasil untuk query '{query}': {top_results}")

if __name__ == "__main__":
    app.run(debug=True)
