import os
from langchain.chat_models import ChatOpenAI
from flask import Flask, request, jsonify
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)

app = Flask(__name__)


chat = ChatOpenAI(
    openai_api_key = os.getenv("OPENAI_API_KEY"),
    model= "gpt-3.5-turbo"
)

messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="Hi AI, how are you today ?"),
    AIMessage(content="I'm great, thank you. How can I help you ?"),
    HumanMessage(content="I'd like to understand string theory.")
]

res = chat(messages)

# add lates AI response to message
messages.append(res)

# now create a new user prompt
prompt = HumanMessage(
    content="Why do physicists believe it can produce a 'unified theory'?"
)

# add to messages
messages.append(prompt)

# send to chat-gpt (gpt-3.5-turbo)
res = chat(messages)


# add latest AI response to messages
messages.append(res)

# now create a new user prompt
prompt = HumanMessage(
    content="Can you tell me about the LLMChain in LangChain?"
)
# add to messages
messages.append(prompt)

# send to OpenAI
res = chat(messages)

# Dealing with hallucinations
print(res.content)

# Adding knowledge to models
llmchain_information = [
    "A LLMChain is the most common type of chain. It consists of a PromptTemplate, a model (either an LLM or a ChatModel), and an optional output parser. This chain takes multiple input variables, uses the PromptTemplate to format them into a prompt. It then passes that to the model. Finally, it uses the OutputParser (if provided) to parse the output of the LLM into a final format.",
    "Chains is an incredibly generic concept which returns to a sequence of modular components (or other chains) combined in a particular way to accomplish a common use case.",
    "LangChain is a framework for developing applications powered by language models. We believe that the most powerful and differentiated applications will not only call out to a language model via an api, but will also: (1) Be data-aware: connect a language model to other sources of data, (2) Be agentic: Allow a language model to interact with its environment. As such, the LangChain framework is designed with the objective in mind to enable those types of applications."
]

source_knowledge = "\n".join(llmchain_information)

query = "Can you tell me about the LLMChain in LangChain ?"

augmented_prompt = f"""Using the contexts below, answer the query.

Contexts:
{source_knowledge}

Query: {query}"""

# create a new user prompt
prompt = HumanMessage(
    content = augmented_prompt
)

# add to messages
messages.append(prompt)

# send to OpenAI
res = chat(messages)

print("After adding knowledge", res.content)

# adding pinecone to models
import os
from pinecone import Pinecone

# initialize connection to pinecone (get API key at app.pinecone.io)
api_key = os.environ.get('PINECONE_API_KEY') or '6069b9e1-48d5-414c-976c-758c2cdffd8b'

# configure client
pc = Pinecone(api_key=api_key)

from pinecone import ServerlessSpec

spec = ServerlessSpec(
    cloud="aws", region="us-east-1"
)

import time

index_name = "llama-2-rag-python"
existing_indexes = [
    index_info["name"] for index_info in pc.list_indexes()
]

# check if index already exists
if index_name not in existing_indexes:
    # if does not exist, create index
    pc.create_index(
        index_name,
        dimension=1536,
        metric='dotproduct',
        spec=spec
    )

    # wait for index to be initialized
    while not pc.describe_index(index_name).status['ready']:
        time.sleep(1)

# connect to index
index = pc.Index(index_name)
time.sleep(1)
# view index stats
index.describe_index_stats()

from langchain.embeddings.openai import OpenAIEmbeddings

embed_model = OpenAIEmbeddings(model="text-embedding-ada-002")

texts = [
    'this is the first chunk of text',
    'then another second chunk of text is here'
]

res = embed_model.embed_documents(texts)
len(res), len(res[0])

# Upsert to Pinecone 
from tqdm.auto import tqdm # for progress bar

# data = dataset.to_pandas()  # this makes it easier to iterate over the dataset

# batch_size = 100

# for i in tqdm(range(0, len(data), batch_size)):
#     i_end = min(len(data), i+batch_size)
#     # get batch of data
#     batch = data.iloc[i:i_end]
#     # generate unique ids for each chunk
#     ids = [f"{x['doi']}-{x['chunk-id']}" for i, x in batch.iterrows()]
#     # get text to embed
#     texts = [x['chunk'] for _, x in batch.iterrows()]
#     # embed text
#     embeds = embed_model.embed_documents(texts)
#     # get metadata to store in Pinecone
#     metadata = [
#         {'text' : x['chunk'],
#         'source' : x['source'],
#         'title':x['title']} for i, x in batch.iterrows()
#     ]

#     # add to Pinecone
#     index.upsert(vectors=zip(ids, embeds, metadata))


# Adding RAG To Model
from langchain.vectorstores import Pinecone

text_field = "text"  # the metadata field that contains our text

# initialize the vector store object
vectorstore = Pinecone(
    index, embed_model.embed_query, text_field
)

# Question to models 
query = "What is so special about Llama 2?"  # Gantilah dengan query yang sesuai
namespace = ""  # Gantilah dengan namespace yang sesuai
top_k = 3  # Gantilah dengan jumlah k yang sesuai

# Melakukan pencarian similarity menggunakan query
results = vectorstore.similarity_search(query, top_k)

# Memotong hasil sesuai dengan top_k
top_results = results[:top_k]

# Tampilkan hasil
print("hasil top result", top_results)


# Testing Hasil Anggap ini kayak mainnya
query = "What is so special about Llama 2?"

vectorstore.similarity_search(query, k=3)


def augment_prompt(query : str):
    # get top 3 results from knowledge base
    results = vectorstore.similarity_search(query, k=3)
    # get the text from the results
    source_knowledge = "\n".join([x.page_content for x in results])
    # feed into an augmented prompt
    augmented_prompt = f"""Using the context below, answer the query.

    Contexts:
    {source_knowledge}

    Query: {query}"""
    return augmented_prompt

print(augment_prompt(query))

# create a new user prompt
prompt = HumanMessage(
    content=augment_prompt(query)
)

# add to messages
messages.append(prompt)

res = chat(messages)

print(res.content)

# Add RAG 
prompt = HumanMessage(
    content=augment_prompt(
        "what safety measures were used in the development of llama 2?"
    )
)

res = chat(messages + [prompt])
print("Hasil", res.content)

if __name__ == "__main__":
    app.run(debug=True)


