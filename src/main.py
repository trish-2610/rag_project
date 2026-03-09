## importing required libraries
from loader import load_files
from splitter import split_documents
from embeddings import EmbeddingModel
from vectorstore import VectorStore
from llm_service import generate_response

## Loading all the pdf files from the directory
docs = load_files("../dataset")

## Splitting the data into chunks
chunks = split_documents(docs)

## Create embeddings using chunks
embedder = EmbeddingModel()
texts = [c.page_content for c in chunks] ## fetching page content from chunks
embeddings = embedder.embed(texts)

## Initializing vectorDB
vectorstore = VectorStore()
vectorstore.add_documents(chunks,embeddings)

## Creating query embeddings 
query = "What is NITI Ayog ?"
query_embeddings = embedder.embed([query])[0]

## retrieving context by querying vectorDB
retrieved_docs = vectorstore.collection.query(
    query_embeddings = query_embeddings,
    n_results = 3 ## top-3 results only
)

## creating context 
context = "\n\n".join(retrieved_docs)

## Calling LLm to answer user query 
result = generate_response(query,context)
print("Response = ",result)
