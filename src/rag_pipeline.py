## importing required libraries
from loader import load_files
from splitter import split_documents
from embeddings import EmbeddingModel
from vectorstore import VectorStore
from llm_service import generate_response
import os 

VECTOR_DB_PATH = "../vectorDB"

## Initialize embedder 
embedder = EmbeddingModel()

## Initialize vectorstore
vectorstore = VectorStore()

## Data Ingestion pipeline - only works if vectorDB doesn't exist already
if not os.path.exists(VECTOR_DB_PATH) or len(os.listdir(VECTOR_DB_PATH)) == 0:
    print("Creating vectorDB")

    ## Loading all the pdf files from the directory
    docs = load_files("../dataset")

    ## Splitting the data into chunks
    chunks = split_documents(docs)

    ## Create embeddings using chunks
    texts = [c.page_content for c in chunks] ## fetching page content from chunks
    embeddings = embedder.embed(texts)

    ## adding chunks to vector-store 
    vectorstore.add_documents(chunks,embeddings)
else:
    print("vectorDB already exits so skipping Ingestion")

## user query 
query = "What is NITI Ayog ?"

## Creating query embeddings 
query_embeddings = embedder.embed([query])

## retrieving context by querying vectorDB ( top documents )
retrieved_docs = vectorstore.collection.query(
    query_embeddings = query_embeddings.tolist(),
    n_results = 5 ## top-5 results only
)

## creating context from retrieved docs 
context = "\n\n---\n\n".join(retrieved_docs["documents"][0])

## Calling LLm to answer user query 
result = generate_response(query,context)
print("\nResponse = \n",result)
