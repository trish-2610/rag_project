## importing required libraries
from loader import load_files
from splitter import split_documents
from embeddings import EmbeddingModel
from vectorstore import VectorStore
from llm_service import generate_response
import os


## set vectorDB path 
VECTOR_DB_PATH = "vectorDB"

## initialization 
embedder = EmbeddingModel()
vectorstore = VectorStore()


def ingest_documents():
    """Create vector database if it does not exist"""

    if not os.path.exists(VECTOR_DB_PATH) or len(os.listdir(VECTOR_DB_PATH)) == 0:

        print("Creating vectorDB")

        ## load all the files 
        docs = load_files("../dataset")
        
        ## Split document into chunks 
        chunks = split_documents(docs)
        
        ## fetch page content from chunks
        texts = [c.page_content for c in chunks]

        ## create embeddings 
        embeddings = embedder.embed(texts)
        
        ## adds chunks to vectorDB
        vectorstore.add_documents(chunks, embeddings)

    else:
        print("vectorDB already exists so skipping ingestion")


def retrieve_documents(query):
    """Retrieve relevant documents from vectorDB"""

    query_embedding = embedder.embed([query])

    results = vectorstore.collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=5
    )

    docs = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    filtered_docs = []
    sources = []

    similarity_threshold = 1.2

    for doc, meta, dist in zip(docs , metadatas , distances):

        if dist < similarity_threshold:
            filtered_docs.append(doc)
            source = f"{meta['source']} (page {meta['page']})"
            sources.append(source)

    return filtered_docs, sources


def build_context(filtered_docs):
    """Create context for the LLM"""

    if len(filtered_docs) == 0:
        return None

    return "\n\n---\n\n".join(filtered_docs)


def answer_query(query):
    """Full RAG pipeline for answering a question"""

    filtered_docs, sources = retrieve_documents(query)

    context = build_context(filtered_docs)

    if context is None:
        print("No relevant documents found.")
        return

    answer = generate_response(query, context)
    
    ## prints answer
    print("\nAnswer:\n")
    print(answer)

    ## prints response
    print("\nSources:\n")
    for s in set(sources):
        print("-", s)


def main():
    ingest_documents()
    query = input("Enter your query :")
    answer_query(query)

if __name__ == "__main__":
    main()