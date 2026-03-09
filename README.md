# Production-Ready Retrieval Augmented Generation (RAG) System

A modular and production-style **Retrieval Augmented Generation (RAG)** pipeline designed to answer questions using external documents while minimizing hallucinations and ensuring reliable responses.

This project focuses on **building a clean, scalable and industry-structured RAG system**, rather than just demonstrating the concept. It includes modular components, optimized chunking strategies, vector search, and prompt-controlled LLM responses.

---

## Objective

The main objective of this project is to build a **reliable and production-ready RAG pipeline** that:

- Retrieves relevant information from a document corpus
- Grounds LLM responses strictly in retrieved context
- Reduces hallucinations
- Maintains a **clean, modular architecture used in real-world AI systems**

Instead of a simple notebook-based prototype, the focus is on **structured engineering practices used in production AI applications**.

---

## Key Features

### Modular Architecture
The project is structured into independent components:

- Document loader
- Text chunking pipeline
- Embedding generator
- Vector database manager
- Retrieval pipeline
- LLM response generator

Each module is isolated inside the `src/` directory for **maintainability and scalability**.

---

### Persistent Vector Database
The system stores embeddings in a **persistent ChromaDB vector database**.

Benefits:
- Faster retrieval
- No recomputation of embeddings
- Scalable storage of large document collections

The vector database lives inside the source architecture, making the project **self-contained and reproducible**.

---

### Optimized Document Chunking
Documents are split using **Recursive Character Text Splitting** to preserve semantic meaning.

This prevents:
- context fragmentation  
- overly long chunks that degrade retrieval quality

---

### Embedding-Based Semantic Retrieval
User queries are converted into embeddings and matched with document chunks using **cosine similarity search**.

This allows the system to retrieve **semantically relevant information**, not just keyword matches.

---

### Hallucination Prevention
A major focus of the system is **reducing hallucinated answers**.

This is handled through multiple safeguards:

- The LLM receives **only retrieved document context**
- The prompt instructs the model to **answer strictly from the provided context**
- If the information is missing, the model is guided to respond with **“I don't know” instead of generating false information**

This ensures responses remain **grounded in real data**.

---

### Metadata Tracking
Each document chunk stores metadata such as:

- source file
- chunk id

This allows tracing answers back to the **original document source**, which is important for real-world systems.

---

### Edge Case Handling
The pipeline includes defensive checks such as:

- validating chunk and embedding length consistency
- safe document loading
- structured data handling before vector storage

These checks make the system **more reliable for production workflows**.

---

### Experimentation Notebook
The project includes a notebook for testing:

- different chunk sizes
- retrieval quality
- embedding behavior

This allows experimentation without modifying the main pipeline.

---

## Project Structure



