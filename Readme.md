# RAG Chatbot – Document-Based Question Answering

This project implements a **Retrieval-Augmented Generation (RAG)** chatbot using OpenAI's API, LangChain, and FAISS.  
The system loads a PDF document, generates embeddings, stores them in a vector database, retrieves relevant context, and uses an LLM to answer questions based only on the document content.

## Technologies Used
- OpenAI API (LLM + Embeddings)
- LangChain
- FAISS Vector Database
- PyPDF
- Python 3.11

## How It Works
1. Loads a PDF document.
2. Splits text into chunks.
3. Generates embeddings.
4. Stores embeddings in FAISS.
5. Retrieves relevant context for a query.
6. Sends context to LLM to generate an answer.

## Example Query
**Question:** What is the document about?  
**Answer:** The document describes an IoT monitoring system based on sensors and ESP32 for industrial supervision.

## How to Run
1. Install dependencies:

2. Add your OpenAI API key in a `.env` file:

3. Run:

---

This project demonstrates applied Prompt Engineering and RAG architecture for document-based intelligent assistants.
