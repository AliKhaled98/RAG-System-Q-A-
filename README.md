# RAG-System-Q-A-
PDF Q&A System – RAG with LangChain, LLaMA3, and Flask  
This project is a Retrieval-Augmented Generation (RAG) application that lets you upload a PDF, ask questions about it, and get accurate AI-generated answers — all with the context pulled directly from the document.

Chroma DB and the rest of files in drive: https://drive.google.com/drive/u/3/folders/1P9sRY-Y6E9sGleqkoTPR1NeRy4sGdux7  


It’s perfect for research papers, reports, books, technical manuals, and any long-form document where searching manually would be slow.

🚀 What It Can Do
Upload and process PDFs (supports large documents)

Automatically clean & split the document into manageable chunks

Generate embeddings for each chunk using FastEmbed

Store and search document embeddings in ChromaDB (persistent vector store)

Retrieve the most relevant context for your question

Answer questions using LLaMA3 via Ollama (local inference, no API cost)

Simple Flask Web Interface so you don’t need to run Python commands each time

🛠 Tech Stack
LangChain – For document loading, chunking, retrieval pipeline

FastEmbed – For lightweight and fast embeddings

ChromaDB – For vector storage & similarity search

Ollama – To run LLaMA3 locally

Flask – To create the user-friendly web app


