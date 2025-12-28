---
title: RAG-Based Chatbot
emoji: 📚
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 6.1.0
app_file: app.py
pinned: false
license: mit
---

# RAG-Based Chatbot with PDF Support

A Retrieval-Augmented Generation (RAG) chatbot that accepts multiple PDF documents and answers questions based on their content using Groq LLM and Gradio UI.

## Features

### Base Requirements ✅
- ✅ Upload multiple PDF files via Gradio
- ✅ Extract text from all pages
- ✅ Split content into semantic chunks
- ✅ Retrieve top relevant chunks using vector similarity (sentence-transformers)
- ✅ Send question + context to Groq LLM (llama-3.1-8b-instant)
- ✅ Display answer on Gradio interface

### Enhancements Implemented 🚀
1. **Sentence-Transformers for Embeddings** - Uses `all-MiniLM-L6-v2` model for semantic embeddings instead of TF-IDF
2. **Source References with Page Numbers** - Answers include citations showing which PDF and page number the information came from
3. **Conversational Memory/History** - Maintains conversation context across multiple questions
4. **Download Chat History** - Export conversation history as JSON file

## How RAG Works

RAG (Retrieval-Augmented Generation) combines information retrieval with language generation:

1. **Document Processing**: PDFs are uploaded and text is extracted from all pages
2. **Chunking**: Text is split into semantic chunks with overlap for better context
3. **Embedding**: Each chunk is converted to a vector using sentence-transformers
4. **Retrieval**: When a question is asked, relevant chunks are retrieved using cosine similarity
5. **Generation**: The question + relevant context is sent to the Groq LLM to generate an answer
6. **Response**: The answer is displayed with source references

## Setup Instructions

### Local Development

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up Groq API Key:**
   You can use your existing Groq API key from console.groq.com. Set it as an environment variable:
   ```bash
   export GROQ_API_KEY="your-api-key-here"
   ```
   Or on Windows:
   ```powershell
   $env:GROQ_API_KEY="your-api-key-here"
   ```
   
   **Note:** The same Groq API key you use for other chatbots will work here. No need for a separate key.

3. **Run the application:**
   ```bash
   python app.py
   ```

4. **Access the interface:**
   Open your browser and go to `http://localhost:7860`

## Usage

1. **Upload PDFs**: Click "Upload PDF Files" and select one or more PDF documents
2. **Process Documents**: Click "Process PDFs" to extract and index the content
3. **Ask Questions**: Type your question in the chat input and click "Send"
4. **View Sources**: Each answer includes references to the source PDF and page number
5. **Download History**: Click "Download History" to save the conversation as JSON

## Project Structure

```
rag/
├── app.py              # Main application file
├── requirements.txt    # Python dependencies
├── apt.txt            # System dependencies (if needed)
└── README.md          # This file
```

## Technical Details

- **Embedding Model**: `all-MiniLM-L6-v2` (sentence-transformers)
- **LLM**: Groq `llama-3.1-8b-instant` (updated from deprecated llama3-8b-8192)
- **Chunking**: Semantic chunking with 500-word chunks and 100-word overlap
- **Retrieval**: Top-3 most relevant chunks using cosine similarity
- **UI Framework**: Gradio 6.1.0+
