import gradio as gr
import PyPDF2
import io
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from groq import Groq
import os
from typing import List, Tuple, Dict
import json

# Initialize Groq client
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
client = Groq(api_key=GROQ_API_KEY)

# Initialize sentence transformer model for embeddings
print("Loading sentence transformer model...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Global variables to store document chunks and metadata
document_chunks = []
document_metadata = []  # Store (filename, page_number) for each chunk
chunk_embeddings = None  # Store embeddings for all chunks
chat_history = []  # Store conversation history

def extract_text_from_pdf(pdf_file) -> List[Tuple[str, str, int]]:
    """
    Extract text from PDF file.
    Returns list of tuples: (text, filename, page_number)
    """
    if pdf_file is None:
        return []
    
    text_pages = []
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        filename = os.path.basename(pdf_file.name)
        
        for page_num, page in enumerate(pdf_reader.pages, start=1):
            text = page.extract_text()
            if text.strip():  # Only add non-empty pages
                text_pages.append((text, filename, page_num))
        
        return text_pages
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return []

def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 100) -> List[str]:
    """
    Split text into semantic chunks with overlap.
    """
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - chunk_overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    
    return chunks

def process_pdfs(pdf_files) -> str:
    """
    Process uploaded PDF files: extract text, chunk, and create embeddings.
    """
    global document_chunks, document_metadata
    
    if pdf_files is None or len(pdf_files) == 0:
        return "Please upload at least one PDF file."
    
    document_chunks = []
    document_metadata = []
    all_text_pages = []
    
    # Extract text from all PDFs
    for pdf_file in pdf_files:
        pages = extract_text_from_pdf(pdf_file)
        all_text_pages.extend(pages)
    
    if not all_text_pages:
        return "No text could be extracted from the uploaded PDFs."
    
    # Chunk the text
    for text, filename, page_num in all_text_pages:
        chunks = chunk_text(text)
        for chunk in chunks:
            document_chunks.append(chunk)
            document_metadata.append((filename, page_num))
    
    # Create embeddings for all chunks
    print(f"Creating embeddings for {len(document_chunks)} chunks...")
    
    # Store embeddings in global variable
    global chunk_embeddings
    chunk_embeddings = embedding_model.encode(document_chunks)
    
    total_chunks = len(document_chunks)
    total_pages = len(set((f, p) for f, p in document_metadata))
    filenames = set(f for f, _ in document_metadata)
    
    return f"✅ Successfully processed {len(filenames)} PDF file(s)!\n\n" \
           f"📄 Total pages: {total_pages}\n" \
           f"📝 Total chunks: {total_chunks}\n" \
           f"📚 Files: {', '.join(filenames)}"

def retrieve_relevant_chunks(query: str, top_k: int = 3) -> List[Tuple[str, str, int, float]]:
    """
    Retrieve top-k most relevant chunks using cosine similarity.
    Returns list of tuples: (chunk_text, filename, page_number, similarity_score)
    """
    global chunk_embeddings
    if len(document_chunks) == 0 or chunk_embeddings is None:
        return []
    
    # Create query embedding
    query_embedding = embedding_model.encode([query])
    
    # Calculate cosine similarity
    similarities = cosine_similarity(query_embedding, chunk_embeddings)[0]
    
    # Get top-k indices
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    # Retrieve relevant chunks with metadata
    relevant_chunks = []
    for idx in top_indices:
        chunk_text = document_chunks[idx]
        filename, page_num = document_metadata[idx]
        similarity_score = similarities[idx]
        relevant_chunks.append((chunk_text, filename, page_num, similarity_score))
    
    return relevant_chunks

def generate_answer(query: str, history: List) -> Tuple[str, List]:
    """
    Generate answer using RAG pipeline with Groq LLM.
    """
    global chat_history
    
    if not document_chunks:
        error_msg = "⚠️ Please upload and process PDF files first!"
        updated_history = history + [(query, error_msg)]
        return error_msg, updated_history
    
    if not GROQ_API_KEY:
        error_msg = "⚠️ Please set your GROQ_API_KEY environment variable!"
        updated_history = history + [(query, error_msg)]
        return error_msg, updated_history
    
    # Retrieve relevant chunks
    relevant_chunks = retrieve_relevant_chunks(query, top_k=3)
    
    if not relevant_chunks:
        error_msg = "⚠️ No relevant information found in the documents."
        updated_history = history + [(query, error_msg)]
        return error_msg, updated_history
    
    # Build context from relevant chunks
    context_parts = []
    sources = []
    
    for chunk_text, filename, page_num, score in relevant_chunks:
        context_parts.append(f"[Source: {filename}, Page {page_num}]\n{chunk_text}")
        sources.append(f"{filename} (Page {page_num})")
    
    context = "\n\n".join(context_parts)
    
    # Build prompt with conversation history
    system_prompt = """You are a helpful assistant that answers questions based on the provided context from PDF documents. 
    Always cite your sources when answering. If the context doesn't contain enough information, say so."""
    
    # Include recent conversation history (last 3 exchanges)
    history_context = ""
    if history:
        recent_history = history[-3:]  # Last 3 exchanges
        history_context = "\n\nPrevious conversation:\n"
        for h in recent_history:
            # Handle both tuple and list formats
            if isinstance(h, (tuple, list)) and len(h) == 2:
                history_context += f"User: {h[0]}\nAssistant: {h[1]}\n"
    
    user_prompt = f"""Context from documents:
{context}

{history_context}

Question: {query}

Please provide a comprehensive answer based on the context above. Include source references (filename and page number) in your answer."""
    
    try:
        # Call Groq API
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",  # Updated from deprecated llama3-8b-8192
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            max_tokens=1024
        )
        
        answer = response.choices[0].message.content
        
        # Add sources to answer
        sources_text = "\n\n📚 Sources: " + ", ".join(set(sources))
        final_answer = answer + sources_text
        
        # Update history with new message pair
        updated_history = history + [(query, final_answer)]
        chat_history = updated_history
        
        # Return both answer and updated history (in tuple format for internal use)
        return final_answer, updated_history
    
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error in generate_answer: {error_details}")  # Print to console for debugging
        error_msg = f"⚠️ Error generating answer: {str(e)}"
        updated_history = history + [(query, error_msg)]
        return error_msg, updated_history

def clear_chat():
    """Clear chat history."""
    global chat_history
    chat_history = []
    return []

def download_chat_history():
    """Download chat history as JSON."""
    if not chat_history:
        return None
    
    history_dict = {
        "conversation": [
            {"user": h[0], "assistant": h[1]} for h in chat_history if len(h) == 2
        ]
    }
    
    # Save to temporary file
    import tempfile
    temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json')
    json.dump(history_dict, temp_file, indent=2)
    temp_file.close()
    
    return temp_file.name

# Create Gradio interface
with gr.Blocks(title="RAG Chatbot") as demo:
    gr.Markdown(
        """
        # 📚 RAG-Based Chatbot
        
        Upload multiple PDF files and ask questions based on their content!
        
        **Features:**
        - 📄 Multi-PDF support
        - 🔍 Semantic search with sentence-transformers
        - 💬 Conversational memory
        - 📑 Source references with page numbers
        - 📥 Download chat history
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📤 Upload PDFs")
            pdf_upload = gr.File(
                file_count="multiple",
                file_types=[".pdf"],
                label="Upload PDF Files"
            )
            process_btn = gr.Button("Process PDFs", variant="primary")
            status_output = gr.Textbox(
                label="Status",
                lines=5,
                interactive=False
            )
        
        with gr.Column(scale=2):
            gr.Markdown("### 💬 Chat")
            chatbot = gr.Chatbot(
                label="Conversation",
                height=400,
                container=True
            )
            query_input = gr.Textbox(
                label="Ask a question",
                placeholder="Enter your question here...",
                lines=1
            )
            
            with gr.Row():
                submit_btn = gr.Button("Send", variant="primary")
                clear_btn = gr.Button("Clear Chat")
                download_btn = gr.DownloadButton("Download History", variant="secondary")
    
    # Event handlers
    process_btn.click(
        fn=process_pdfs,
        inputs=[pdf_upload],
        outputs=[status_output]
    )
    
    def respond(message, history):
        # Handle empty message
        if not message or not message.strip():
            return history if history else []
        
        # Initialize history
        if history is None:
            history = []
        
        # Convert history to tuple format for internal processing
        # Gradio 6.1.0 sends/receives dicts, but we use tuples internally
        tuple_history = []
        for item in history:
            if isinstance(item, dict) and "role" in item and "content" in item:
                if item["role"] == "user":
                    tuple_history.append((item["content"], ""))
                elif item["role"] == "assistant" and tuple_history:
                    tuple_history[-1] = (tuple_history[-1][0], item["content"])
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                tuple_history.append((str(item[0]), str(item[1])))
        
        # Generate answer using tuple format internally
        try:
            answer, _ = generate_answer(str(message), tuple_history)
            if not answer:
                answer = "⚠️ No response generated."
            
            # Return in Gradio 6.1.0 format: list of dicts with role and content
            result = list(history) + [
                {"role": "user", "content": str(message)},
                {"role": "assistant", "content": str(answer)}
            ]
            return result
            
        except Exception as e:
            import traceback
            print(f"ERROR in respond:\n{traceback.format_exc()}")
            error_msg = f"⚠️ Error: {str(e)}"
            result = list(history) + [
                {"role": "user", "content": str(message)},
                {"role": "assistant", "content": error_msg}
            ]
            return result
    
    def clear_input():
        return ""
    
    # Make Enter key work (not just Shift+Enter)
    query_input.submit(
        fn=respond,
        inputs=[query_input, chatbot],
        outputs=[chatbot],
        show_progress=False
    ).then(
        fn=clear_input,
        outputs=[query_input]
    )
    
    submit_btn.click(
        fn=respond,
        inputs=[query_input, chatbot],
        outputs=[chatbot]
    ).then(
        fn=clear_input,
        outputs=[query_input]
    )
    
    clear_btn.click(
        fn=clear_chat,
        outputs=[chatbot]
    )
    
    download_btn.click(
        fn=download_chat_history,
        outputs=[download_btn]
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, theme=gr.themes.Soft(), share=True)

