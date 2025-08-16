#!/usr/bin/env python3
"""
Local PDF-aware chatbot with Gradio web UI (Ollama + FAISS)
"""
import os
import faiss
import numpy as np
import requests
import gradio as gr
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# === Settings ===
EMBED_MODEL = "sentence-transformers/paraphrase-MiniLM-L3-v2"  # faster loading
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
TOP_K = 3
INDEX_FILE = "vectors.faiss"
META_FILE = "meta.npy"
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_MODEL = "gemma:2b"

# === Global state ===
embedder = None         # Lazy load
documents = []
index = None
chat_history = []


# --------- Utilities ----------
def split_text(text):
    chunks, i = [], 0
    while i < len(text):
        chunk = text[i:i+CHUNK_SIZE]
        chunks.append(chunk)
        i += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def add_pdf_to_index(pdf_path):
    global index, documents, embedder
    if embedder is None:  # Lazy load here
        embedder = SentenceTransformer(EMBED_MODEL)

    reader = PdfReader(pdf_path)
    text_chunks = []
    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        for piece in split_text(text):
            if piece.strip():
                tag = f"[{os.path.basename(pdf_path)}:p{page_num}]"
                text_chunks.append((piece.strip(), tag))

    # Create or extend FAISS index
    embeddings = embedder.encode([t[0] for t in text_chunks], normalize_embeddings=True)
    embeddings = np.asarray(embeddings, dtype="float32")

    if index is None:
        dim = embeddings.shape[1]
        idx = faiss.IndexFlatIP(dim)
        idx.add(embeddings)
    else:
        idx = index
        idx.add(embeddings)

    index = idx
    documents.extend(text_chunks)
    return idx


def save_index():
    if index:
        faiss.write_index(index, INDEX_FILE)
        np.save(META_FILE, np.array(documents, dtype=object))


def load_index():
    global index, documents, embedder
    if os.path.exists(INDEX_FILE) and os.path.exists(META_FILE):
        if embedder is None:  # Ensure embedder is loaded before searching
            embedder = SentenceTransformer(EMBED_MODEL)
        index = faiss.read_index(INDEX_FILE)
        documents = np.load(META_FILE, allow_pickle=True).tolist()
        return True
    return False


def retrieve(query, k=TOP_K):
    q_emb = embedder.encode([query], normalize_embeddings=True)
    q_emb = np.asarray(q_emb, dtype="float32")
    scores, idxs = index.search(q_emb, k)
    results = []
    for i, score in zip(idxs[0], scores[0]):
        if i != -1:
            results.append((documents[i], float(score)))
    return results


def ollama_chat(context, question, model=DEFAULT_MODEL):
    system_prompt = "You are a private assistant. Use context to answer, cite sources, and keep memory of the conversation."
    # include memory
    messages = [{"role": "system", "content": system_prompt}] + chat_history
    messages.append({"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"})

    try:
        r = requests.post(
            f"{OLLAMA_HOST}/api/chat",
            json={"model": model, "messages": messages, "stream": False},
            timeout=120
        )
        r.raise_for_status()
        data = r.json()
        reply = data.get("message", {}).get("content", "No response from model.")
        # update memory
        chat_history.append({"role": "user", "content": question})
        chat_history.append({"role": "assistant", "content": reply})
        return reply
    except Exception as e:
        return f"❌ Ollama Error: {str(e)}"


# --------- Core Functions ----------
def upload_and_index(files):
    global index
    if files is None:
        return "No file uploaded"
    if not isinstance(files, list):
        files = [files]

    if index is None and not load_index():
        index = None  # will rebuild

    for f in files:
        index = add_pdf_to_index(f.name)
    save_index()
    return f"Indexed {len(documents)} chunks from {len(files)} PDF(s)."


def chat_with_bot(question, model):
    if not index:
        return [("Assistant", "❌ Please upload and index PDFs first.")]

    retrieved = retrieve(question)
    if not retrieved:
        return [("Assistant", "No relevant text found in PDFs.")]

    context = "\n\n".join([f"{tag}\n{text}" for text, tag in [r[0] for r in retrieved]])
    answer = ollama_chat(context, question, model=model)
    sources = "Sources:\n" + "\n".join([f"{r[0][1]} (score {r[1]:.2f})" for r in retrieved])
    return [("Assistant", f"{answer}\n\n---\n{sources}")]


def summarize_pdfs():
    if not index:
        return "❌ Please upload and index PDFs first."

    all_texts = [doc[0] for doc in documents]
    context = "\n".join(all_texts[:50])  # limit for speed
    return ollama_chat(context, "Summarize this PDF collection in simple points.", model=DEFAULT_MODEL)


# --------- Gradio UI ----------
with gr.Blocks() as demo:
    gr.Markdown("# 📚 Enhanced Local PDF Chatbot (Lazy Load + Ollama + FAISS)")

    with gr.Row():
        pdf_input = gr.File(file_types=[".pdf"], file_count="multiple", label="Upload PDFs")
        upload_btn = gr.Button("Index PDFs")
        status = gr.Textbox(label="Status")

    chatbot_ui = gr.Chatbot(label="Chat with your PDFs", height=400)
    question = gr.Textbox(label="Your Question")
    model = gr.Textbox(value=DEFAULT_MODEL, label="Model", interactive=True)
    ask_btn = gr.Button("Ask")

    summarize_btn = gr.Button("Summarize PDFs")
    summary_output = gr.Textbox(label="Summary", lines=6)

    upload_btn.click(upload_and_index, inputs=pdf_input, outputs=status)
    ask_btn.click(chat_with_bot, inputs=[question, model], outputs=chatbot_ui)
    question.submit(chat_with_bot, inputs=[question, model], outputs=chatbot_ui)
    summarize_btn.click(summarize_pdfs, outputs=summary_output)


if __name__ == "__main__":
    # No load_index() here → faster startup
    demo.launch()
