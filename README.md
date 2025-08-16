# 📚 PDF Chatbot (Local RAG with Ollama)

A local **PDF Question-Answering Chatbot** built with:
- [Python](https://www.python.org/)
- [Gradio](https://gradio.app/) for the web interface
- [FAISS](https://github.com/facebookresearch/faiss) for vector search
- [Sentence Transformers](https://www.sbert.net/) for embeddings
- [Ollama](https://ollama.ai/) for running local LLMs (Gemma, TinyLlama, etc.)

---

## ✨ Features
✅ Upload one or more PDFs  
✅ Ask natural questions and get answers from your documents  
✅ Sources are highlighted from your PDFs  
✅ Works **fully offline** (no OpenAI API required)  
✅ Error logging to track issues  
✅ Reset mode to start fresh  

---

## ⚡ Requirements
- Python **3.9+** (tested with 3.13)  
- Git (for version control)  
- [Ollama](https://ollama.ai/) installed with at least one model (recommended: `gemma:2b`)  

---

## 🔧 Installation

1. **Clone this repo**
   ```bash
   git clone https://github.com/samyaksharma0007/pdf-chat.git
   cd pdf-chat
