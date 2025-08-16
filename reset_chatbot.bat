@echo off
cd /d "C:\Users\sunil\OneDrive\Desktop\pdf-chat"

echo 🔄 Resetting chatbot memory and index...

:: Delete FAISS index and metadata if they exist
if exist vectors.faiss del vectors.faiss
if exist meta.npy del meta.npy

:: Activate venv
call .venv\Scripts\activate.bat

:: Run chatbot fresh
python local_rag_chatbot.py

pause
