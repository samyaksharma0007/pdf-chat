@echo off
cd /d "C:\Users\sunil\OneDrive\Desktop\pdf-chat"

:: Activate venv
call .venv\Scripts\activate.bat

:: Run chatbot
python local_rag_chatbot.py

:: Keep window open after exit
pause
