@echo off
echo ========================================================
echo  Launching Dynamic RAG Explorer & Telemetry Engine
echo ========================================================
"C:\ProgramData\anaconda3\python.exe" -m streamlit run app.py --server.port 8501 --server.headless false
pause
