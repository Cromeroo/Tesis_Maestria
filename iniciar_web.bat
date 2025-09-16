@echo off
echo 🚀 Iniciando interfaz web del sistema de diagnostico de tomate...
echo.
echo 💡 El sistema se abrira automaticamente en tu navegador
echo 🌐 URL: http://localhost:8501
echo.
echo ⚡ Para detener el servidor: Ctrl + C
echo.
cd /d "%~dp0"
call env\Scripts\activate
streamlit run web_interface.py --server.port 8501 --server.headless false
