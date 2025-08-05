import os
from dotenv import load_dotenv
from langchain_google_vertexai import ChatVertexAI
from langchain.schema import HumanMessage

# Cargar variables de entorno
load_dotenv()

# Configurar credenciales
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"C:\Users\danil\OneDrive\Escritorio\Tesis\credentials.json"
try:
    # Inicializar el modelo
    llm = ChatVertexAI(
        model_name="gemini-2.5-flash",
        temperature=0.7,
        max_output_tokens=1024,
        location="us-central1"  # Ajusta según tu región
    )
    
    # Prueba simple
    print("🔄 Probando conexión...")
    response = llm.invoke("Hola, ¿cómo estás? Responde en español.")
    print("✅ Conexión exitosa!")
    print(f"📝 Respuesta: {response.content}")
    
    # Prueba más compleja
    print("\n🔄 Probando consulta técnica...")
    query = "Explica brevemente qué es LangGraph y cómo se relaciona con LangChain"
    response = llm.invoke(query)
    print("✅ Consulta técnica exitosa!")
    print(f"📝 Respuesta: {response.content}")
    
except Exception as e:
    print(f"❌ Error de conexión: {e}")
    print("💡 Verifica:")
    print("   - Que las credenciales estén configuradas correctamente")
    print("   - Que tengas permisos en Google Cloud")
    print("   - Que la región sea correcta")