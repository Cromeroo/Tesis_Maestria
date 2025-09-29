# 🔒 Configuración Segura de Credenciales

## ✅ SOLUCIÓN IMPLEMENTADA: credentials.json

### 🎯 Ventajas de esta aproximación:

- ✅ **GitHub seguro**: No hay tokens hardcodeados en el código
- ✅ **Fácil de usar**: Solo ejecutar `streamlit run web_langgraph_demo.py`
- ✅ **Centralizado**: Todas las credenciales en un solo archivo
- ✅ **Protegido**: `credentials.json` está en `.gitignore`

### 📁 Estructura del archivo credentials.json:

```json
{
  "type": "service_account",
  "project_id": "your-project-id",
  // ... otras credenciales Google Cloud ...
  "huggingface_token": "hf_YOUR_HUGGINGFACE_TOKEN_HERE"
}
```

### 🚀 Como usar:

```bash
# Solo esto y funciona automáticamente:
streamlit run web_langgraph_demo.py --server.port 8535
```

### 🛡️ Seguridad:

- `credentials.json` nunca se sube a GitHub (está en .gitignore)
- Los tokens se cargan dinámicamente al iniciar la app
- Fallback a variables de entorno si no existe el archivo

### ✅ Logs de confirmación:

- `✅ Token HF cargado desde credentials.json`
- `[LOG] Token cargado desde credentials.json`
- `[LOG] ✅ Modelo cargado correctamente desde HuggingFace`

¡Perfecto para desarrollo local y seguro para GitHub! 🎉
