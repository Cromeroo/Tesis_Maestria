import torch
import torch.nn as nn
from torchvision import transforms as T
from PIL import Image
import streamlit as st
import os
from consulta_rag import ConsultaRAG
import time

# --------------------- Arquitectura CNN ---------------------
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

# --------------------- Clase Agente Multimodal ---------------------
class AgenteMultimodal:
    def __init__(self):
        """Inicializa el agente multimodal"""
        self.model = self.cargar_modelo()
        self.rag_system = ConsultaRAG()
        self.transform = T.Compose([
            T.Resize((128, 128)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.idx_to_label = {0: "Sana", 1: "Tizon_tardio", 2: "Otras_enfermedades"}
        
        print("🤖 Agente multimodal inicializado")
    
    @st.cache_resource
    def cargar_modelo(self, path="best_model_3class.pth"):
        """Carga el modelo CNN entrenado"""
        model = SimpleCNN()
        model.load_state_dict(torch.load(path, map_location="cpu"))
        model.eval()
        return model
    
    def clasificar_imagen(self, imagen):
        """
        Clasifica una imagen usando el modelo CNN
        
        Args:
            imagen: Imagen PIL
            
        Returns:
            Tupla (clase_predicha, confianza)
        """
        # Preprocesar imagen
        tensor = self.transform(imagen).unsqueeze(0)
        
        # Predicción
        with torch.no_grad():
            salida = self.model(tensor)
            probabilidades = torch.softmax(salida, dim=1)
            pred = salida.argmax(1).item()
            confianza = probabilidades[0][pred].item()
            clase = self.idx_to_label[pred]
        
        return clase, confianza
    
    def obtener_recomendaciones(self, clase_diagnostico, confianza):
        """
        Obtiene recomendaciones basadas en el diagnóstico
        
        Args:
            clase_diagnostico: Clase predicha por el modelo
            confianza: Nivel de confianza de la predicción
            
        Returns:
            Recomendaciones del sistema RAG
        """
        if clase_diagnostico == "Tizon_tardio":
            consultas = [
                "¿Cuáles son los síntomas específicos del tizón tardío en tomate?",
                "¿Qué tratamientos químicos y biológicos son efectivos contra el tizón tardío?",
                "¿Cómo se puede prevenir la propagación del tizón tardío?",
                "¿Cuáles son las mejores prácticas de manejo para el tizón tardío?"
            ]
        elif clase_diagnostico == "Otras_enfermedades":
            consultas = [
                "¿Cuáles son las enfermedades más comunes del tomate?",
                "¿Cómo diferenciar el tizón tardío de otras enfermedades del tomate?",
                "¿Qué medidas generales de control de enfermedades se recomiendan?"
            ]
        else:  # Sana
            consultas = [
                "¿Cómo mantener las plantas de tomate saludables?",
                "¿Cuáles son las mejores prácticas de prevención de enfermedades?",
                "¿Qué medidas preventivas se recomiendan para evitar el tizón tardío?"
            ]
        
        recomendaciones = []
        for consulta in consultas:
            try:
                respuesta = self.rag_system.consultar(consulta, n_results=3)
                recomendaciones.append({
                    "pregunta": consulta,
                    "respuesta": respuesta
                })
                time.sleep(0.5)  # Pausa para evitar rate limits
            except Exception as e:
                st.error(f"Error al obtener recomendación: {e}")
        
        return recomendaciones
    
    def generar_diagnostico_completo(self, imagen):
        """
        Genera un diagnóstico completo con recomendaciones
        
        Args:
            imagen: Imagen PIL del usuario
            
        Returns:
            Diccionario con diagnóstico y recomendaciones
        """
        # Clasificación visual
        clase, confianza = self.clasificar_imagen(imagen)
        
        # Obtener recomendaciones
        recomendaciones = self.obtener_recomendaciones(clase, confianza)
        
        return {
            "diagnostico": {
                "clase": clase,
                "confianza": confianza,
                "descripcion": self.obtener_descripcion_diagnostico(clase, confianza)
            },
            "recomendaciones": recomendaciones
        }
    
    def obtener_descripcion_diagnostico(self, clase, confianza):
        """Genera una descripción del diagnóstico"""
        if confianza < 0.7:
            nivel_confianza = "baja"
        elif confianza < 0.9:
            nivel_confianza = "moderada"
        else:
            nivel_confianza = "alta"
        
        descripciones = {
            "Sana": f"La hoja aparece saludable con confianza {nivel_confianza} ({confianza:.1%})",
            "Tizon_tardio": f"Se detectan síntomas compatibles con tizón tardío con confianza {nivel_confianza} ({confianza:.1%})",
            "Otras_enfermedades": f"Se detectan síntomas de enfermedad, pero no específicamente tizón tardío, con confianza {nivel_confianza} ({confianza:.1%})"
        }
        
        return descripciones.get(clase, "Diagnóstico no disponible")

# --------------------- Interfaz Streamlit ---------------------
def main():
    st.set_page_config(
        page_title="Agente Multimodal - Tizón Tardío",
        page_icon="🌿",
        layout="wide"
    )
    
    st.title("🤖 Agente Multimodal para Diagnóstico de Tizón Tardío")
    st.markdown("---")
    
    # Inicializar agente
    if 'agente' not in st.session_state:
        with st.spinner("Inicializando agente multimodal..."):
            st.session_state.agente = AgenteMultimodal()
    
    # Sidebar para información
    with st.sidebar:
        st.header("ℹ️ Información")
        st.markdown("""
        Este agente combina:
        - **Visión por computadora**: Clasificación automática de imágenes
        - **RAG (Retrieval Augmented Generation)**: Base de conocimiento especializada
        
        **Clases detectadas:**
        - 🟢 Sana
        - 🔴 Tizón Tardío
        - 🟡 Otras Enfermedades
        """)
        
        st.markdown("---")
        st.markdown("**Desarrollado para:** Tesis sobre tizón tardío en tomate")
    
    # Área principal
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Subir Imagen")
        archivo = st.file_uploader(
            "Selecciona una imagen de hoja de tomate",
            type=["jpg", "jpeg", "png"],
            help="Sube una imagen clara de la hoja para análisis"
        )
        
        if archivo:
            imagen = Image.open(archivo).convert("RGB")
            st.image(imagen, caption="Imagen cargada", use_column_width=True)
            
            # Botón de análisis
            if st.button("🔍 Analizar Imagen", type="primary"):
                with st.spinner("Analizando imagen..."):
                    diagnostico = st.session_state.agente.generar_diagnostico_completo(imagen)
                    st.session_state.diagnostico = diagnostico
    
    with col2:
        st.header("📊 Resultados del Análisis")
        
        if 'diagnostico' in st.session_state:
            diagnostico = st.session_state.diagnostico
            
            # Mostrar diagnóstico
            st.subheader("🎯 Diagnóstico")
            clase = diagnostico["diagnostico"]["clase"]
            confianza = diagnostico["diagnostico"]["confianza"]
            descripcion = diagnostico["diagnostico"]["descripcion"]
            
            # Color según clase
            if clase == "Sana":
                st.success(f"🟢 {descripcion}")
            elif clase == "Tizon_tardio":
                st.error(f"🔴 {descripcion}")
            else:
                st.warning(f"🟡 {descripcion}")
            
            # Barra de confianza
            st.progress(confianza)
            st.caption(f"Confianza: {confianza:.1%}")
            
            # Recomendaciones
            st.subheader("💡 Recomendaciones")
            recomendaciones = diagnostico["recomendaciones"]
            
            for i, rec in enumerate(recomendaciones):
                with st.expander(f"📋 {rec['pregunta']}", expanded=(i==0)):
                    st.markdown(rec['respuesta'])
        else:
            st.info("👆 Sube una imagen y haz clic en 'Analizar Imagen' para ver los resultados")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
        🤖 Agente Multimodal - Tizón Tardío en Tomate<br>
        Combina visión por computadora y base de conocimiento especializada
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main() 