#!/usr/bin/env python3
"""
Clasificador de Imágenes de Tomate        # Mapeo de clasificaciones a consultas RAG específicas con enfoque de manejo integrado
        self.classification_to_query = {
            "Sana": [
                "prevención manejo integrado tomate control cultural preventivo",
                "buenas prácticas cultivo tomate espaciamiento riego ventilación",
                "programa preventivo control biológico trichoderma bacillus",
                "calendario aplicaciones preventivas fungicidas cobre mancozeb"
            ],
            "Tizon_tardio": [
                "tizón tardío tomate Phytophthora control tratamiento",
                "control cultural eliminación infectados riego espaciamiento",
                "control biológico trichoderma bacillus cobre orgánico",
                "fungicidas mancozeb metalaxil clorotalonil aplicación",
                "manejo integrado combinado cultural biológico químico",
                "phytophthora infestans tomate tratamiento prevención"
            ],
            "Otras_enfermedades": [
                "enfermedades tomate control tratamiento fungicidas",
                "control cultural enfermedades ventilación riego poda",
                "control biológico microorganismos benéficos cobre",
                "diagnóstico enfermedades foliares tomate síntomas"
            ]
        }===============

Clasifica imágenes de tomates descargando el modelo desde Hugging Face
y mapea las clasificaciones a consultas específicas para el RAG.

Clases:
- 0: Sana
- 1: Tizon_tardio (Late Blight)  
- 2: Otras_enfermedades
"""

import os
import torch
import torch.nn as nn
from torchvision import transforms as T
from PIL import Image
from huggingface_hub import hf_hub_download
from pathlib import Path
import numpy as np


class SimpleCNN(nn.Module):
    """Arquitectura CNN simple para clasificación de tomates"""
    
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


class TomatoImageClassifier:
    """Clasificador de imágenes de tomate integrado con Hugging Face"""
    
    def __init__(self, model_repo="DaniloR2011/Tomato_accuracy", device=None):
        self.model_repo = model_repo
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Mapeo de clases
        self.class_names = {
            0: "Sana",
            1: "Tizon_tardio", 
            2: "Otras_enfermedades"
        }
        
        # Mapeo de clasificaciones a consultas RAG
        self.classification_to_query = {
            "Sana": [
                "mantenimiento preventivo tomate",
                "buenas prácticas cultivo tomate saludable",
                "nutrición tomate sano"
            ],
            "Tizon_tardio": [
                "tratamiento tizón tardío tomate Phytophthora infestans",
                "control químico tizón tardío fungicidas",
                "manejo integrado tizón tardío prevención",
                "síntomas tizón tardío identificación"
            ],
            "Otras_enfermedades": [
                "enfermedades comunes tomate tratamiento",
                "diagnóstico enfermedades foliares tomate",
                "control integrado plagas enfermedades tomate",
                "fungicidas bactericidas tomate"
            ]
        }
        
        # Transformaciones para la imagen
        self.transform = T.Compose([
            T.Resize((128, 128)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.model = None
        self.model_path = None
        
    def download_model(self):
        """Descarga el modelo desde Hugging Face"""
        try:
            print(f"🔄 Descargando modelo desde {self.model_repo}...")
            self.model_path = hf_hub_download(
                repo_id=self.model_repo,
                filename="best_model_3class.pth",
                cache_dir="./models_cache"
            )
            print(f"✅ Modelo descargado: {self.model_path}")
            return True
        except Exception as e:
            print(f"❌ Error descargando modelo: {e}")
            return False
    
    def load_model(self):
        """Carga el modelo PyTorch"""
        try:
            # Descargar si no existe
            if not self.model_path or not os.path.exists(self.model_path):
                if not self.download_model():
                    return False
            
            # Crear e instanciar modelo
            self.model = SimpleCNN(num_classes=3)
            
            # Cargar pesos
            state_dict = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()
            
            print(f"✅ Modelo cargado exitosamente en {self.device}")
            return True
            
        except Exception as e:
            print(f"❌ Error cargando modelo: {e}")
            return False
    
    def preprocess_image(self, image_path_or_pil):
        """Preprocesa la imagen para el modelo"""
        try:
            if isinstance(image_path_or_pil, str) or isinstance(image_path_or_pil, Path):
                image = Image.open(image_path_or_pil).convert("RGB")
            elif isinstance(image_path_or_pil, Image.Image):
                image = image_path_or_pil.convert("RGB")
            else:
                raise ValueError("Input debe ser ruta de imagen o PIL.Image")
            
            # Aplicar transformaciones
            tensor = self.transform(image).unsqueeze(0)  # Agregar batch dimension
            return tensor.to(self.device)
            
        except Exception as e:
            print(f"❌ Error preprocesando imagen: {e}")
            return None
    
    def classify_image(self, image_path_or_pil):
        """
        Clasifica una imagen y retorna la predicción
        
        Args:
            image_path_or_pil: Ruta a imagen o objeto PIL.Image
            
        Returns:
            dict con clasificación, confianza y consultas RAG sugeridas
        """
        # Cargar modelo si no está cargado
        if self.model is None:
            if not self.load_model():
                return {"error": "No se pudo cargar el modelo"}
        
        # Preprocesar imagen
        input_tensor = self.preprocess_image(image_path_or_pil)
        if input_tensor is None:
            return {"error": "No se pudo procesar la imagen"}
        
        try:
            with torch.no_grad():
                # Predicción
                outputs = self.model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_class].item()
                
                # Obtener nombre de la clase
                class_name = self.class_names[predicted_class]
                
                # Obtener consultas RAG sugeridas
                suggested_queries = self.classification_to_query.get(class_name, [])
                
                return {
                    "predicted_class": predicted_class,
                    "class_name": class_name,
                    "confidence": float(confidence),
                    "probabilities": {
                        self.class_names[i]: float(probabilities[0][i]) 
                        for i in range(len(self.class_names))
                    },
                    "suggested_queries": suggested_queries,
                    "device_used": str(self.device)
                }
                
        except Exception as e:
            print(f"❌ Error en clasificación: {e}")
            return {"error": f"Error en clasificación: {e}"}
    
    def get_rag_queries_for_classification(self, classification_result):
        """
        Obtiene las consultas RAG más relevantes basadas en la clasificación
        
        Args:
            classification_result: Resultado de classify_image()
            
        Returns:
            Lista de consultas RAG priorizadas
        """
        if "error" in classification_result:
            return []
        
        class_name = classification_result["class_name"]
        confidence = classification_result["confidence"]
        
        queries = self.classification_to_query.get(class_name, [])
        
        # Si la confianza es baja, agregar consultas generales
        if confidence < 0.7:
            queries.extend([
                "diagnóstico general enfermedades tomate",
                "identificación problemas cultivo tomate"
            ])
        
        return queries


# Función de utilidad para uso rápido
def classify_tomato_image(image_path, model_repo="DaniloR2011/Tomato_accuracy"):
    """
    Función de utilidad para clasificar rápidamente una imagen
    
    Args:
        image_path: Ruta a la imagen
        model_repo: Repositorio de Hugging Face
        
    Returns:
        Resultado de clasificación
    """
    classifier = TomatoImageClassifier(model_repo=model_repo)
    return classifier.classify_image(image_path)


if __name__ == "__main__":
    # Ejemplo de uso
    classifier = TomatoImageClassifier()
    
    # Probar con imagen (necesitarías tener una imagen de prueba)
    # result = classifier.classify_image("path/to/tomato_image.jpg")
    # print("Resultado:", result)
    
    print("✅ Clasificador de imágenes de tomate inicializado")
    print(f"📱 Dispositivo: {classifier.device}")
    print(f"🏷️ Clases: {list(classifier.class_names.values())}")
