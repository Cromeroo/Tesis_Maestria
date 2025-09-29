#!/usr/bin/env python3
"""
🔍 ANALIZADOR DE IMÁGENES DE TOMATE
Módulo para clasificación de enfermedades en hojas de tomate usando CNN
"""

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import json
from pathlib import Path
from huggingface_hub import hf_hub_download, login

class TomatoCNN(nn.Module):
    """Red CNN para clasificación de enfermedades de tomate"""
    
    def __init__(self, num_classes=3):
        super(TomatoCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            # Primera capa convolucional
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Segunda capa convolucional
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Tercera capa convolucional
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Cuarta capa convolucional
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.fc_layers = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 8 * 8, 512),  # 16384 = 256 * 8 * 8
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

class ImageAnalyzer:
    """Analizador de imágenes de tomate para detección de enfermedades"""
    
    def __init__(self, model_path=None, use_huggingface=True):
        """Inicializar el analizador con el modelo CNN"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if use_huggingface:
            # Usar modelo desde Hugging Face
            try:
                # Cargar token desde credentials.json
                token = self._load_huggingface_token()
                if token:
                    login(token=token)
                print("✅ Autenticación exitosa con Hugging Face")
                
                # Descargar modelo desde Hugging Face
                model_path = hf_hub_download(
                    repo_id="DaniloR2011/Tomato_accuracy",
                    filename="best_model_3class.pth",
                    cache_dir="./models_cache"
                )
                print(f"✅ Modelo descargado desde Hugging Face: {model_path}")
                
                # Cargar el modelo original sin modificar la arquitectura
                checkpoint = torch.load(model_path, map_location=self.device)
                
                # Extraer información del checkpoint para determinar la arquitectura correcta
                if isinstance(checkpoint, dict):
                    if 'state_dict' in checkpoint:
                        state_dict = checkpoint['state_dict']
                    elif 'model_state_dict' in checkpoint:
                        state_dict = checkpoint['model_state_dict']
                    else:
                        state_dict = checkpoint
                else:
                    state_dict = checkpoint
                
                # Detectar la arquitectura correcta del modelo
                self.model = self._create_model_from_state_dict(state_dict)
                self.model.load_state_dict(state_dict)
                self.model.to(self.device)
                self.model.eval()
                
                print(f"✅ Modelo cargado correctamente desde Hugging Face")
                
            except Exception as e:
                print(f"❌ Error al cargar desde Hugging Face: {e}")
                print("🔄 Intentando cargar modelo local...")
                use_huggingface = False
        
        if not use_huggingface:
            # Fallback al modelo local
            if model_path is None:
                model_path = Path(__file__).parent.parent.parent / "best_model_3class.pth"
            
            self.model = TomatoCNN(num_classes=3)
            
            try:
                if os.path.exists(model_path):
                    checkpoint = torch.load(model_path, map_location=self.device)
                    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                        self.model.load_state_dict(checkpoint['state_dict'])
                    else:
                        self.model.load_state_dict(checkpoint)
                    self.model.eval()
                    print(f"✅ Modelo local cargado desde {model_path}")
                else:
                    print(f"❌ No se encontró el modelo en {model_path}")
                    raise FileNotFoundError(f"Modelo no encontrado: {model_path}")
            except Exception as e:
                print(f"❌ Error al cargar el modelo local: {e}")
                raise
        
        # Definir las transformaciones - Usando 128x128 para obtener 8x8 después de 4 poolings (128/16=8)
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Mapeo de clases
        self.class_names = {
            0: "Saludable",
            1: "Tizón Temprano", 
            2: "Tizón Tardío"
        }
        
        # Niveles de severidad
        self.severity_levels = {
            "Saludable": "Ninguna",
            "Tizón Temprano": "Moderada",
            "Tizón Tardío": "Alta"
        }
    
    def _load_huggingface_token(self):
        """Carga el token de HuggingFace desde credentials.json o variables de entorno"""
        try:
            # Intentar cargar desde credentials.json
            with open('credentials.json', 'r') as f:
                credentials = json.load(f)
                if 'huggingface_token' in credentials:
                    print("✅ Token de HuggingFace cargado desde credentials.json")
                    return credentials['huggingface_token']
        except FileNotFoundError:
            print("⚠️ Archivo credentials.json no encontrado")
        except KeyError:
            print("⚠️ Token huggingface_token no encontrado en credentials.json")
        
        # Fallback a variable de entorno
        token = os.getenv('HUGGINGFACE_TOKEN')
        if token:
            print("✅ Token de HuggingFace cargado desde variable de entorno")
            return token
        
        print("❌ No se pudo cargar el token de HuggingFace")
        return None
    
    def _create_model_from_state_dict(self, state_dict):
        """Crear modelo con arquitectura correcta basada en el state_dict"""
        # Analizar las claves para determinar la arquitectura
        keys = list(state_dict.keys())
        
        # Detectar el nombre de las capas principales
        if any('conv_layers' in key for key in keys):
            # Usar la arquitectura TomatoCNN existente
            return TomatoCNN(num_classes=3)
        elif any('features' in key for key in keys):
            # Crear modelo con 'features' en lugar de 'conv_layers'
            return self._create_features_model()
        else:
            # Detectar automáticamente el tamaño de entrada de la primera capa FC
            fc_keys = [key for key in keys if 'fc' in key or 'classifier' in key or 'linear' in key]
            if fc_keys:
                # Encontrar la primera capa fully connected
                first_fc_key = None
                for key in fc_keys:
                    if 'weight' in key:
                        first_fc_key = key
                        break
                
                if first_fc_key:
                    fc_input_size = state_dict[first_fc_key].shape[1]
                    return self._create_adaptive_model(fc_input_size)
            
            # Fallback: usar modelo estándar
            return TomatoCNN(num_classes=3)
    
    def _create_features_model(self):
        """Crear modelo con arquitectura 'features'"""
        model = nn.Module()
        model.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        model.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 14 * 14, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 3)
        )
        
        def forward(x):
            x = model.features(x)
            x = x.view(x.size(0), -1)
            x = model.classifier(x)
            return x
        
        model.forward = forward
        return model
    
    def _create_adaptive_model(self, fc_input_size):
        """Crear modelo adaptivo basado en el tamaño de entrada FC"""
        # Calcular dimensiones convolucionales que resulten en fc_input_size
        # Asumiendo 4 capas de pooling (stride=2), imagen inicial 224x224
        # Después de 4 poolings: 224/16 = 14
        # fc_input_size = channels * 14 * 14
        channels = fc_input_size // (14 * 14)
        
        if channels == 0:  # Si no calza, usar valores por defecto
            channels = 256
            fc_input_size = 256 * 8 * 8  # Ajustar para 8x8
        
        class AdaptiveCNN(nn.Module):
            def __init__(self):
                super(AdaptiveCNN, self).__init__()
                self.conv_layers = nn.Sequential(
                    nn.Conv2d(3, 32, kernel_size=3, padding=1),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    
                    nn.Conv2d(32, 64, kernel_size=3, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    
                    nn.Conv2d(64, 128, kernel_size=3, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    
                    nn.Conv2d(128, channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                )
                
                self.fc_layers = nn.Sequential(
                    nn.Dropout(0.5),
                    nn.Linear(fc_input_size, 512),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.5),
                    nn.Linear(512, 3)
                )
            
            def forward(self, x):
                x = self.conv_layers(x)
                x = x.view(x.size(0), -1)
                x = self.fc_layers(x)
                return x
        
        return AdaptiveCNN()
    
    def classify_image(self, image):
        """Clasificar una imagen de hoja de tomate"""
        try:
            # Debug: Verificar que el modelo y class_names estén inicializados
            if not hasattr(self, 'class_names') or not self.class_names:
                raise AttributeError("class_names no está inicializado")
            
            if not hasattr(self, 'model') or self.model is None:
                raise AttributeError("model no está inicializado")
            
            # Asegurar que la imagen esté en formato RGB
            if isinstance(image, str):
                image = Image.open(image)
            
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Aplicar transformaciones
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Realizar predicción
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                predicted_class = torch.argmax(outputs, dim=1).item()
                confidence = probabilities[0][predicted_class].item()
            
            # Validar que la clase predicha esté en el rango esperado
            if predicted_class not in self.class_names:
                print(f"⚠️ Clase predicha fuera de rango: {predicted_class}, clases disponibles: {list(self.class_names.keys())}")
                predicted_class = 0  # Fallback a "Saludable"
            
            # Preparar resultado
            disease_name = self.class_names[predicted_class]
            severity = self.severity_levels[disease_name]
            
            # Obtener probabilidades para todas las clases
            all_probabilities = {}
            try:
                for i, class_name in self.class_names.items():
                    if i < probabilities.shape[1]:  # Verificar que el índice esté en rango
                        all_probabilities[class_name] = probabilities[0][i].item()
                    else:
                        print(f"⚠️ Índice fuera de rango: {i}, dimensiones: {probabilities.shape}")
                        all_probabilities[class_name] = 0.0
            except Exception as prob_error:
                print(f"❌ Error al calcular probabilidades: {prob_error}")
                # Fallback: solo incluir las clases principales
                all_probabilities = {
                    "Saludable": 0.0,
                    "Tizón Temprano": 0.0,
                    "Tizón Tardío": 0.0
                }
            
            result = {
                'predicted_class': disease_name,
                'confidence': confidence,
                'severity': severity,
                'probabilities': all_probabilities,
                'is_healthy': predicted_class == 0,
                'requires_treatment': predicted_class > 0
            }
            
            return result
            
        except Exception as e:
            print(f"❌ Error en clasificación: {e}")
            return {
                'predicted_class': 'Error',
                'confidence': 0.0,
                'severity': 'Desconocida',
                'probabilities': {},
                'is_healthy': False,
                'requires_treatment': False,
                'error': str(e)
            }
    
    def analyze_multiple_images(self, images):
        """Analizar múltiples imágenes y generar consenso"""
        results = []
        for image in images:
            result = self.classify_image(image)
            results.append(result)
        
        if not results:
            return None
        
        # Calcular consenso
        disease_counts = {}
        total_confidence = 0
        
        for result in results:
            disease = result['predicted_class']
            if disease != 'Error':
                disease_counts[disease] = disease_counts.get(disease, 0) + 1
                total_confidence += result['confidence']
        
        if not disease_counts:
            return None
        
        # Enfermedad más común
        consensus_disease = max(disease_counts, key=disease_counts.get)
        consensus_count = disease_counts[consensus_disease]
        consensus_percentage = (consensus_count / len(results)) * 100
        avg_confidence = total_confidence / len(results)
        
        return {
            'consensus_disease': consensus_disease,
            'consensus_percentage': consensus_percentage,
            'average_confidence': avg_confidence,
            'total_images': len(results),
            'individual_results': results,
            'disease_distribution': disease_counts
        }

# Alias para compatibilidad
TomatoImageClassifier = ImageAnalyzer
