import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import json
from huggingface_hub import hf_hub_download
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TomatoCNN(nn.Module):
    def __init__(self, num_classes=3):
        super(TomatoCNN, self).__init__()
        
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
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.fc_layers = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 8 * 8, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

class TomatoImageClassifier:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Usando dispositivo: {self.device}")
        
        # Cargar token de HuggingFace desde credentials.json
        self.token = self._load_huggingface_token()
        
        # Definir las clases
        self.classes = ['Sana', 'Tizon_tardio', 'Otras_enfermedades']
        
        # Definir transformaciones
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Cargar modelo
        self.model = self._load_model()
    
    def _load_huggingface_token(self):
        """Carga el token de HuggingFace desde credentials.json o variables de entorno"""
        try:
            # Intentar cargar desde credentials.json
            with open('credentials.json', 'r') as f:
                credentials = json.load(f)
                if 'huggingface_token' in credentials:
                    logger.info("Token de HuggingFace cargado desde credentials.json")
                    return credentials['huggingface_token']
        except FileNotFoundError:
            logger.warning("Archivo credentials.json no encontrado")
        except KeyError:
            logger.warning("Token huggingface_token no encontrado en credentials.json")
        
        # Fallback a variable de entorno
        token = os.getenv('HUGGINGFACE_TOKEN')
        if token:
            logger.info("Token de HuggingFace cargado desde variable de entorno")
            return token
        
        logger.error("No se pudo cargar el token de HuggingFace")
        return None
    
    def _load_model(self):
        """Carga el modelo desde HuggingFace Hub"""
        try:
            logger.info("Descargando modelo desde HuggingFace Hub...")
            
            # Descargar modelo con token
            model_path = hf_hub_download(
                repo_id="DaniloR2011/Tomato_accuracy",
                filename="best_model_3class.pth",
                token=self.token,
                cache_dir="./models_cache"
            )
            
            logger.info(f"Modelo descargado en: {model_path}")
            
            # Cargar modelo
            model = TomatoCNN(num_classes=3)
            state_dict = torch.load(model_path, map_location=self.device)
            model.load_state_dict(state_dict)
            model.to(self.device)
            model.eval()
            
            logger.info("Modelo cargado correctamente")
            return model
            
        except Exception as e:
            logger.error(f"Error al cargar modelo desde HuggingFace: {e}")
            # Fallback al modelo local si existe
            local_path = "best_model_3class.pth"
            if os.path.exists(local_path):
                logger.info("Usando modelo local como fallback")
                model = TomatoCNN(num_classes=3)
                state_dict = torch.load(local_path, map_location=self.device)
                model.load_state_dict(state_dict)
                model.to(self.device)
                model.eval()
                return model
            else:
                logger.error("No se pudo cargar ningún modelo")
                raise Exception("No se pudo cargar el modelo")
    
    def classify_image(self, image_path):
        """Clasifica una imagen de tomate"""
        try:
            # Cargar y procesar imagen
            image = Image.open(image_path).convert('RGB')
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Realizar predicción
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                predicted_class_idx = torch.argmax(outputs, dim=1).item()
                confidence = probabilities[0][predicted_class_idx].item()
            
            predicted_class = self.classes[predicted_class_idx]
            
            logger.info(f"Clasificación: {predicted_class} (confianza: {confidence:.2%})")
            
            return {
                'class': predicted_class,
                'confidence': confidence,
                'probabilities': {
                    self.classes[i]: prob.item() 
                    for i, prob in enumerate(probabilities[0])
                }
            }
            
        except Exception as e:
            logger.error(f"Error en clasificación: {e}")
            raise Exception(f"Error al clasificar imagen: {e}")