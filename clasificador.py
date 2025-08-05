import torch
import torch.nn as nn
from torchvision import transforms as T
from PIL import Image
import streamlit as st

# --------------------- Arquitectura ---------------------
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
            nn.ReLU(),-
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

# --------------------- Cargar modelo ---------------------
@st.cache_resource
def cargar_modelo(path="best_model_3class.pth"):
    model = SimpleCNN()
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model

model = cargar_modelo()

# --------------------- Mapeo clases ---------------------
idx_to_label = {0: "Sana", 1: "Tizon_tardio", 2: "Otras_enfermedades"}

# --------------------- Transformación ---------------------
transform = T.Compose([
    T.Resize((128, 128)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --------------------- Interfaz Streamlit ---------------------
st.title("🌿 Clasificador de enfermedades en tomate")
st.write("Sube una imagen de una hoja para identificar su estado:")

archivo = st.file_uploader("📤 Selecciona una imagen de hoja", type=["jpg", "jpeg", "png"])

if archivo:
    imagen = Image.open(archivo).convert("RGB")
    st.image(imagen, caption="Imagen cargada", use_column_width=True)

    # Procesamiento
    tensor = transform(imagen).unsqueeze(0)
    with torch.no_grad():
        salida = model(tensor)
        pred = salida.argmax(1).item()
        clase = idx_to_label[pred]

    # Resultado
    st.markdown(f"### 🧠 Resultado: **{clase}**")

    if clase == "Tizon_tardio":
        st.warning("⚠️ La hoja muestra síntomas de **Tizón Tardío**. Se recomienda consultar documentos de tratamiento.")
    elif clase == "Sana":
        st.success("✅ Hoja sana. No se requiere intervención.")
    else:
        st.info("🔍 Hoja enferma, pero no es Tizón Tardío.")
