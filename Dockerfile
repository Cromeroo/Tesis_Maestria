# Imagen por defecto: CPU. Funciona en cualquier host (incl. Docker Desktop Windows).
# La GPU (tu 9060 XT) se usa en NATIVO (Windows + torch ROCm), no aquí:
# Docker Desktop/WSL2 no hace passthrough de ROCm a contenedores Linux.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEVICE=cpu \
    TESIS_OFFLINE=0

WORKDIR /app
COPY requirements.txt requirements.docker.txt ./
RUN pip install --no-cache-dir -r requirements.docker.txt
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY configs/ ./configs/

EXPOSE 8000 8501
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
