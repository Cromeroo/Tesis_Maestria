# 🍅 Agente LangGraph — Tizón tardío del tomate (v2 escalable)

![ci](https://github.com/Cromeroo/Tesis_Maestria/actions/workflows/ci.yml/badge.svg?branch=v2-langgraph)

Refactor de `Tesis_Maestria`. El original funcionaba pero no era escalable;
esta v2 es un **sistema de manejo del tizón tardío**: diagnóstico + severidad
foliar + seguimiento por lote, como microservicios Dockerizados.

## Estructura (cada capa en su paquete)

```
src/
  config.py            # settings pydantic (.env) — cero paths hardcodeados
  utils/device.py      # get_device(): auto|cuda|cpu (ROCm en AMD)
  models/
    cnn.py             # SimpleCNN + build_model(simple_cnn|efficientnet_b0)
    classifier.py      # LeafClassifier device-aware (único punto de visión)
    severity.py        # severidad foliar HSV (diferencial, sin modelo nuevo)
  rag/
    preprocessing.py   # extract_text_pdf, detect_lang, clean_text (sin unidecode)
    ingest.py          # IngestPipeline (fusiona los 2 pipelines duplicados de v1)
    retriever.py       # misma embedding function que la ingesta (v1 las mezclaba)
  tools/               # ← capa de tools, pura y testeable sin grafo
    vision_tool.py     # classify_leaf() + wrapper @tool para ToolNode
    rag_tool.py        # search_knowledge() + wrapper @tool (nunca lanza)
  graph/               # ← LangGraph (v1 no lo tenía, solo en el README)
    state.py           # AgentState tipado
    prompts.py         # prompts centralizados
    nodes/             # classify.py, plan.py, retrieve.py, synthesize.py
    edges.py           # routing (route_after_plan)
    builder.py         # build_graph() con functools.partial (sin globales)
  services/diagnosis.py# fachada DiagnosisService (UI/API nunca tocan el grafo)
  services/lots.py     # LotStore: timeline + tendencia por lote (sqlite)
  vision_svc/app.py    # microservicio visión :8001 (POST /classify)
  rag_svc/app.py       # microservicio RAG :8002 (POST /search)
  api/main.py          # FastAPI: /health, /diagnose, /metrics, /threads/{id}/history
                     # rate-limit, X-API-Key opcional, contadores por clase
  ui/app.py            # Streamlit delgada
scripts/benchmark_gpu.py | ingest_docs.py | evaluate.py
tests/test_pipeline.py
```

Flujo: `classify → triage → plan → retrieve → synthesize`, con dos salidas
especiales: `request_info` (confianza baja → pide mejor foto, human-in-the-loop)
y abstención directa si la pregunta está fuera de dominio. 1 sola llamada LLM
por diagnóstico (v1 hacía 4). Memoria conversacional por `thread_id`
(`SqliteSaver` en `./checkpoints.db`).

## GPU (medido en RX 9060 XT)

```bash
python scripts/benchmark_gpu.py --batch 32
# CPU: ~121 imgs/s · GPU (ROCm): ~2344 imgs/s (x19.4)
```

`DEVICE` en `.env`: `auto` usa CUDA/ROCm si existe, si no CPU. Forzar: `cuda` o `cpu`.

## Docker

> Nota honesta: Docker Desktop en Windows/WSL2 **no** hace passthrough de ROCm a
> contenedores Linux. Por eso los contenedores corren en **CPU** y la GPU se usa
> en **nativo** (benchmark/entrenamiento). `Dockerfile.rocm` es para hosts Linux con AMD.

```bash
cp .env.example .env
docker compose up --build -d   # vision:8001 + rag:8002 + api:8000 + ui:8501
curl localhost:8001/health     # vision-svc
curl localhost:8002/health     # rag-svc (docs indexados)
curl localhost:8000/health     # orquestador (mode: microservices)
docker compose --profile ingest run --rm ingest   # ingesta de ./Documentos
docker compose down
```

Sin compose (in-process): deja `VISION_URL`/`RAG_URL` vacíos y cada proceso
carga lo suyo. La UI usa `AGENT_URL=http://api:8000` en compose.

Endurecimiento API: `RATE_LIMIT_PER_MIN` (429 si se excede), `API_KEY`
(si se define, `/diagnose` exige header `X-API-Key`), `GET /threads/{id}/history`
lee la memoria del caso. CI en `.github/workflows/ci.yml` (pytest + build).

## Uso nativo

```bash
pip install -r requirements.txt
cp .env.example .env
python scripts/ingest_docs.py --docs ./Documentos
uvicorn src.api.main:app --reload --port 8000
streamlit run src/ui/app.py
python -m pytest tests/ -v   # offline, sin GCP
```
