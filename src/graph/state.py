"""Estado del agente LangGraph. Un solo lugar, tipado, serializable.

- `contexts`: se SOBRESCRIBE cada turno (sin reducer) para no contaminar
  turnos nuevos con retrieval viejo cuando se reanuda un thread.
- `history`/`warnings`: append, es la memoria conversacional (vía checkpointer).
"""
from typing import Annotated, List
from typing_extensions import TypedDict
from operator import add


class Diagnosis(TypedDict):
    label: str
    confidence: float
    level: str  # baja | moderada | alta
    probs: List[dict]  # [{"label":..., "confidence":...}] ordenado desc
    severity: float  # 0..1 fracción foliar afectada (diferencial)
    severity_level: str  # leve | moderada | severa | indeterminada


class RAGContext(TypedDict):
    query: str
    documents: List[str]
    sources: List[str]


class AgentState(TypedDict):
    image_path: str
    user_question: str
    diagnosis: Diagnosis
    in_scope: bool          # guardrail: ¿la pregunta es del dominio?
    needs_input: bool       # True si hay que pedir mejor foto antes de responder
    clarification: str      # pregunta al usuario cuando needs_input
    planned_queries: List[str]
    contexts: List[RAGContext]
    final_answer: str
    history: Annotated[List[dict], add]  # memoria: eventos por turno
    warnings: Annotated[List[str], add]
