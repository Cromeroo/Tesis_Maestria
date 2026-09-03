"""Bordes condicionales: el routing vive aquí, no escondido en los nodos."""
from typing import Literal


def route_after_triage(state: dict) -> Literal["plan", "synthesize"]:
    return "plan" if state.get("in_scope", True) else "synthesize"


def route_after_plan(state: dict) -> Literal["retrieve", "request_info"]:
    # Confianza baja -> pedir mejor foto (human-in-the-loop) en vez de
    # gastar retrieval y arriesgar una recomendación floja.
    if state["diagnosis"]["level"] == "baja":
        return "request_info"
    return "retrieve"
