"""Ensamblaje del grafo.

classify -> triage -> plan -> retrieve -> synthesize
                    |          |
                    |          └-> request_info (confianza baja, human-in-the-loop)
                    └-> synthesize (abstención fuera de dominio)

Dependencias por `functools.partial` (sin globales mutables).
`checkpointer` (SqliteSaver/MemorySaver) da memoria conversacional por thread.
"""
from functools import partial
from langgraph.graph import StateGraph, END
from src.graph.state import AgentState
from src.graph.nodes import classify, plan, retrieve, synthesize, triage, request_info
from src.graph.edges import route_after_triage, route_after_plan


def build_graph(classifier, retriever, llm, settings, checkpointer=None):
    g = StateGraph(AgentState)
    g.add_node("classify", partial(classify.run, classifier=classifier, settings=settings))
    g.add_node("triage", partial(triage.run, llm=llm))
    g.add_node("plan", partial(plan.run, llm=llm, settings=settings))
    g.add_node("retrieve", partial(retrieve.run, retriever=retriever))
    g.add_node("request_info", request_info.run)
    g.add_node("synthesize", partial(synthesize.run, llm=llm))
    g.set_entry_point("classify")
    g.add_edge("classify", "triage")
    g.add_conditional_edges("triage", route_after_triage,
                            {"plan": "plan", "synthesize": "synthesize"})
    g.add_conditional_edges("plan", route_after_plan,
                            {"retrieve": "retrieve", "request_info": "request_info"})
    g.add_edge("retrieve", "synthesize")
    g.add_edge("request_info", END)
    g.add_edge("synthesize", END)
    return g.compile(checkpointer=checkpointer)
