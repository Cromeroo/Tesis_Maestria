"""Tool RAG: búsqueda en la base de conocimiento. Función pura + wrapper @tool."""


def search_knowledge(query: str, retriever, k: int = 5) -> dict:
    """Busca y devuelve documentos + fuentes. Nunca lanza: devuelve vacío + warning."""
    try:
        documents, sources = retriever.search(query, k=k)
        return {"query": query, "documents": documents, "sources": sources}
    except Exception as e:
        return {"query": query, "documents": [], "sources": [],
                "warning": f"retrieval falló: {e}"}


def make_rag_tool(retriever, k: int = 5):
    """Wrapper @tool `search_knowledge` para un futuro ToolNode."""
    from langchain_core.tools import tool

    @tool("search_knowledge")
    def _search_knowledge(query: str) -> str:
        """Busca en la base de conocimiento sobre tizón tardío del tomate."""
        out = search_knowledge(query, retriever, k=k)
        if not out["documents"]:
            return "(sin resultados)"
        return "\n\n".join(f"[Fuente: {s}]\n{d}"
                           for d, s in zip(out["documents"], out["sources"]))

    return _search_knowledge
