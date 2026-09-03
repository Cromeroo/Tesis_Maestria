"""UI Streamlit: delgada (sin lógica de negocio) pero cuidada.

- Estado del sistema legible en el sidebar (no un dict crudo).
- Input por archivo o cámara, preview, barras por clase, historial en sesión,
  reporte descargable .md y manejo de errores en cada capa.
"""
import io
import os
import tempfile
import time
import traceback
import uuid
from datetime import datetime

import streamlit as st
from PIL import Image

LABEL_META = {
    "Sana": ("🟢 Sana", "success", "#22c55e"),
    "Tizon_tardio": ("🔴 Tizón tardío", "error", "#ef4444"),
    "Otras_enfermedades": ("🟡 Otra enfermedad", "warning", "#f59e0b"),
}

CSS = """
<style>
.prob-track { background: #1f2937; border-radius: 8px; height: 18px; margin: 2px 0 10px 0; }
.prob-fill { border-radius: 8px; height: 18px; text-align: right; font-size: 12px;
             color: white; padding-right: 6px; line-height: 18px; }
.result-card { border: 1px solid #374151; border-radius: 12px; padding: 16px; margin-bottom: 12px; }
.small-muted { color: #9ca3af; font-size: 0.85em; }
</style>
"""


# ---------- servicio ----------
class AgentHttpClient:
    """Habla con el orquestador por HTTP (modo compose: AGENT_URL=http://api:8000)."""

    def __init__(self, base_url: str):
        import httpx
        self.client = httpx.Client(base_url=base_url.rstrip("/"), timeout=120.0)

    def health(self) -> dict:
        return self.client.get("/health").json()

    def diagnose(self, image_path: str, question: str = "¿Qué debo hacer?",
                 thread_id: str | None = None, lot_id: str | None = None) -> dict:
        with open(image_path, "rb") as f:
            data = {"question": question or "¿Qué debo hacer?",
                    "thread_id": thread_id or "", "lot_id": lot_id or ""}
            r = self.client.post("/diagnose", files={"file": ("img.jpg", f)}, data=data)
        r.raise_for_status()
        return r.json()

    def lot(self, lot_id: str) -> dict:
        return self.client.get(f"/lots/{lot_id}").json()


@st.cache_resource(show_spinner="Cargando modelo y grafo...")
def get_service():
    import os
    if os.getenv("AGENT_URL"):
        return AgentHttpClient(os.environ["AGENT_URL"])
    from src.services.diagnosis import DiagnosisService
    return DiagnosisService()


def health_sidebar(health: dict) -> None:
    st.sidebar.header("🖥️ Sistema")
    dev = health.get("device", {})
    c1, c2 = st.sidebar.columns(2)
    c1.metric("Device", dev.get("device", "?"))
    c2.metric("Chroma docs", health.get("chroma_docs", 0))
    st.sidebar.write("CNN:", "✅ pesos cargados" if health.get("cnn_loaded") else "⚠️ sin pesos (random)")
    st.sidebar.write("LLM:", "✅ Gemini" if health.get("llm") else "⚠️ offline (respuestas locales)")
    if not health.get("chroma_docs"):
        st.sidebar.info("Sin documentos: corre `python scripts/ingest_docs.py --docs ./Documentos`.")


# ---------- resultados ----------
def prob_bars(probs: list[dict]) -> None:
    for p in probs:
        _, _, color = LABEL_META.get(p["label"], ("", "", "#6b7280"))
        pct = p["confidence"] * 100
        st.markdown(f"**{p['label']}** — {pct:.1f}%"
                    f"<div class='prob-track'><div class='prob-fill' "
                    f"style='width:{max(pct, 2):.1f}%; background:{color};'>{pct:.0f}%</div></div>",
                    unsafe_allow_html=True)


def build_report(filename: str, question: str, out: dict) -> str:
    d = out["diagnosis"]
    lines = [f"# Diagnóstico — {filename}", "",
             f"- Fecha: {datetime.now():%Y-%m-%d %H:%M}",
             f"- Clase: **{d['label']}** ({d['confidence']:.1%}, confianza {d['level']})",
             f"- Severidad foliar: {(d.get('severity') or 0):.1%} ({d.get('severity_level', '?')})",
             f"- Pregunta: {question}", ""]
    for p in d.get("probs", []):
        lines.append(f"- {p['label']}: {p['confidence']:.1%}")
    lines += ["", "## Recomendación", "", out.get("final_answer", ""), ""]
    if out.get("warnings"):
        lines += ["## Avisos", ""] + [f"- ⚠️ {w}" for w in out["warnings"]] + [""]
    if out.get("contexts"):
        lines += ["## Fuentes", ""]
        for c in out["contexts"]:
            lines.append(f"- {c['query']}: {', '.join(set(c['sources'])) or 'sin fuentes'}")
    return "\n".join(lines)


def show_result(image: Image.Image, filename: str, question: str, out: dict) -> None:
    if out.get("needs_input"):
        st.warning(f"📸 {out.get('clarification', 'Sube una mejor foto.')}")
        st.caption(f"El agente recuerda este caso (hilo `{out.get('thread_id', '?')}`). "
                   "Sube otra foto para continuar con la misma memoria.")
        return
    if not out.get("in_scope", True):
        st.info(f"🚧 {out.get('final_answer', '')}")
        return
    d = out["diagnosis"]
    label_txt, box, _ = LABEL_META.get(d["label"], (d["label"], "info", ""))
    getattr(st, box)(f"{label_txt} — {d['confidence']:.1%} (confianza {d['level']})")
    st.progress(min(max(d["confidence"], 0.0), 1.0))
    with st.expander("📊 Probabilidades por clase", expanded=True):
        prob_bars(d.get("probs", []))
    for w in out.get("warnings", []):
        st.warning(f"⚠️ {w}")
    st.subheader("💡 Recomendación")
    st.markdown(out.get("final_answer", "*Sin respuesta*"))
    with st.expander(f"📚 Fuentes RAG ({len(out.get('contexts', []))} consultas)"):
        for c in out.get("contexts", []):
            st.markdown(f"**{c['query']}**")
            st.caption(f"Fuentes: {', '.join(set(c['sources'])) or 'sin fuentes'}")
            for doc in c.get("documents", [])[:2]:
                st.caption(f"> {doc[:300]}…")
    st.download_button("⬇️ Descargar reporte .md", build_report(filename, question, out),
                       file_name=f"reporte_{int(time.time())}.md", mime="text/markdown")


def diagnose_and_store(service, image: Image.Image, filename: str, question: str) -> None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        image.convert("RGB").save(tmp.name)
        path = tmp.name
    try:
        with st.spinner("Agente en curso: visión → triage → RAG → síntesis..."):
            out = service.diagnose(path, question or "¿Qué debo hacer?",
                                   thread_id=st.session_state.thread_id,
                                   lot_id=st.session_state.get("lot_id") or None)
    except Exception as e:
        st.error(f"Falló el diagnóstico: {e}")
        with st.expander("Detalle técnico"):
            st.code(traceback.format_exc())
        return
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass
    st.session_state.history.insert(0, {"time": datetime.now().strftime("%H:%M:%S"),
                                        "file": filename, "question": question,
                                        "image": image.copy(), "result": out})
    st.session_state.last_result = (filename, question, out)


# ---------- app ----------
def main() -> None:
    st.set_page_config(page_title="Tizón Tardío — Agente LangGraph", page_icon="🍅", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    st.title("🍅 Agente LangGraph — Tizón tardío del tomate")
    st.caption("Visión por computadora + RAG científico. Sube una hoja y obtén diagnóstico con fuentes.")

    if "history" not in st.session_state:
        st.session_state.history = []
    if "last_result" not in st.session_state:
        st.session_state.last_result = None
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = uuid.uuid4().hex[:12]

    try:
        service = get_service()
        health = service.health()
    except Exception as e:
        st.error(f"No se pudo inicializar el servicio: {e}")
        with st.expander("Detalle técnico"):
            st.code(traceback.format_exc())
        st.stop()
    health_sidebar(health)
    st.sidebar.caption(f"Hilo: `{st.session_state.thread_id}` (memoria del caso)")
    st.session_state.lot_id = st.sidebar.text_input(
        "🌱 Lote (seguimiento)", value=st.session_state.get("lot_id", "lote-1"),
        help="Cada diagnóstico se registra en el timeline de este lote.")
    if st.sidebar.button("🆕 Nuevo caso (hilo nuevo)"):
        st.session_state.thread_id = uuid.uuid4().hex[:12]
        st.rerun()
    if st.sidebar.button("🧹 Vaciar historial"):
        st.session_state.history = []
        st.session_state.last_result = None
        st.rerun()

    tab_diag, tab_hist, tab_lote = st.tabs(
        ["🔍 Diagnóstico", f"🕘 Historial ({len(st.session_state.history)})", "🌱 Lote"])

    with tab_diag:
        col_in, col_out = st.columns([1, 1.2])
        with col_in:
            st.header("📤 Entrada")
            up_tab, cam_tab = st.tabs(["Archivo", "Cámara"])
            with up_tab:
                uploaded = st.file_uploader("Imagen de la hoja", type=["jpg", "jpeg", "png"])
            with cam_tab:
                uploaded = st.camera_input("Foto directa") if not uploaded else uploaded
            question = st.text_input("Pregunta (opcional)", "¿Qué debo hacer?")
            image = None
            if uploaded:
                try:
                    image = Image.open(uploaded).convert("RGB")
                    st.image(image, caption=getattr(uploaded, "name", "captura"),
                             use_container_width=True)
                except Exception as e:
                    st.error(f"Imagen inválida: {e}")
            run = st.button("🔍 Diagnosticar", type="primary", disabled=image is None,
                            use_container_width=True)
            if run and image is not None:
                diagnose_and_store(service, image,
                                   getattr(uploaded, "name", "captura.jpg"), question)
        with col_out:
            st.header("📊 Resultado")
            if st.session_state.last_result is None:
                st.info("👈 Sube una imagen y pulsa **Diagnosticar**.")
            else:
                filename, question, out = st.session_state.last_result
                thumb = st.session_state.history[0]["image"] if st.session_state.history else None
                show_result(thumb or image, filename, question, out)

    with tab_lote:
        lot_id = st.session_state.get("lot_id", "lote-1")
        st.header(f"🌱 Lote `{lot_id}` — progresión")
        try:
            data = service.lot(lot_id)
        except Exception as e:
            st.error(f"No se pudo leer el lote: {e}")
            data = {"summary": {"records": 0}, "timeline": []}
        s = data.get("summary", {})
        if not s.get("records"):
            st.info("Sin registros: diagnostica con este lote activo para empezar el timeline.")
        else:
            m1, m2, m3 = st.columns(3)
            m1.metric("Registros", s["records"])
            m2.metric("Última severidad", f"{(s.get('last_severity') or 0):.1%}")
            trend = s.get("trend", 0) or 0
            m3.metric("Tendencia", f"{trend:+.1%}",
                      delta="mejorando" if s.get("improving") else "empeorando")
            st.caption(f"Por clase: {s.get('by_label', {})}")
            for r in data.get("timeline", [])[:20]:
                sev = r.get("severity") or 0
                mark = "❓" if r.get("needs_input") else "📷"
                st.markdown(
                    f"{mark} **{r.get('label')}** ({(r.get('confidence') or 0):.0%}) · "
                    f"sev {sev:.1%} "
                    f"<div class='prob-track'><div class='prob-fill' style='width:"
                    f"{max(sev * 100, 2):.1f}%; background:#f59e0b;'>{sev:.0%}</div></div>",
                    unsafe_allow_html=True)

    with tab_hist:
        if not st.session_state.history:
            st.info("Aún no hay diagnósticos en esta sesión.")
        for i, h in enumerate(st.session_state.history):
            d = h["result"]["diagnosis"]
            with st.expander(f"{h['time']} · {h['file']} → {d['label']} ({d['confidence']:.0%})",
                             expanded=(i == 0)):
                c1, c2 = st.columns([1, 2])
                c1.image(h["image"], use_container_width=True)
                with c2:
                    st.markdown(f"**Pregunta:** {h['question'] or '—'}")
                    st.markdown(h["result"].get("final_answer", "")[:1200])
                    if st.button("Ver completo", key=f"hist_{i}"):
                        st.session_state.last_result = (h["file"], h["question"], h["result"])
                        st.rerun()


if __name__ == "__main__":
    main()
