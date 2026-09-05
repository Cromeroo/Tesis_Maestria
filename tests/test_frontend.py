"""Front robusto: corre src/ui/app.py de verdad (headless) y exige cero excepciones.

Si alguien mete un bug en la UI (widget deprecado, key inexistente, crash en
init del servicio), este test lo caza sin abrir el navegador.
"""
import os
os.environ["TESIS_OFFLINE"] = "1"

from streamlit.testing.v1 import AppTest


def _run() -> "AppTest":
    root = __import__("pathlib").Path(__file__).resolve().parent.parent
    at = AppTest.from_file(str(root / "src" / "ui" / "app.py"), default_timeout=180)
    at.run()
    return at


def test_front_renders_without_exceptions():
    at = _run()
    assert not at.exception, f"la UI crasheó: {at.exception}"


def test_front_has_expected_structure():
    at = _run()
    assert not at.exception
    assert any("Tiz" in t.value for t in at.title), "falta el título principal"
    labels = [t.label if hasattr(t, "label") else str(t) for t in at.tabs]
    for expected in ("Diagnóstico", "Historial", "Lote"):
        assert any(expected in lb for lb in labels), f"falta el tab {expected}"
    assert len(at.button) >= 1, "falta el botón Diagnosticar"
    assert len(at.sidebar.metric) >= 2, "faltan métricas de estado en el sidebar"
    assert len(at.text_input) >= 1, "falta el input de pregunta/lote"


def test_front_source_has_no_deprecated_apis():
    src = open("src/ui/app.py", encoding="utf-8").read()
    assert "use_column_width" not in src, "st.image deprecado: usar use_container_width"
    assert "st.stop" in src or "st.error" in src, "sin manejo de errores visible"
