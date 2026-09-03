"""LotStore: timeline y tendencia (sqlite temporal)."""
from src.services.lots import LotStore


def _rec(label="Sana", conf=0.9, sev=0.01):
    return {"label": label, "confidence": conf, "severity": sev,
            "severity_level": "leve"}


def test_timeline_and_trend(tmp_path):
    store = LotStore(str(tmp_path / "lots.db"))
    assert store.summary("l1") == {"records": 0}
    store.record("l1", _rec(sev=0.20), False, "q1", "t1")
    store.record("l1", _rec(sev=0.05), False, "q2", "t1")
    tl = store.timeline("l1")
    assert len(tl) == 2 and tl[0]["severity"] == 0.05  # DESC por tiempo
    s = store.summary("l1")
    assert s["records"] == 2 and s["improving"] is True
    assert s["by_label"] == {"Sana": 2}
