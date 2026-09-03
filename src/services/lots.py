"""Seguimiento por lote: timeline de diagnósticos en sqlite (el diferencial de manejo).

Cada diagnóstico con lot_id deja un registro; el timeline muestra progresión
de severidad en el tiempo -> de "foto suelta" a "manejo del cultivo".
"""
import sqlite3
import time


class LotStore:
    def __init__(self, path: str = "./lots.db"):
        self.path = path
        con = sqlite3.connect(path)
        con.execute("""CREATE TABLE IF NOT EXISTS lot_records(
            id INTEGER PRIMARY KEY AUTOINCREMENT, lot_id TEXT NOT NULL,
            ts REAL NOT NULL, label TEXT, confidence REAL,
            severity REAL, severity_level TEXT, needs_input INTEGER,
            question TEXT, thread_id TEXT)""")
        con.execute("CREATE INDEX IF NOT EXISTS ix_lot ON lot_records(lot_id, ts)")
        con.commit()
        con.close()

    def record(self, lot_id: str, diagnosis: dict, needs_input: bool,
               question: str, thread_id: str) -> None:
        con = sqlite3.connect(self.path)
        con.execute(
            "INSERT INTO lot_records(lot_id, ts, label, confidence, severity,"
            " severity_level, needs_input, question, thread_id)"
            " VALUES (?,?,?,?,?,?,?,?,?)",
            (lot_id, time.time(), diagnosis.get("label"), diagnosis.get("confidence"),
             diagnosis.get("severity", 0.0), diagnosis.get("severity_level", "?"),
             int(needs_input), (question or "")[:300], thread_id))
        con.commit()
        con.close()

    def timeline(self, lot_id: str, limit: int = 50) -> list[dict]:
        con = sqlite3.connect(self.path)
        con.row_factory = sqlite3.Row
        rows = con.execute(
            "SELECT ts, label, confidence, severity, severity_level, needs_input,"
            " question FROM lot_records WHERE lot_id=? ORDER BY ts DESC LIMIT ?",
            (lot_id, limit)).fetchall()
        con.close()
        return [dict(r) for r in rows]

    def summary(self, lot_id: str) -> dict:
        rows = self.timeline(lot_id, limit=1000)
        if not rows:
            return {"records": 0}
        sevs = [r["severity"] or 0 for r in rows if not r["needs_input"]]
        labels: dict[str, int] = {}
        for r in rows:
            labels[r["label"]] = labels.get(r["label"], 0) + 1
        trend = (sevs[0] - sevs[-1]) if len(sevs) >= 2 else 0.0  # desc=mejorando
        return {"records": len(rows), "by_label": labels,
                "last_severity": sevs[0] if sevs else 0.0,
                "trend": round(trend, 4),
                "improving": trend < 0}
