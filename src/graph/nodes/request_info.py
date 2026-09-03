"""Nodo request_info: confianza baja -> pedir mejor foto EN VEZ de diagnosticar.

Es el human-in-the-loop barato: termina el turno con needs_input=True;
el cliente reanuda el MISMO thread con otra imagen y la memoria (history)
conserva el intento anterior.
"""


def run(state: dict) -> dict:
    d = state["diagnosis"]
    question = (
        f"La foto no es concluyente ({d['label']} al {d['confidence']:.0%}, "
        f"confianza {d['level']}). Sube otra foto con luz natural, hoja enfocada "
        f"y sin sombras, o confirma si ves manchas oscuras con borde amarillento."
    )
    return {"needs_input": True, "clarification": question, "final_answer": question,
            "history": [{"event": "request_info", "label": d["label"],
                         "confidence": round(d["confidence"], 3)}]}
