"""Utilidades de device (CPU / CUDA-ROCm).

En tu 9060 XT, `torch.cuda` es el backend ROCm (`torch 2.9+rocmsdk`):
`cuda.is_available()` ya devuelve True, no hay que hacer nada raro.
"""
import torch


def get_device(preference: str = "auto") -> torch.device:
    """Resuelve el device a usar.

    preference: "auto" | "cuda" | "cpu".
    - "auto": CUDA (ROCm en AMD / NVIDIA) si está disponible, si no CPU.
    - "cuda": fuerza GPU, lanza RuntimeError si no hay.
    """
    if preference == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if preference == "cuda":
        raise RuntimeError("Se pidió CUDA pero torch.cuda.is_available() es False.")
    return torch.device("cpu")


def device_info(device: torch.device | None = None) -> dict:
    device = device or get_device()
    info = {"device": str(device), "cuda_available": torch.cuda.is_available(),
            "torch_version": torch.__version__}
    if device.type == "cuda":
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["capability"] = ".".join(map(str, torch.cuda.get_device_capability(0)))
        info["vram_total_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
    return info
