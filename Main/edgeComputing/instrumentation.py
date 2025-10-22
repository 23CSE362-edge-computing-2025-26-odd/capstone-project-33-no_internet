# Main/edgeComputing/instrumentation.py
import os, time, json, io, psutil
from typing import Any, Dict

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def write_event(results_dir: str, node: str, event: str, value: Any, unit: str = ""):
    ensure_dir(results_dir)
    rec = {"ts": time.time(), "node": str(node), "event": event, "value": value, "unit": unit}
    with open(os.path.join(results_dir, f"{node}_events.log"), "a") as f:
        f.write(json.dumps(rec) + "\n")

def model_bytes_from_state(state_dict) -> int:
    import torch, io
    buf = io.BytesIO()
    torch.save(state_dict, buf)
    return len(buf.getvalue())

def cpu_mem_snapshot() -> Dict[str, Any]:
    p = psutil.Process()
    return {"cpu_percent": psutil.cpu_percent(interval=0.0), "mem_rss": p.memory_info().rss}
