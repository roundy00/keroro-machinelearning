# backend/replay_client.py
import time
import json
import numpy as np
import requests

API = "http://127.0.0.1:8000/ingest"

def replay(npy_path: str, machine_id="machine-1-1", hz=1.0, max_steps=None):
    data = np.load(npy_path).astype(np.float32)  # (T,K)
    dt = 1.0 / hz

    T = data.shape[0]
    if max_steps is not None:
        T = min(T, max_steps)

    for t in range(T):
        payload = {
            "machine_id": machine_id,
            "timestamp": float(t),
            "values": data[t].tolist(),
        }
        r = requests.post(API, json=payload, timeout=30)
        print(t, r.json())
        time.sleep(dt)

if __name__ == "__main__":
    replay("/content/drive/MyDrive/CODE 침략! 케로로 - 시계열프로젝트/ryui/telemanom/data/test/machine-1-1.npy", hz=1.0, max_steps=1000)
