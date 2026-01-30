# backend/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np

from .window import SlidingWindow
from .model_runner import IMDiffusionRunner
from .alert_engine import AlertEngine

app = FastAPI()

# ---- 설정 (너 환경에 맞게 수정) ----
FEATURE_DIM = 38
WINDOW_LEN = 100

CONFIG_YAML = "imdiffusion/config/base.yaml"        # 경로 맞춰줘
CKPT_PATH   = "/content/drive/MyDrive/CODE 침략! 케로로 - 시계열프로젝트/ryui/IMDiffusion/train_result/save0/machine-1-1_unconditional:True_split:10_diffusion_step:50/best-model.pth"  # 너가 학습한 ckpt

runner = IMDiffusionRunner(
    config_yaml_path=CONFIG_YAML,
    ckpt_path=CKPT_PATH,
    device="cuda:0",
    feature_dim=FEATURE_DIM,
    window_length=WINDOW_LEN,
    split=4,
    multiply_20=True,
    nsample=1,
    use_middle_uncertainty=True,
)

# threshold는 compute_score.py로 offline에서 best_proper 찾았던 것처럼 “운영용”으로 정해야 함.
# 일단 시작은 보수적으로:
alert_engine = AlertEngine(
    alarm_th=1.5,
    watch_th=1.0,
    unc_th=0.8,
    consecutive_alarm=3,
    consecutive_watch=2
)

windows = {}     # machine_id -> SlidingWindow
last_state = {}  # machine_id -> dict

class IngestRequest(BaseModel):
    machine_id: str
    timestamp: float | None = None
    values: list[float]  # length=FEATURE_DIM

@app.post("/ingest")
def ingest(req: IngestRequest):
    if len(req.values) != FEATURE_DIM:
        raise HTTPException(status_code=400, detail=f"values must have length {FEATURE_DIM}")

    if req.machine_id not in windows:
        windows[req.machine_id] = SlidingWindow(WINDOW_LEN, FEATURE_DIM)

    w = windows[req.machine_id]
    w.push(np.array(req.values, dtype=np.float32), timestamp=req.timestamp)

    if not w.ready():
        last_state[req.machine_id] = {
            "ready": False,
            "buffer_len": len(w.buf),
            "level": "WARMUP"
        }
        return last_state[req.machine_id]

    window = w.get_window()  # (L,K)
    out = runner.score_window(window)

    score_now = out["score_now"]
    unc_now = out.get("uncertainty_now", None)
    state = alert_engine.update(score_now, unc_now)

    payload = {
        "ready": True,
        "timestamp": req.timestamp,
        "score_now": score_now,
        "uncertainty_now": unc_now,
        "level": state.level,
        "reason": state.reason,
    }
    last_state[req.machine_id] = payload
    return payload

@app.get("/state/{machine_id}")
def get_state(machine_id: str):
    if machine_id not in last_state:
        raise HTTPException(status_code=404, detail="no state yet")
    return last_state[machine_id]
