import torch
import torch.nn as nn
import numpy as np
import joblib
import os
from fastapi import FastAPI, HTTPException
from collections import deque
from pydantic import BaseModel

# 제공해주신 모델 파일들 임포트 (디렉토리 구조에 맞게 조정 필요)
from model.AnomalyTransformer import AnomalyTransformer

app = FastAPI()

# --- 전역 설정 ---
THRESHOLD = 0.50546515  # 사용자가 확인한 machine-1-1 임계치
WIN_SIZE = 100
ENC_IN = 38
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 모델 및 스케일러 로드 ---
# 1. 모델 가중치 로드
model = AnomalyTransformer(win_size=WIN_SIZE, enc_in=ENC_IN, c_out=ENC_IN).to(DEVICE)
checkpoint_path = "checkpoints/machine-1-1/SMD_machine-1-1_checkpoint.pth"
if os.path.exists(checkpoint_path):
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model.eval()
    print("✅ Model loaded successfully.")
else:
    print(f"❌ Checkpoint not found at {checkpoint_path}")

# 2. 스케일러 로드 (추출한 pkl 파일)
scaler_path = "scaler_machine-1-1.pkl"
scaler = joblib.load(scaler_path)
print("✅ Scaler loaded successfully.")

# 3. 실시간 데이터 버퍼 (최신 100개 데이터 유지)
data_buffer = deque(maxlen=WIN_SIZE)

# --- 요청 데이터 모델 정의 ---
class SensorData(BaseModel):
    values: list  # 38개의 float 리스트

# --- 핵심 연산 함수 (solver.py의 my_kl_loss 로직 포함) ---
def my_kl_loss(p, q):
    res = p * (torch.log(p + 0.0001) - torch.log(q + 0.0001))
    return torch.mean(torch.sum(res, dim=-1), dim=1)

def calculate_anomaly_score(window_tensor):
    with torch.no_grad():
        # 1. 모델 추론
        output, series, prior, _ = model(window_tensor)
        
        # 2. Reconstruction Loss (MSE)
        # input: [1, 100, 38], output: [1, 100, 38]
        loss = torch.mean((window_tensor - output) ** 2, dim=-1) # [1, 100]
        
        # 3. Association Discrepancy (KL Loss)
        series_loss = 0.0
        prior_loss = 0.0
        for i in range(len(series)):
            series_loss += my_kl_loss(series[i], prior[i])
            prior_loss += my_kl_loss(prior[i], series[i])
        
        # 4. 점수 산출 (최신 타임스텝의 점수만 추출)
        metric = torch.softmax((-series_loss - prior_loss), dim=-1)
        score = metric * loss
        
        return score[0, -1].item() # 윈도우의 가장 마지막 점수

# --- API 엔드포인트 ---
@app.post("/predict")
async def predict(data: SensorData):
    if len(data.values) != ENC_IN:
        raise HTTPException(status_code=400, detail=f"Input must have {ENC_IN} features.")

    # 1. 정규화 (학습 시 기준 적용)
    scaled_val = scaler.transform([data.values])[0]
    
    # 2. 버퍼 업데이트
    data_buffer.append(scaled_val)
    
    # 3. 데이터가 충분히 쌓였는지 확인
    if len(data_buffer) < WIN_SIZE:
        return {
            "status": "collecting",
            "progress": f"{len(data_buffer)}/{WIN_SIZE}",
            "is_anomaly": False
        }

    # 4. 텐서 변환 및 추론
    input_tensor = torch.FloatTensor(list(data_buffer)).unsqueeze(0).to(DEVICE)
    score = calculate_anomaly_score(input_tensor)
    
    # 5. 결과 반환
    is_anomaly = score > THRESHOLD
    
    return {
        "status": "ready",
        "score": float(score),
        "threshold": THRESHOLD,
        "is_anomaly": bool(is_anomaly)
    }

if __name__ == "__main__":
    import uvicorn
    # 0.5초 주기에 대응하기 위해 workers는 1로 설정 (순서 보장)
    uvicorn.run(app, host="0.0.0.0", port=8000)