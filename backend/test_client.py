import requests
import time
import numpy as np
import os

# 1. 설정 및 경로
url = "http://localhost:8000/predict"
# 실제 파일 경로
FILE_PATH = "/content/drive/MyDrive/CODE 침략! 케로로 - 시계열프로젝트/prototype/backend/machine-1-1.npy"
START_INDEX = 15800  # 요청하신 시작 지점

# 2. 데이터 로드
if not os.path.exists(FILE_PATH):
    print(f"❌ 파일을 찾을 수 없습니다: {FILE_PATH}")
    exit()

print(f"📂 데이터를 로드 중입니다: {FILE_PATH}")
all_data = np.load(FILE_PATH)
# 15800 지점부터 슬라이싱
target_data = all_data[START_INDEX:]

print(f"🚀 실시간 전송 시작 (시작 지점: {START_INDEX}, 데이터 총 {len(target_data)}개)")

# 3. 데이터 루프 전송
for i, row in enumerate(target_data):
    current_index = START_INDEX + i
    # numpy array를 list로 변환
    payload = {"values": row.tolist()}
    
    try:
        start_time = time.time()
        response = requests.post(url, json=payload)
        end_time = time.time()
        
        result = response.json()
        
        # 결과 출력
        if result["status"] == "ready":
            status_str = "⚠️ 이상 발생!" if result["is_anomaly"] else "✅ 정상"
            # 15849 지점부터 이상이 발생하는지 확인하기 위해 인덱스 표시 추가
            print(f"[{current_index}] {status_str} | 점수: {result['score']:.4f} (지연: {end_time-start_time:.3f}s)")
            
            # 15849 근처에서 시각적으로 구분하기 위함
            if current_index == 15849:
                print("-" * 50)
                print("🚨 여기서부터 실제 Anomaly 구간입니다!")
                print("-" * 50)
        else:
            print(f"⏳ 빌드업 중... ({result['progress']}) (Index: {current_index})")
            
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        break

    # 0.5초 대기
    time.sleep(0.2)