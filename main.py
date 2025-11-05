from __future__ import annotations

import base64
import re
import time
from pathlib import Path
from typing import Dict, Optional

import cv2
import mediapipe as mp
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from capture_service import (
    app as webrtc_app,
    set_recognizer,
    get_latest_sign,
    get_stats,
)

BASE_DIR = Path(__file__).parent
DATASET_PATH = BASE_DIR / "assets" / "dataSet.txt"

if not DATASET_PATH.exists():
    raise FileNotFoundError(
        f"Dataset file not found: {DATASET_PATH}\n"
        f"'{BASE_DIR.name}/assets/dataSet.txt' 경로를 확인하세요."
    )

# 학습 데이터 로드
file = np.genfromtxt(str(DATASET_PATH), delimiter=",", dtype=np.float32)
anglefile = np.array(file[:, :-1], dtype=np.float32)
labelfile = np.array(file[:, -1], dtype=np.float32)

gesture: Dict[int, str] = {
    0: "alpha",
    1: "bravo",
    2: "charlie",
    3: "delta",
    4: "echo",
    5: "foxtrot",
    6: "golf",
    7: "hotel",
    8: "india",
    9: "juliet",
    10: "kilo",
    11: "lima",
    12: "mike",
    13: "november",
    14: "oscar",
    15: "papa",
    16: "quebec",
    17: "romeo",
    18: "sierra",
    19: "tango",
    20: "uniform",
    21: "victor",
    22: "whiskey",
    23: "x-ray",
    24: "yankee",
    25: "zulu",
    26: "spacing",
    27: "clear",
}

knn = cv2.ml.KNearest_create()
knn.train(anglefile, cv2.ml.ROW_SAMPLE, labelfile)

# WebRTC 앱 가져오기
app = webrtc_app

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def classify_from_landmarks(landmarks: np.ndarray) -> str:

    if landmarks.shape != (42, 3):
        return ""

    left = landmarks[:21]
    right = landmarks[21:]
    # handedness score 합이 더 큰 손을 선택 (score가 0이면 손이 없는 것으로 간주)
    if right[:, 2].sum() > left[:, 2].sum():
        active = right
    else:
        active = left

    if active[:, 2].sum() < 1e-3:
        return ""

    joint = np.zeros((21, 3), dtype=np.float32)
    joint[:, :2] = active[:, :2]
    # 3D 벡터 계산 (z값 없으므로 0으로 두어 2D 각도 기반으로 추정)
    v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :]
    v2 = joint[
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], :
    ]
    v = v2 - v1
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    v = v / norms

    compare_v1 = v[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18], :]
    compare_v2 = v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], :]
    dot = np.einsum("nt,nt->n", compare_v1, compare_v2)
    dot = np.clip(dot, -1.0, 1.0)
    angle = np.degrees(np.arccos(dot)).astype(np.float32)

    _, results, _, _ = knn.findNearest(angle.reshape(1, -1), 3)
    index = int(results[0][0])
    return gesture.get(index, "")


# WebRTC 서비스에 recognizer 등록
set_recognizer(classify_from_landmarks)


class FrameRequest(BaseModel):
    frame_data: str


class FrameResponse(BaseModel):
    detected_sign: str
    processed_frame: str


@app.post("/translate", response_model=FrameResponse)
async def recognize(req: FrameRequest):
    """
    기존 Base64 업로드 경로 유지 (필요 시).
    """
    img_str = re.sub("^data:image/.+;base64,", "", req.frame_data)
    img_bytes = base64.b64decode(img_str)
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    detected_sign = ""
    result = mp_hands.Hands(
        max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5
    ).process(img_rgb)

    if result.multi_hand_landmarks:
        for res in result.multi_hand_landmarks:
            joint = np.array(
                [[lm.x, lm.y, lm.z] for lm in res.landmark], dtype=np.float32
            )
            v1 = joint[
                [0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :
            ]
            v2 = joint[
                [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                :,
            ]
            v = v2 - v1
            v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]
            compare_v1 = v[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18], :]
            compare_v2 = v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], :]
            angle = np.degrees(np.arccos(np.einsum("nt,nt->n", compare_v1, compare_v2)))
            _, results, _, _ = knn.findNearest(angle.reshape(1, -1), 3)
            index = int(results[0][0])
            detected_sign = gesture.get(index, "")
            mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

    _, buffer = cv2.imencode(".jpg", img)
    img_base64 = base64.b64encode(buffer).decode("utf-8")
    return FrameResponse(
        detected_sign=detected_sign,
        processed_frame=f"data:image/jpeg;base64,{img_base64}",
    )


# ---------------------------
#  WebRTC 상태 조회 라우트
# ---------------------------
@app.get("/capture/status")
async def capture_status():
    """WebRTC 처리 상태 조회"""
    stats = get_stats()
    return {
        "running": stats["frames_captured"] > 0,
        "latest_sign": stats["latest_sign"],
        "frames_captured": stats["frames_captured"],
        "duration_sec": stats["duration_sec"],
        "avg_fps": stats["avg_fps"],
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
