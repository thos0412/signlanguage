from __future__ import annotations

import asyncio
import time
from typing import Any, Callable, Optional, Tuple

import av
import cv2
import mediapipe as mp
import numpy as np
from aiortc import RTCPeerConnection, RTCSessionDescription
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

# ===== 기본 설정 =====
TO_PIXELS = False  # 0~1 좌표 유지 (recognizer가 기대하는 형태)
MODEL_COMPLEX = 0
MIN_DET_CONF = 0.5
MIN_TRK_CONF = 0.5

mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# FastAPI 객체 생성 및 CORS 설정(all open)
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# 공용 함수
# ---------------------------------------------------------------------------


def create_hands() -> mp_hands.Hands:
    return mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=MODEL_COMPLEX,
        min_detection_confidence=MIN_DET_CONF,
        min_tracking_confidence=MIN_TRK_CONF,
    )


def extract_xyc_both_hands(
    result: Optional[Any],
    frame_shape: Tuple[int, int],  # (H, W)
    to_pixels: bool = False,
) -> np.ndarray:
    H, W = frame_shape

    def zeros21() -> np.ndarray:
        return np.zeros((21, 3), np.float32)

    left = right = None
    if result and result.multi_hand_landmarks and result.multi_handedness:
        items = []
        for lm, hd in zip(result.multi_hand_landmarks, result.multi_handedness):
            label = hd.classification[0].label
            score = float(hd.classification[0].score)
            items.append((label, score, lm))
        items.sort(key=lambda x: (x[0] != "Left", -x[1]))

        for label, score, lm in items[:2]:
            xy = np.array([(p.x, p.y) for p in lm.landmark], np.float32)
            if to_pixels:
                xy[:, 0] *= W
                xy[:, 1] *= H
            xyc = np.zeros((21, 3), np.float32)
            xyc[:, :2] = xy
            xyc[:, 2] = score
            if label == "Left":
                left = xyc
            else:
                right = xyc

    if left is None:
        left = zeros21()
    if right is None:
        right = zeros21()
    return np.concatenate([left, right], axis=0)


# ---------------------------------------------------------------------------
# WebRTC 엔드포인트
# ---------------------------------------------------------------------------

# 전역 변수로 recognizer 저장
_recognizer: Optional[Callable[[np.ndarray], str]] = None
_latest_sign: str = ""
_frame_count: int = 0
_start_time: float = 0


def set_recognizer(recognizer: Callable[[np.ndarray], str]) -> None:
    global _recognizer
    _recognizer = recognizer


def get_latest_sign() -> str:
    return _latest_sign


def get_stats() -> dict:
    elapsed = time.time() - _start_time if _start_time > 0 else 0
    fps = _frame_count / elapsed if elapsed > 0 else 0
    return {
        "frames_captured": _frame_count,
        "duration_sec": round(elapsed, 2),
        "avg_fps": round(fps, 2),
        "latest_sign": _latest_sign,
    }


# WebRTC 연결 주소 매핑 (외부에서 http://localhost:8000/offer 이렇게 접속)
# 클라이언트(WebRTC Peer)로부터 offer SDP(Session Description Protocol)를 받아 WebRTC 연결을 시작하는 엔드포인트
@app.post("/offer")
async def offer(request: Request):
    global _frame_count, _start_time, _latest_sign

    # 클라이언트에서 받은 offer SDP 를 저장
    data = await request.json()
    offer_sdp = data["sdp"]

    # Peer 연결 생성
    pc = RTCPeerConnection()

    # 데이터 수신부 콜백
    @pc.on("track")
    def on_track(track):
        # 영상 데이터를 받은 경우에 대한 처리
        if track.kind == "video":
            # 영상 데이터 프레임 처리 함수
            async def process_video():
                global _frame_count, _start_time, _latest_sign

                _frame_count = 0
                _start_time = time.time()
                _latest_sign = ""

                with create_hands() as hands:
                    while True:
                        try:
                            # track 에서 영상 프레임 받아 오기
                            frame: av.VideoFrame = await track.recv()

                            # 프레임 데이터 이미지 형식 변경
                            img = frame.to_ndarray(format="bgr24")  # OpenCV용으로 변환

                            # MediaPipe로 손 인식 처리
                            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            results = hands.process(rgb)

                            # 랜드마크 추출
                            landmarks = extract_xyc_both_hands(
                                results,
                                (img.shape[0], img.shape[1]),
                                TO_PIXELS,
                            )

                            # 수어 인식
                            if _recognizer:
                                try:
                                    _latest_sign = _recognizer(landmarks)
                                except Exception as rec_err:
                                    print(f"Recognizer error: {rec_err}")
                                    _latest_sign = ""

                            _frame_count += 1

                        except Exception as e:
                            print(f"Frame processing error: {e}")
                            break

            # 프레임 관련 처리 함수를 비동기로 실행
            asyncio.create_task(process_video())

    # 클라이언트의 offer SDP 를 수락
    await pc.setRemoteDescription(
        RTCSessionDescription(sdp=offer_sdp["sdp"], type=offer_sdp["type"])
    )

    # 서버에서 answer SDP 를 생성 후 설정
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    # 클라이언트에 서버의 answer SDP 를 응답으로 전송
    # 클라이언트는 이 정보를 통해 WebRTC 연결을 완료
    return {
        "sdp": {
            "sdp": pc.localDescription.sdp,
            "type": pc.localDescription.type,
        }
    }


@app.get("/stats")
async def stats():
    """현재 처리 상태 조회"""
    return get_stats()
