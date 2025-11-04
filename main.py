# main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import cv2, numpy as np, mediapipe as mp, base64, re, uvicorn
from pathlib import Path


BASE_DIR = Path(__file__).parent
DATASET_PATH = BASE_DIR / "assets" / "dataSet.txt"

if not DATASET_PATH.exists():

    raise FileNotFoundError(
        f"Dataset file not found: {DATASET_PATH}\n"
        f"→ 'signlanguage/assets/dataSet.txt' 에 파일을 두세요."
    )

file = np.genfromtxt(str(DATASET_PATH), delimiter=",", dtype=np.float32)
anglefile = np.array(file[:, :-1], dtype=np.float32)
labelfile = np.array(file[:, -1], dtype=np.float32)


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    # allow_origins=["http://localhost:3000"],
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5
)

# KNN 모델 로드
# file = np.genfromtxt(r"c:/Users/jun/Documents/aiprac/dataSet.txt", delimiter=",")
file = np.genfromtxt(str(DATASET_PATH), delimiter=",", dtype=np.float32)
anglefile = np.array(file[:, :-1], dtype=np.float32)
labelfile = np.array(file[:, -1], dtype=np.float32)
knn = cv2.ml.KNearest_create()
knn.train(anglefile, cv2.ml.ROW_SAMPLE, labelfile)

gesture = {
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


class FrameRequest(BaseModel):
    frame_data: str


class FrameResponse(BaseModel):
    detected_sign: str
    processed_frame: str


@app.post("/translate", response_model=FrameResponse)
async def recognize(req: FrameRequest):
    # Base64 → OpenCV
    img_str = re.sub("^data:image/.+;base64,", "", req.frame_data)
    img_bytes = base64.b64decode(img_str)
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    detected_sign = ""

    result = hands.process(imgRGB)
    if result.multi_hand_landmarks:
        for res in result.multi_hand_landmarks:
            joint = np.array([[lm.x, lm.y, lm.z] for lm in res.landmark])
            v1 = joint[
                [0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :
            ]
            v2 = joint[
                [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                :,
            ]
            v = v2 - v1
            v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]
            compareV1 = v[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18], :]
            compareV2 = v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], :]
            angle = np.degrees(np.arccos(np.einsum("nt,nt->n", compareV1, compareV2)))
            data = np.array([angle], dtype=np.float32)
            _, results, _, _ = knn.findNearest(data, 3)
            index = int(results[0][0])
            detected_sign = gesture.get(index, "")

            mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

    _, buffer = cv2.imencode(".jpg", img)
    img_base64 = base64.b64encode(buffer).decode("utf-8")

    return FrameResponse(
        detected_sign=detected_sign,
        processed_frame=f"data:image/jpeg;base64,{img_base64}",
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
