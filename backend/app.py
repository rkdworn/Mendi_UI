"""YOLO 신발 탐지 추론 서버 (FastAPI) + 정적 프런트(frontend/) 서빙.

    python -m backend.app [--weights weights/best.pt] [--host 127.0.0.1] [--port 8000]

엔드포인트
    GET  /health    서버 상태, 모델 이름, 클래스 목록
    GET  /classes   클래스 목록
    POST /predict   이미지(file) → 탐지 결과 JSON   (?conf=0.25&iou=0.7&imgsz=512)
    GET  /          frontend/index.html (같은 출처로 서빙)

환경변수: WEIGHTS, ALLOW_ORIGINS(쉼표 구분, 기본 *), MAX_UPLOAD_MB(기본 10), SERVE_WEB(기본 1)
프런트를 다른 곳(GitHub Pages 등)에 따로 올릴 때는 ALLOW_ORIGINS를 그 주소로 좁힐 것.
"""
import argparse
import io
import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import torch
from PIL import Image, ImageOps
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]  # 레포 루트 (backend/app.py 기준)
DEFAULT_WEIGHTS = ROOT / "weights" / "best.pt"
WEB_DIR = Path(__file__).resolve().parents[1] / "frontend"


@asynccontextmanager
async def lifespan(app: FastAPI):
    weights = Path(os.environ.get("WEIGHTS", DEFAULT_WEIGHTS))
    if not weights.is_absolute():
        weights = ROOT / weights
    if not weights.exists():
        raise RuntimeError(f"가중치 파일이 없습니다: {weights}")
    app.state.weights = weights
    app.state.device = 0 if torch.cuda.is_available() else "cpu"
    app.state.device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    app.state.model = YOLO(str(weights))
    app.state.names = {int(k): v for k, v in app.state.model.names.items()}
    app.state.lock = threading.Lock()  # GPU 추론은 한 번에 하나씩
    # 첫 요청 지연을 없애려고 더미 이미지로 워밍업
    app.state.model.predict(Image.new("RGB", (512, 512)), imgsz=512, device=app.state.device, verbose=False)
    yield


app = FastAPI(title="Mendi shoe detection API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in os.environ.get("ALLOW_ORIGINS", "*").split(",")],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": app.state.weights.parent.parent.name,
        "weights": app.state.weights.name,
        "device": app.state.device_name,
        "classes": [app.state.names[i] for i in sorted(app.state.names)],
    }


@app.get("/classes")
def classes():
    return {"classes": [{"id": i, "name": n} for i, n in sorted(app.state.names.items())]}


@app.post("/predict")
def predict(
    file: UploadFile = File(...),
    conf: float = Query(0.25, ge=0.01, le=1.0),
    iou: float = Query(0.7, ge=0.1, le=1.0),
    imgsz: int = Query(512, ge=320, le=1280),
):
    max_bytes = int(os.environ.get("MAX_UPLOAD_MB", "10")) * 1024 * 1024
    data = file.file.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise HTTPException(413, f"파일이 너무 큽니다 (최대 {max_bytes // 1024 // 1024}MB)")
    try:
        img = Image.open(io.BytesIO(data))
        img = ImageOps.exif_transpose(img).convert("RGB")  # 폰 사진의 EXIF 회전 반영 → 브라우저 표시와 일치
    except Exception:
        raise HTTPException(400, "이미지를 읽을 수 없습니다 (jpg/png 등 이미지 파일인지 확인)")

    w, h = img.size
    t0 = time.perf_counter()
    with app.state.lock:
        r = app.state.model.predict(
            img, conf=conf, iou=iou, imgsz=imgsz, max_det=100, device=app.state.device, verbose=False
        )[0]
    ms = (time.perf_counter() - t0) * 1000

    dets = []
    for c, s, (x1, y1, x2, y2) in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(), r.boxes.xyxy.tolist()):
        dets.append({
            "class_id": int(c),
            "label": app.state.names[int(c)],
            "conf": round(float(s), 4),
            "box": {"x1": round(x1, 1), "y1": round(y1, 1), "x2": round(x2, 1), "y2": round(y2, 1)},
            "box_norm": {"x1": x1 / w, "y1": y1 / h, "x2": x2 / w, "y2": y2 / h},
        })
    dets.sort(key=lambda d: -d["conf"])
    counts = {}
    for d in dets:
        counts[d["label"]] = counts.get(d["label"], 0) + 1

    return {
        "filename": file.filename,
        "image": {"width": w, "height": h},
        "params": {"conf": conf, "iou": iou, "imgsz": imgsz},
        "inference_ms": round(ms, 1),
        "count": len(dets),
        "counts": counts,
        "top": dets[0] if dets else None,
        "detections": dets,
    }


# API 라우트 뒤에 마운트해야 /health 등이 가려지지 않는다
if os.environ.get("SERVE_WEB", "1") != "0" and WEB_DIR.exists():
    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", help="가중치 경로 (기본: weights/best.pt)")
    ap.add_argument("--host", default="127.0.0.1", help="외부 접속 허용은 0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    if args.weights:
        os.environ["WEIGHTS"] = args.weights

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
