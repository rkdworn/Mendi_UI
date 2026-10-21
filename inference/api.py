"""사진 한 장을 확정한 50:25:25 앙상블로 판정한다. 임계값은 고정한다."""
from __future__ import annotations

from contextlib import asynccontextmanager
import hashlib
import io
import json
import logging
import os
from pathlib import Path
import threading
import time
from unittest.mock import patch

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
import torch

from .data import preprocess
from .model import DamageDINO

LOG = logging.getLogger("uvicorn.error")
ORDER = ("normal", "heel", "outsole", "both")
SCORE_NAMES = ("heel_lining", "outsole")
WEIGHTS = (.5, .25, .25)
THRESHOLDS = (.54, .40)
MAX_BYTES = 20 * 1024 * 1024


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def normalized_photo(payload):
    """build_damage.save_image와 같은 draft/EXIF/thumbnail/JPEG95 변환.

    이미 1024 이하이고 회전 태그가 없으면 원본 바이트를 보존한다.
    학습용 사본을 다시 JPEG 압축하지 않는다.
    """
    with Image.open(io.BytesIO(payload)) as image:
        original_size = image.size
        turned = image.getexif().get(274, 1) != 1
        changed = turned or max(image.size) > 1024
        if changed:
            image.draft("RGB", (1024, 1024))
            oriented = ImageOps.exif_transpose(image).convert("RGB")
            if max(oriented.size) > 1024:
                oriented.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
            buffer = io.BytesIO()
            oriented.save(buffer, format="JPEG", quality=95)
            payload = buffer.getvalue()
    with Image.open(io.BytesIO(payload)) as image:
        image = image.convert("RGB")
        return image, payload, {"original_size": list(original_size),
                                "model_image_size": list(image.size), "resaved": changed}


def valid_scores(values):
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (2,) or not np.isfinite(array).all() or ((array < 0) | (array > 1)).any():
        raise ValueError("모델 점수 범위/형식 오류")
    return array


class Ensemble:
    def __init__(self, config):
        self.config = config
        self.lock = threading.Lock()
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU가 필요합니다. CPU 대체 추론은 실행하지 않습니다.")
        if config["weights"] != list(WEIGHTS) or config["thresholds"] != list(THRESHOLDS):
            raise ValueError("확정 앙상블 비중/임계값 불일치")
        self.fingerprints = {}
        for name, item in config["artifacts"].items():
            actual = sha256(item["path"])
            if actual != item["sha256"]:
                raise ValueError("가중치 변경: " + name)
            self.fingerprints[name] = actual
        torch.set_num_threads(2)
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        torch.backends.cudnn.benchmark = False
        torch.set_float32_matmul_precision("highest")
        self.memory_budget = int(config.get("memory_budget_mib", 6656))
        torch.cuda.set_per_process_memory_fraction(
            self.memory_budget * 1024**2 / torch.cuda.get_device_properties(0).total_memory, 0)
        self.use_bf16 = torch.cuda.is_bf16_supported(including_emulation=False)

        origin = Path(config["dino_run"])
        protocol = json.loads((origin / "protocol.json").read_text(encoding="utf-8"))
        self.dino_size = protocol["config"]["input_size"]
        checkpoint = torch.load(config["artifacts"]["dino"]["path"], map_location="cpu", weights_only=False)
        self.dino = DamageDINO(protocol["config"], config["backbone_path"]).to("cuda")
        self.dino.load_trainable(checkpoint["model"])
        self.dino.eval()
        del checkpoint
        LOG.info("DINOv3 EX-06 loaded")

        import rfdetr
        detr_weights = Path(config["artifacts"]["detr"]["path"]).resolve()

        def local_only(path, *args, **kwargs):
            if Path(path).resolve() != detr_weights or not detr_weights.is_file():
                raise RuntimeError("추가 가중치 다운로드 차단: " + str(path))

        with patch("rfdetr.detr.download_pretrain_weights", local_only), \
             patch("rfdetr.models.weights.download_pretrain_weights", local_only):
            self.detr = rfdetr.RFDETRSmall(pretrain_weights=str(detr_weights), resolution=1024,
                                           num_classes=2, backbone_lora=True, device="cuda")
        LOG.info("RF-DETR-S LoRA loaded")
        from ultralytics import YOLO, settings
        settings.update({"sync": False})
        self.yolo = YOLO(config["artifacts"]["yolo"]["path"])
        if {int(k): v for k, v in self.yolo.names.items()} != {0: "outsole", 1: "heel_lining"}:
            raise ValueError("YOLO 클래스 순서 오류")
        LOG.info("YOLO26m A3 loaded")

    def predict(self, payload):
        # 파일명이나 외부 threshold 매개변수를 추론에 사용하지 않는다.
        image, encoded, image_info = normalized_photo(payload)
        with self.lock, torch.inference_mode():
            started = time.perf_counter()
            views = []
            for view in (image, ImageOps.mirror(image)):
                pixels = preprocess(view, self.dino_size).unsqueeze(0).to("cuda")
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.use_bf16):
                    logits = self.dino(pixels)
                views.append(logits.float().sigmoid().cpu().numpy()[0])
            # 학습 당시 float32 원본/반전 확률 평균을 그대로 보존한다.
            dino = valid_scores((views[0] + views[1]) / 2)
            del pixels, logits, views
            torch.cuda.empty_cache()

            detections = self.detr.predict(image, threshold=.001, include_source_image=False)
            names = detections.data.get("class_name")
            if names is None or len(names) != len(detections):
                raise ValueError("RF-DETR 클래스 이름 누락")
            detr_values = {name: 0.0 for name in SCORE_NAMES}
            for name, score in zip(names, detections.confidence):
                name = str(name)
                if name == "__background__":
                    continue
                if name not in detr_values:
                    raise ValueError("RF-DETR의 알 수 없는 클래스: " + name)
                detr_values[name] = max(detr_values[name], round(float(score), 5))
            detr = valid_scores([detr_values[name] for name in SCORE_NAMES])
            del detections
            torch.cuda.empty_cache()

            # 팀 CSV 생성 때의 OpenCV JPEG 디코딩 경로를 따른다.
            import cv2
            bgr = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
            if bgr is None:
                raise ValueError("YOLO 이미지 디코딩 오류")
            result = self.yolo.predict(bgr, imgsz=1024, conf=.001, iou=.7, max_det=300,
                                       device=0, batch=1, augment=True, verbose=False, save=False)[0]
            yolo_values = {0: 0.0, 1: 0.0}
            for class_id, score in zip(result.boxes.cls.int().tolist(), result.boxes.conf.tolist()):
                if class_id not in yolo_values:
                    raise ValueError("YOLO의 알 수 없는 클래스")
                yolo_values[class_id] = max(yolo_values[class_id], float(score))
            # 기준 CSV의 네 자리 저장 규칙을 적용한 뒤 앙상블한다.
            yolo = valid_scores([float(f"{yolo_values[k]:.4f}") for k in (1, 0)])
            combined = WEIGHTS[0] * dino + WEIGHTS[1] * detr + WEIGHTS[2] * yolo
            verdict = ORDER[int(combined[0] >= THRESHOLDS[0]) + 2 * int(combined[1] >= THRESHOLDS[1])]
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            peak = torch.cuda.max_memory_reserved() / 1024**2
            if peak > self.memory_budget:
                raise RuntimeError("GPU 예약 메모리 한도 초과")
            del result
            torch.cuda.empty_cache()
        return {"prediction": verdict,
                "scores": dict(zip(SCORE_NAMES, combined.tolist())),
                "component_scores": {name: dict(zip(SCORE_NAMES, score.tolist()))
                                     for name, score in (("dino", dino), ("detr", detr), ("yolo", yolo))},
                "thresholds": dict(zip(SCORE_NAMES, THRESHOLDS)),
                "weights": dict(zip(("dino", "detr", "yolo"), WEIGHTS)),
                "input_sha256": hashlib.sha256(payload).hexdigest(),
                "image": image_info, "inference_seconds": elapsed,
                "peak_reserved_mib": peak, "dino_precision": "bfloat16" if self.use_bf16 else "float32"}


@asynccontextmanager
async def lifespan(app):
    config = json.loads(Path(os.environ["MENDAI_DEPLOYMENT"]).read_text(encoding="utf-8"))
    app.state.ensemble = Ensemble(config)
    yield
    del app.state.ensemble
    torch.cuda.empty_cache()


app = FastAPI(title="MEND AI final ensemble", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"], allow_headers=["*"], allow_credentials=False)


@app.get("/health")
def health():
    ensemble = app.state.ensemble
    return {"status": "ok", "model": "EX-06-TTA + RF-DETR-S-LoRA + YOLO26m-A3-TTA",
            "weights": list(WEIGHTS), "thresholds": dict(zip(SCORE_NAMES, THRESHOLDS)),
            "artifact_sha256": ensemble.fingerprints, "device": torch.cuda.get_device_name(0)}


@app.post("/predict")
def predict(file: UploadFile = File(...)):
    payload = file.file.read(MAX_BYTES + 1)
    if not payload or len(payload) > MAX_BYTES:
        raise HTTPException(413, "빈 파일이거나 20MiB를 넘는 사진입니다.")
    try:
        return app.state.ensemble.predict(payload)
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError) as exc:
        raise HTTPException(400, "사진을 읽을 수 없습니다.") from exc
    except Exception as exc:
        LOG.exception("모델 추론 실패")
        raise HTTPException(500, "모델 추론에 실패했습니다. 서버 로그를 확인하세요.") from exc


# API ?? ?? ????. Colab? ?? ?? ??? ?? ?? API? ?? ????.
WEB_DIR = Path(__file__).resolve().parents[1] / "frontend"
if WEB_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")
