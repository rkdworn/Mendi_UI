"""검증한 모델 묶음을 사용해 웹과 API를 함께 실행한다."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

EXPECTED = {
    "dino": "558358de05eda5cac02c72a0c2dfe8211398fec085c95599e42c4b80658410a6",
    "backbone": "9a21ac3df0c63839d62612dda6f454d816c25611cc7a52966ed5a5a94921dc8b",
    "detr": "bfe2d85f99dba8a22c60709df81a35a93421e13e4f4dbaaf2597d16592a077e8",
    "yolo": "ca49a9de1926f5daef0c9ab7bb77d80f13a195eed5fa0068048c120602259918",
}


def deployment(bundle, runtime, memory_budget):
    bundle, runtime = Path(bundle).resolve(), Path(runtime).resolve()
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    artifacts = {}
    for name, expected in EXPECTED.items():
        entry = manifest["artifacts"][name]
        path = (bundle / entry["path"]).resolve()
        path.relative_to(bundle)
        if entry["sha256"] != expected or not path.is_file():
            raise ValueError("확정 가중치가 아닙니다: " + name)
        artifacts[name] = {"path": str(path), "sha256": expected}
    config = {"artifacts": artifacts, "dino_run": str(bundle / "dino"),
              "backbone_path": str(Path(artifacts["backbone"]["path"]).parent),
              "weights": [.5, .25, .25], "thresholds": [.54, .40], "memory_budget_mib": memory_budget}
    runtime.mkdir(parents=True, exist_ok=True)
    path = runtime / "deployment.json"
    path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--runtime", default=".runtime")
    parser.add_argument("--memory-budget-mib", type=int, default=6656)
    args = parser.parse_args()
    runtime = Path(args.runtime).resolve()
    for folder in ("yolo", "matplotlib", "torch"):
        (runtime / folder).mkdir(parents=True, exist_ok=True)
    os.environ.update({"YOLO_CONFIG_DIR": str(runtime / "yolo"), "MPLCONFIGDIR": str(runtime / "matplotlib"),
                       "TORCH_HOME": str(runtime / "torch"), "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                       "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "TOKENIZERS_PARALLELISM": "false"})
    os.environ["MENDAI_DEPLOYMENT"] = str(deployment(args.bundle, runtime, args.memory_budget_mib))
    import uvicorn
    uvicorn.run("inference.api:app", host=args.host, port=args.port, access_log=False)


if __name__ == "__main__":
    main()
