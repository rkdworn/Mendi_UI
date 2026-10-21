"""ngrok 연결 직후 공개 주소가 준비될 때까지 확인한다."""
from __future__ import annotations

import time
import requests


def wait_for_public_health(url, expected_health, timeout=60):
    deadline = time.monotonic() + timeout
    last_error = "공개 주소 준비 대기 중"
    while time.monotonic() < deadline:
        remaining = deadline - time.monotonic()
        try:
            response = requests.get(
                url.rstrip("/") + "/health",
                headers={"ngrok-skip-browser-warning": "true", "Cache-Control": "no-cache"},
                timeout=min(10, max(.1, remaining)))
            if response.status_code == 200:
                data = response.json()
                if (data.get("status") == "ok"
                        and data.get("artifact_sha256") == expected_health["artifact_sha256"]
                        and data.get("thresholds") == expected_health["thresholds"]):
                    return data
                last_error = "공개 주소의 응답이 로컬 모델과 다릅니다."
            else:
                last_error = "HTTP " + str(response.status_code)
                code = response.headers.get("ngrok-error-code")
                if code:
                    last_error += " / ngrok " + str(code)
        except (requests.RequestException, ValueError) as error:
            last_error = type(error).__name__
        remaining = deadline - time.monotonic()
        if remaining > 0:
            time.sleep(min(2, remaining))
    raise RuntimeError("로컬 서버는 준비됐지만 공개 주소 연결 확인에 실패했습니다: " + last_error
                       + ". ngrok 도메인과 실행 상태를 확인하세요. 주소: " + url)
