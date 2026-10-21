# MEND AI · 신발 손상 분석

기존 Mendi_UI 디자인을 재사용한 최종 3모델 앙상블 웹입니다.

- 웹: https://rkdworn.github.io/Mendi_UI/
- **Colab 실행:** https://colab.research.google.com/github/rkdworn/Mendi_UI/blob/main/mendi_serve.ipynb

## 빠른 실행

1. 제공된 `mendai_final_bundle.zip`을 Google Drive **내 드라이브 최상위**에 업로드합니다. 가중치는 GitHub에 올리지 않습니다.
2. Colab을 열고 GPU 런타임을 선택합니다. L4/A100은 BF16, T4는 FP32로 추론합니다.
3. 1 → 4번 셀을 실행합니다. ngrok 토큰은 Colab 비밀 키 `NGROK_AUTHTOKEN` 또는 비공개 입력창에 넣습니다. 고정 도메인이 있다면 `NGROK_DOMAIN`도 추가합니다.
4. 마지막 셀이 출력하는 **교수님 제출 주소**를 제출합니다. 웹 화면과 평가 API를 같은 주소에서 사용할 수 있습니다.

GitHub Pages를 쓰는 경우 화면의 서버 연결에 Colab 주소를 입력하거나, 노트북이 출력하는 `?api=` 링크를 사용합니다. Colab 런타임이 꺼지면 서버도 멈춥니다.

## 확정 모델

| 모델 | 추론 | 비중 |
|---|---|---:|
| DINOv3 EX-06 LoRA | 원본 + 좌우 반전 점수 평균 | 50% |
| RF-DETR-S LoRA 1024 | 원본 | 25% |
| YOLO26m A3 | 다중 크기 + 반전 TTA | 25% |

앙상블 점수에 안감 0.54, 밑창 0.40을 적용해 normal / heel / outsole / both로 판정합니다. 손상 유무를 판정하며 위치 박스를 표시하는 서비스는 아닙니다. 원본은 기존 데이터 생성 방식으로 1024px에 맞춰 처리합니다.

`GET /health`는 준비 상태, `POST /predict`는 multipart `file`의 판정을 JSON으로 반환합니다. 업로드 크기는 최대 20MiB입니다. 교수님 프로그램의 예시용 매개변수는 고정 판정 기준을 바꾸지 않습니다. 일반 JSON 응답으로 호환되며 최종 제출 스키마는 교수님 안내를 따릅니다.

## 검증 범위

로컬 RTX 5060에서는 교수님 코드의 HTTP/CSV 함수로 val 1,319장 전송을 완료했고 기존 최종 판정과 혼동행렬이 일치했습니다. 균형 정확도 88.19%, 전체 정확도 86.81%입니다. Colab GPU와 패키지 조합, 특히 T4 FP32에서는 점수 차이가 있을 수 있습니다.

가중치·사진·토큰은 저장소에 포함하지 않습니다. 서버 시작 때 가중치 해시를 검사합니다.

참고: [pyngrok Colab 연동](https://pyngrok.readthedocs.io/en/stable/integrations.html), [PyTorch BF16 지원 확인](https://docs.pytorch.org/docs/stable/generated/torch.cuda.is_bf16_supported.html).
