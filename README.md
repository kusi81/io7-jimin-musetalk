# IO7 Sales — RunPod Serverless MuseTalk 1.5 워커 배포 가이드

김지민 과장의 **1080p Hub 제스처 클립(`explain_hands_v2.mp4`, `greet_wave_v2.mp4`, `show_pouch_sample_v2.mp4`, `idle_listen_v2.mp4`)** 위에 Gemini 3.8 Live 음성을 실시간 25fps로 립싱크하는 RunPod Serverless 워커입니다.

## 핵심 비용·성능 최적화 포인트
- **Speaking-Only 호출**: 경청·미소·갤럭시 폰 터치 구간은 로컬 1080p 클립을 무료(\$0)로 재생하고, 실제 발화 순간에만 RunPod Serverless를 호출합니다.
- **Pre-bake + VRAM 캐싱**: 도커 빌드 시 얼굴 좌표·크롭을 사전 추출하고, 워커 기동 시 VAE Latent를 VRAM에 1회 적재해 청크당 추론 지연을 **150ms 이하**로 줄입니다.
- **Face-Patch 합성**: 1080p 전체를 재인코딩하지 않고 입/턱 주변 `256×256` WebP 패치 + 1920×1080 좌표만 전송해 원본 1080p 보케·손·파우치 선명도를 100% 유지합니다.
- **Pre-warm**: IO7 Sales에서 고객이 세션을 여는 순간 백엔드가 자동으로 `{"action": "warmup"}` 핑을 보내 콜드 스타트를 숨깁니다.

---

## 1단계: 김지민 과장 1080p 클립 복사 및 도커 빌드

```powershell
cd C:\Users\Kim\.gemini\antigravity\scratch\io7-sales-agent\lipsync-worker

# 1) 최신 김지민 과장 1080p 클립을 assets/ 폴더로 복사
.\sync-assets.ps1

# 2) 도커 이미지 빌드 & Docker Hub (또는 GHCR) 푸시
docker build -t <YOUR_DOCKERHUB_ID>/io7-jimin-musetalk:v1 .
docker push <YOUR_DOCKERHUB_ID>/io7-jimin-musetalk:v1
```

> **Tip**: 로컬 PC에 GPU나 Docker가 없다면 GitHub 리포지토리에 `lipsync-worker/` 폴더를 올리고 RunPod Console의 **Deploy from GitHub Repo** 기능으로 RunPod 클라우드에서 바로 빌드할 수 있습니다.

---

## 2단계: RunPod Serverless Endpoint 생성 (권장 스펙)

1. [RunPod Serverless Console](https://www.runpod.io/console/serverless) → **New Endpoint** 클릭
2. **Container Image**: `<YOUR_DOCKERHUB_ID>/io7-jimin-musetalk:v1`
3. **GPU 선택 (Flex Workers)**:
   - 가성비 권장: **NVIDIA L4 (24GB)** — 초당 약 \$0.00021 (분당 약 18원, 대기 시 \$0)
   - 최고 속도: **NVIDIA RTX 4090 (24GB)** — 초당 약 \$0.00037 (분당 약 31원, 대기 시 \$0)
4. **Workers 설정**:
   - **Active Workers (Min)**: `0` *(대기 비용 \$0)*
   - **Max Workers**: `2` *(초기 PoC 기준, 100~200명 확장 시 20~50으로 증설)*
   - **Idle Timeout**: `20`초 *(대화 중 턴 사이에 워커가 식지 않도록 유지)*
   - **FlashBoot**: **ON** *(콜드 스타트 2~5초 단축)*
5. **Container Disk**: `25 GB`

---

## 3단계: IO7 Sales `.env` 연동

생성된 Endpoint ID와 RunPod API Key를 `io7-sales-agent/.env`에 추가한 뒤 서버를 재시작하면 즉시 활성화됩니다.

```env
RUNPOD_API_KEY=rpa_xxxxxxxxxxxxxxxxxxxxxxxx
RUNPOD_ENDPOINT_ID=xxxxxxxxxxxxxx
LIPSYNC_GPU_TIER=L4
LIPSYNC_CHUNK_MS=480
LIPSYNC_TIMEOUT_MS=2800
```

- 키가 비어 있거나 RunPod 호출이 지연될 때는 **자동으로 기존 1080p Hub 클립 모드로 무중단 폴백**되므로 영업 통화가 절대 끊기지 않습니다.
