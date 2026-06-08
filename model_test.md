# TensorRT 최적화 작업 기록

MuseTalk 아바타 생성 파이프라인에 TensorRT(TRT) 8.6.1 가속을 적용하는 과정에서 시도한 것들과 결과를 기록합니다.

---

## 환경

- GPU: NVIDIA RTX 4060 (8GB VRAM, SM89 Ada Lovelace)
- TensorRT: 8.6.1.6
- PyTorch: 2.x (CUDA 11.8)
- cuDNN: 8.7.0 (TRT 링크 버전 8.9.0과 미스매치 — 빌드 시 경고 발생하지만 동작)

---

## 목표

| 단계 | 내용 |
|------|------|
| UNet 추론 | 배치 16 기준 가능한 빠르게 |
| VAE 디코딩 | 16프레임 기준 가능한 빠르게 |
| 품질 | 시각적 왜곡(픽셀 깨짐, 색상 노이즈) 없어야 함 |

---

## 해결된 문제들

### 1. TRT 추론 30~100초 → ~200ms

**원인**  
PyTorch UNet 모델이 GPU에 남아있는 상태에서 TRT가 추론 중 VRAM 부족 → 스왑 발생

**해결**  
TRT UNet 엔진 로드 후 PyTorch 모델을 CPU로 내림
```python
# ml_manager.py
if trt_unet is not None:
    unet.model = unet.model.cpu()
    torch.cuda.empty_cache()
```

---

### 2. 첫 번째 추론 3~4초 콜드스타트

**원인**  
서버 시작 후 첫 TRT 호출 시 CUDA 커널 JIT 컴파일

**해결**  
TRT 엔진 로드 직후 더미 입력으로 웜업 1회 실행
```python
# ml_manager.py
with torch.inference_mode():
    _d = torch.zeros(16, 8, 32, 32, dtype=torch.float16, device=device)
    _t = torch.tensor([0], device=device)
    _a = torch.zeros(16, 5, 384, dtype=torch.float16, device=device)
    trt_unet(_d, _t, encoder_hidden_states=_a)
torch.cuda.empty_cache()
```

---

### 3. PyTorch VAE 디코딩 후 TRT UNet 다시 느려짐 (600ms → 정상 200ms)

**원인**  
PyTorch VAE가 VRAM 캐시 ~2.9GB를 점유한 채로 반환하지 않음 → TRT 다음 추론에 공간 부족

**해결**  
VAE 디코딩 직후 캐시 반환
```python
# stream_inference.py
torch.cuda.empty_cache()
```

---

### 4. TRT VAE → 포기, PyTorch 프레임 보간으로 대체

**시도한 것들**
- Dynamic shape (min=1, opt=8, max=16): 4~5초로 여전히 느림
- FP16 GPU ONNX 내보내기: 12~18초 + GroupNorm 정밀도 문제로 영상 왜곡 발생
- FP32 CPU ONNX + 정적 shape (batch=16): 빌드는 성공, 추론 여전히 느림

**결론**  
TRT VAE는 RTX 4060에서 PyTorch 대비 이점 없음. 비활성화.

**대안**  
짝수 인덱스 프레임만 VAE 디코딩 후 홀수 프레임은 선형 보간 → 약 400ms (2배 속도 향상)
```python
# stream_inference.py
key_latents = latents[::2]
key_decoded = vae.vae.decode(key_latents.to(dtype)).sample
interp = (key_decoded[:-1] + key_decoded[1:]) / 2
```

---

### 5. 목소리 드롭다운 비어있음

**원인**  
프론트엔드에서 `/api/voices` 요청 시 인증 토큰 미전송

**해결**  
토큰 포함 + 실패 시 하드코딩 폴백 추가 (`frontend/js/app.js`)

---

## NaN 문제 — 영상 왜곡의 근본 원인

TRT UNet 추론 출력에 NaN이 포함되면 VAE 디코딩 결과에 색상 깨짐, 픽셀 노이즈가 발생.

### 시도 1 — dtype 불일치 의심
FP32 입력을 FP16 바인딩 엔진에 전달하는 문제로 의심 → dtype 맞춰도 NaN 지속

### 시도 2 — FP32 ONNX 내보내기 (model.float())
```
에러: OutOfMemory (2GB 요청 실패)
에러: Could not find any implementation for ForeignNode[attention/Transpose...]
```
실패 원인
- PyTorch 모델이 VRAM에 남은 채 TRT 빌드 → VRAM 부족
- `FP16 플래그 + FP32 ONNX` 조합: TRT 8.6이 FP32 attention 커널 미지원

### 시도 3 — SDPA 패치에서 softmax만 FP32 캐스트
```python
attn = (q.float() @ k.float().transpose(-2,-1)) * scale  # ONNX에 Cast 노드 삽입
attn = torch.softmax(attn, dim=-1).to(q.dtype)
```
TRT FP16 플래그 빌드 시 Cast 노드를 무시하고 FP16으로 통합 → NaN 지속

### 시도 4 — PREFER_PRECISION_CONSTRAINTS + Softmax FP32 강제
```python
cfg.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
layer.precision = trt.DataType.FLOAT  # Softmax 레이어
```
`PREFER`는 TRT가 선택적으로 무시 가능 → NaN 지속

### 시도 5 — OBEY_PRECISION_CONSTRAINTS + Softmax 32개 FP32
빌드 로그: `FP32 강제: Softmax=32, Normalization=0`  
Softmax 32개 강제 적용 성공, 그러나 **NaN 지속**

→ **확정**: NaN의 원인은 Softmax가 아닌 **LayerNorm**

TRT 빌드 경고 (처음부터 있었음):
```
[W] Detected layernorm nodes in FP16
[W] Running layernorm after self-attention in FP16 may cause overflow
```

LayerNorm은 TRT가 **ForeignNode**(퓨즈드 커널)로 묶어버려 `layer.precision` API 접근 불가.  
`Normalization=0`: `trt.LayerType.NORMALIZATION`으로 검색해도 아무것도 검출되지 않음.

---

## 최종 결론

### FP32 TRT — 빌드 성공, 그러나 속도 미흡

빌드 결과: 1719.7MB, NaN 없음 (`nan=False`, `range=[-6.163, 4.071]` 정상 출력)  
실제 추론 속도: **11~31초/배치** (FP16 TRT 200ms 대비 60~150배 느림)

원인: RTX 4060 FP32 처리량(15 TFLOPS, CUDA 코어) vs FP16(61 TFLOPS, 텐서 코어) — 4배 이상 차이.  
결론: **FP32 TRT는 RTX 4060에서 실사용 불가**.

---

### 최종 채택: PyTorch FP16

FP16/FP32 TRT 모두 실패 후 PyTorch FP16 + Flash Attention으로 복귀.

| 지표 | FP16 TRT | FP32 TRT | **PyTorch FP16 (채택)** |
|------|---------|---------|------------------------|
| UNet/배치 | 200ms | 11~31초 | **300~450ms** |
| VAE/배치 | 400ms | 3~6초 | **500~800ms** |
| 첫 청크 | ~5.6초 | 1299초 | **11.8초** |
| NaN 여부 | ✗ 있음 | ✓ 없음 | **✓ 없음** |
| 영상 왜곡 | ✗ 있음 | ✓ 없음 | **✓ 없음** |

**PyTorch FP16이 채택된 이유:**
- Flash Attention이 수치적으로 안정한 연산 → NaN 없음
- LayerNorm / Softmax 오버플로우 없음
- 300~450ms/배치는 실시간 스트리밍에 충분
- TRT 엔진 빌드 및 관리 불필요

**설정 (`ml_manager.py`):**
```python
trt_unet, trt_vae = _load(device, unet_path="", vae_path="")
```
UNet은 GPU FP16으로 유지, TRT 없이 PyTorch로 직접 추론.

---

## 핵심 교훈

| 항목 | 내용 |
|------|------|
| TRT FP16 플래그 | FP16 플래그 활성화 시 TRT가 LayerNorm을 FP16 ForeignNode로 퓨즈 → overflow |
| ForeignNode | TRT가 여러 op을 하나로 퓨즈한 커널, `layer.precision` API로 접근 불가 |
| OBEY_PRECISION_CONSTRAINTS | 표준 레이어(Softmax 등)에만 효과 있음, ForeignNode에는 무효 |
| VRAM 관리 | TRT 빌드 중에는 PyTorch 모델을 완전히 삭제해야 OOM 방지 |
| `torch.cuda.empty_cache()` | PyTorch 캐시를 반환하지 않으면 TRT 추론 시 VRAM 부족 발생 |
| TRT VAE | RTX 4060 기준 UNet 대비 TRT 이점 미미, 프레임 보간이 더 효율적 |

---

---

## TRT 이후 — 스트리밍 버퍼링 제거 작업

TRT 포기 후 PyTorch FP16 기반에서 실시간 스트리밍 버퍼링을 없애기 위해 진행한 작업.

---

### 문제 1 — WDDM이 GPU 메모리를 회수해 두 번째 요청이 느려짐

**원인**  
Windows WDDM(Windows Display Driver Model)은 GPU가 ~2초 이상 유휴 상태가 되면 VRAM을 OS에 반환.  
TTS 처리 중 GPU가 쉬는 동안 메모리 회수 → 다음 추론 첫 배치에서 재로딩 오버헤드 발생.

**해결**  
NVIDIA 제어판 → 3D 설정 → 전원 관리 모드 → **최대 성능 선호** 설정  
(WDDM의 공격적인 메모리 회수 억제)

---

### 문제 2 — 마지막 배치(소배치) UNet/VAE 3초 이상 느림

**원인**  
배치 크기가 16이 아닌 소배치(B=3 등)가 마지막에 오면 CUDA 알고리즘 탐색(컨볼루션 알고리즘 선택) 오버헤드 발생.  
예: B=3 마지막 배치 3083ms (정상 배치 850ms 대비 3.6배)

**해결**  
서버 시작 시 소배치 크기로도 미리 웜업 실행 → CUDA 커널 캐시에 사전 등록
```python
# ml_manager.py
for b in [4, 8, 12]:
    _w = torch.zeros(b, 5, 384, dtype=weight_dtype, device=device)
    _l = torch.zeros(b, 8, 32, 32, dtype=weight_dtype, device=device)
    _o = unet.model(_l, timesteps, encoder_hidden_states=pe(_w)).sample
    vae.vae.decode(_o[::2].to(vae.vae.dtype))
    torch.cuda.synchronize()
```
결과: B=3 마지막 배치 3083ms → **569ms**

---

### 문제 3 — CUDA 스트림 파이프라인 → 포기

**시도**  
UNet(텐서 코어)과 VAE(CUDA 코어)를 별도 CUDA 스트림에서 동시 실행하면 두 연산이 겹쳐서 처리될 것이라 기대.

**결과**  
두 스트림이 동시에 VRAM을 점유 → WDDM이 메모리 회수 → 109.5초 (정상 9초 대비 12배 느림)

**원인**  
두 스트림이 서로 다른 VRAM 블록을 동시에 보유하면서 메모리 압박 발생 → WDDM이 회수 트리거.  
최대 성능 모드에서도 동시 점유 패턴에서는 회수 발생.

**결론**  
순차 실행(현재 방식)이 유일하게 안정적.

---

### 문제 4 — 재생 시작 전 2~3초 대기 ("버퍼링 중..." 표시)

**원인**  
서버 생성 속도 = 0.75배속 (배치 840ms로 640ms 영상 생성).  
재생 중 버퍼가 드레인되므로 초기에 충분한 버퍼를 쌓아야 중간 멈춤 방지.

**최적화**
1. TTS 직후 불필요한 GPU 웜업 블록 제거 (`generation.py`) → 첫 청크 **0.4~0.5초 단축**
2. `RESUME_THRESHOLD` 4.0초 → **3.0초** 축소 → 재생 시작 ~1초 단축  
   (측정 총 드레인 2.3~2.5초, 마진 0.5~0.7초로 안전)

**결과**  
첫 청크: 11.8초 → **2.1~2.9초**  
재생 시작까지: ~7초 → **~5초**

---

### 문제 5 — 첫 글자("오") 잘림

**원인**  
HTML `<video autoplay>` 속성으로 첫 청크 도착 즉시 자동 재생 시작.  
`monitorBuffer`가 `ahead < 0.3s` 감지 → 즉시 일시정지.  
3초 버퍼 쌓인 후 재개 시 `currentTime`이 이미 첫 음절 이후로 밀려있음.

**해결**  
`index.html`에서 `autoplay` 제거. 재생 시작은 `monitorBuffer`의 `play()` 호출로만 제어.

---

### 문제 6 — 영상 끝부분에서 멈추고 완료가 안 됨

**원인**  
스트리밍이 끝날 무렵 남은 영상이 0.3초 미만 → `monitorBuffer`가 일시정지.  
재개 조건: `ahead >= RESUME_THRESHOLD(3.0초)` → 남은 영상이 3초 이하면 **영원히 재개 불가**.

**해결**  
스트림 완료(`streamDone = true`) 후에는 `ahead > 0` (1프레임이라도 있으면) 즉시 재개.
```javascript
// app.js
} else if (videoEl.paused && playAllowed && ahead > 0) {
  // 스트림 완료 후: 남은 프레임 있으면 즉시 재생
  videoEl.play().catch(() => {});
  statusEl.textContent = '완료!';
}
```
스트림 완료 순간에도 즉시 resume 시도 추가.

---

### 현재 상태 (버퍼링 없음)

| 지표 | TRT 포기 직후 | 최적화 후 (현재) |
|------|-------------|----------------|
| 첫 청크 도달 | 11.8초 | **2.1~2.9초** |
| 배치 처리 (B=16) | 800~1000ms | **730~1000ms** |
| 마지막 배치 (B=3) | 3083ms | **559~640ms** |
| 중간 버퍼링 | 있음 | **없음** |
| 끝부분 멈춤 | 있음 | **없음** |
| 첫 글자 잘림 | 있음 | **없음** |

**남은 과제**  
재생 시작 전 "버퍼링 중..." 1~2초 대기는 생성 속도(0.75배속)의 구조적 한계.  
완전 제거를 위해서는 TensorRT 재시도 필요 (예상 1.0배속 이상 달성 시 버퍼 불필요).

---

## 관련 파일

| 파일 | 역할 |
|------|------|
| `backend/convert_unet_trt.py` | UNet ONNX 내보내기 + TRT 엔진 빌드 |
| `backend/convert_vae_trt.py` | VAE ONNX 내보내기 + TRT 엔진 빌드 |
| `backend/build_vae_trt_only.py` | 기존 VAE ONNX에서 TRT만 재빌드 |
| `backend/MuseTalk/trt_engines.py` | TRT UNet / VAE 추론 래퍼 |
| `backend/services/ml_manager.py` | 모델 로딩, TRT 엔진 관리, 웜업 |
| `backend/MuseTalk/stream_inference.py` | 스트리밍 추론 파이프라인 |

엔진 파일 위치 (git 제외):
- `backend/MuseTalk/models/musetalkV15/unet_fp16.trt` — UNet TRT 엔진
- `backend/MuseTalk/models/sd-vae/vae_decoder.trt` — VAE TRT 엔진 (현재 미사용)
