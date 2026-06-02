# 동작 원리 가이드

---

## 전체 파이프라인

```
[사용자 텍스트 입력]
        ↓
[Claude Haiku — 바텐더 응답 생성]
        ↓
[Edge TTS — 텍스트를 음성(wav)으로 변환]
        ↓
[MuseTalk — 아바타 영상 + 음성 → 립싱크 영상]
        ↓
[FFmpeg — 음성 길이에 맞게 영상 트리밍]
        ↓
[Gradio UI에 영상 출력]
```

커스텀 아바타 사용 시 (사진 업로드):

```
[사진 업로드]
        ↓
[LivePortrait — 사진 + 드라이빙 영상 → 표정 애니메이션 영상]
        ↓
[MuseTalk — 해당 영상으로 아바타 초기화]
        ↓ (이후 위 파이프라인 동일)
```

---

## 각 컴포넌트 설명

### Claude Haiku (LLM)

- 사용 모델: `claude-haiku-4-5-20251001`
- 역할: 사용자의 감정 텍스트를 받아 바텐더 말투로 2~3문장 응답 생성 + 칵테일 추천
- 스트리밍 방식으로 응답을 받아 화면에 실시간 표시
- API 키: 환경변수 `ANTHROPIC_API_KEY`

### Edge TTS

- 라이브러리: `edge-tts` (Microsoft Azure TTS 무료 사용)
- 지원 목소리: 한국어 여성(SunHi), 한국어 남성(InJoon), 영어 여성/남성
- 텍스트 → `.wav` 파일로 변환 후 MuseTalk에 전달

### MuseTalk

딥러닝 기반 립싱크 모델. 아바타 소스 영상과 오디오를 합성하여 입 모양이 맞는 영상을 생성합니다.

**내부 구성요소:**

| 컴포넌트 | 역할 |
|----------|------|
| VAE (sd-vae) | 영상 프레임을 잠재 공간으로 인코딩/디코딩 |
| UNet (musetalkV15) | 오디오 특징 + 얼굴 영역을 입력받아 입 모양 생성 |
| Whisper | 오디오를 음성 특징 벡터로 변환 |
| DWPose | 얼굴 랜드마크 검출 |
| FaceParsing | 입 주변 마스크 생성 (jaw 모드) |

**Avatar 초기화 (`preparation=True`):**

1. 소스 영상 전체 프레임 추출
2. 각 프레임에서 얼굴 랜드마크 검출 (DWPose)
3. 입 주변 마스크 생성 (FaceParsing)
4. 프레임을 VAE로 인코딩하여 잠재 벡터 캐싱
5. 결과를 `results/v15/avatars/{avatar_id}/` 에 저장

초기화가 완료되면 이후 실행 시 캐시에서 불러와 빠르게 동작합니다.

**영상 생성 흐름 (`avatar.inference()`):**

1. Whisper로 오디오 → 음성 특징 추출
2. 음성 특징 + 캐싱된 얼굴 잠재벡터를 UNet에 입력
3. UNet이 입 모양이 적용된 잠재벡터 출력
4. VAE 디코더로 실제 픽셀 복원
5. 원본 프레임에 입 영역만 합성 (마스크 블렌딩)
6. FFmpeg로 프레임 → mp4 인코딩

### LivePortrait

이미지 한 장(또는 짧은 영상)을 드라이빙 영상의 표정/동작으로 움직이게 합니다.

**동작 원리:**

1. **소스(Source)**: 움직일 대상 이미지/영상
2. **드라이빙(Driving)**: 참고할 표정/동작 영상
3. Motion Extractor로 드라이빙 영상에서 모션 코드 추출
4. Appearance Extractor로 소스 이미지의 외형 특징 추출
5. Warping Network로 소스의 외형을 드라이빙 모션에 맞게 변형
6. SPADE Generator로 최종 영상 합성

**주요 파라미터 (moodtender_app.py UI에서 조정 가능):**

| 파라미터 | 설명 |
|----------|------|
| `driving_multiplier` | 움직임 강도 (0.2~1.0). 낮을수록 눈·표정 덜 움직임 |
| `animation_region` | 애니메이션 범위. `all` / `exp`(표정만) / `lip`(입만) 등 |
| `bbox_shift` | MuseTalk 입 위치 보정. 얼굴 크기에 따라 ±조정 |

### VRAM 관리 (moodtender_app.py)

LivePortrait와 MuseTalk를 동시에 GPU에 올리면 VRAM 부족이 발생할 수 있어, 순서에 따라 CPU ↔ GPU 간 모델을 이동합니다:

```
LivePortrait 실행 전 → MuseTalk 모델 CPU로 내림
LivePortrait 완료 후 → MuseTalk 모델 GPU 복귀
MuseTalk 실행
```

---

## 앱 비교

| 항목 | moodtender_app.py | persona_app.py |
|------|-------------------|----------------|
| LLM (Claude) | O | X |
| LivePortrait | O (선택적) | X |
| MuseTalk | O | O |
| 포트 | 7862 | 7861 |
| 용도 | 감정 바텐더 서비스 | 텍스트 → 아바타 영상 단순 변환 |

---

## 폴더 구조

```
2차 프로젝트/
├── LivePortrait/               # 얼굴 애니메이션 모델
│   ├── src/                    # 핵심 모듈
│   ├── pretrained_weights/     # 모델 가중치 (gitignore)
│   └── venv/                   # Python 가상환경 (gitignore)
│
└── MuseTalk/                   # 립싱크 모델
    ├── moodtender_app.py       # 메인 앱 (Claude + LP + MT)
    ├── persona_app.py          # 단순 TTS + 립싱크 앱
    ├── config_example.py       # config 템플릿
    ├── config.py               # 개인 경로 설정 (gitignore)
    ├── scripts/                # 추론 스크립트
    ├── musetalk/               # 모델 유틸리티
    ├── models/                 # 모델 가중치 (gitignore)
    ├── data/audio/             # 테스트용 오디오
    └── venv/                   # Python 가상환경 (gitignore)
```
