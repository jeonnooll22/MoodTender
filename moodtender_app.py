# coding: utf-8
"""
MoodTender: 감정 바텐더
사용자 텍스트 입력 → LLM 바텐더 응답 → TTS → MuseTalk 립싱크 영상
"""

import os
import sys
import asyncio
import tempfile
import time
import shutil
import subprocess
import json
import traceback
import cv2
from argparse import Namespace

import gradio as gr
import torch
import edge_tts
from anthropic import Anthropic

# ─────────────────────────────────────────────
# 루트에서 실행할 경우 MuseTalk/ 디렉토리로 전환
# ─────────────────────────────────────────────
_MUSETALK_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(_MUSETALK_DIR) != "MuseTalk":
    _MUSETALK_DIR = os.path.join(_MUSETALK_DIR, "MuseTalk")
os.chdir(_MUSETALK_DIR)
sys.path.insert(0, _MUSETALK_DIR)

# ─────────────────────────────────────────────
# 경로 설정 (config.py에서 로드, 없으면 환경변수 사용)
# ─────────────────────────────────────────────
try:
    from config import FFMPEG_PATH, LP_DIR, LP_PYTHON
except ImportError:
    FFMPEG_PATH = os.environ.get("FFMPEG_PATH", "")
    LP_DIR      = os.environ.get("LP_DIR", "")
    LP_PYTHON   = os.environ.get("LP_PYTHON", "")
    if not FFMPEG_PATH:
        print("[경고] config.py 또는 FFMPEG_PATH 환경변수가 없습니다.")

os.environ["PATH"] = FFMPEG_PATH + ";" + os.environ.get("PATH", "")

LP_DRIVING_VIDEOS = {
    "기본 (자연스러운 움직임)": os.path.join(LP_DIR, "assets", "examples", "driving", "d0.mp4"),
    "활발한 움직임":            os.path.join(LP_DIR, "assets", "examples", "driving", "d9.mp4"),
    "차분한 움직임":            os.path.join(LP_DIR, "assets", "examples", "driving", "d13.mp4"),
}

# ─────────────────────────────────────────────
# Anthropic API 키
# ─────────────────────────────────────────────
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
if not ANTHROPIC_API_KEY:
    print("[경고] ANTHROPIC_API_KEY 환경변수가 없습니다.")
    print("       실행 전: $env:ANTHROPIC_API_KEY='your-key'")

# ─────────────────────────────────────────────
# MuseTalk 모델 로드
# ─────────────────────────────────────────────
args = Namespace(
    version="v15",
    extra_margin=10,
    parsing_mode="jaw",
    left_cheek_width=40,
    right_cheek_width=40,
    batch_size=8,
    fps=25,
    audio_padding_length_left=2,
    audio_padding_length_right=2,
    skip_save_images=False,
    result_dir="./results",
)

import scripts.realtime_inference as rt
rt.args = args

from scripts.realtime_inference import Avatar
from musetalk.utils.utils import load_all_model
from musetalk.utils.audio_processor import AudioProcessor
from musetalk.utils.face_parsing import FaceParsing
from transformers import WhisperModel

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"[MoodTender] 디바이스: {device}")

print("[MoodTender] 모델 로딩 중...")
vae, unet, pe = load_all_model(
    unet_model_path="./models/musetalkV15/unet.pth",
    vae_type="sd-vae",
    unet_config="./models/musetalkV15/musetalk.json",
    device=device,
)
timesteps = torch.tensor([0], device=device)
pe = pe.half().to(device)
vae.vae = vae.vae.half().to(device)
unet.model = unet.model.half().to(device)
weight_dtype = unet.model.dtype

audio_processor = AudioProcessor(feature_extractor_path="./models/whisper")
whisper = WhisperModel.from_pretrained("./models/whisper")
whisper = whisper.to(device=device, dtype=weight_dtype).eval()
whisper.requires_grad_(False)

fp = FaceParsing(left_cheek_width=40, right_cheek_width=40)

rt.vae = vae
rt.unet = unet
rt.pe = pe
rt.timesteps = timesteps
rt.whisper = whisper
rt.audio_processor = audio_processor
rt.weight_dtype = weight_dtype
rt.device = device
rt.fp = fp

print("[MoodTender] 아바타 준비 중...")
avatar_short = Avatar(
    avatar_id="bartender",
    video_path="data/video/bartender.mp4",
    bbox_shift=0,
    batch_size=args.batch_size,
    preparation=True,
)
avatar_long = Avatar(
    avatar_id="bartender_long",
    video_path="data/video/Bartender_long.mp4",
    bbox_shift=0,
    batch_size=args.batch_size,
    preparation=True,
)
print("[MoodTender] 아바타 준비 완료!")

AVATARS = {
    "bartender (4초)": avatar_short,
    "bartender_long (10초)": avatar_long,
}

# ─────────────────────────────────────────────
# 바텐더 시스템 프롬프트
# ─────────────────────────────────────────────
BARTENDER_SYSTEM = """당신은 MoodTender의 감정 바텐더입니다.
손님이 오늘 하루의 감정을 털어놓으면, 따뜻하게 공감하고 그 감정에 맞는 칵테일을 추천합니다.

응답 규칙:
- 2~3문장으로 짧고 따뜻하게
- 반드시 칵테일 하나를 추천하고 왜 어울리는지 한 줄로 설명
- 친근하고 포근한 바텐더 말투 (반말 금지)
- 한국어로만 응답
- 음성으로 읽힐 텍스트이므로 이모지나 특수문자 사용 금지"""

# ─────────────────────────────────────────────
# LLM 바텐더 응답 생성 (스트리밍)
# ─────────────────────────────────────────────
def stream_bartender_response(emotion_text: str):
    client = Anthropic(api_key=ANTHROPIC_API_KEY)
    with client.messages.stream(
        model="claude-haiku-4-5-20251001",
        max_tokens=200,
        system=BARTENDER_SYSTEM,
        messages=[{"role": "user", "content": emotion_text}],
    ) as stream:
        for text in stream.text_stream:
            yield text

# ─────────────────────────────────────────────
# TTS 헬퍼
# ─────────────────────────────────────────────
TTS_VOICE = "ko-KR-SunHiNeural"

async def _tts(text: str, out_path: str):
    communicate = edge_tts.Communicate(text, TTS_VOICE)
    await communicate.save(out_path)

def text_to_audio(text: str, out_path: str):
    asyncio.run(_tts(text, out_path))

def get_audio_duration(audio_path: str) -> float:
    result = subprocess.run([
        os.path.join(FFMPEG_PATH, "ffprobe.exe"),
        "-v", "quiet", "-print_format", "json", "-show_format", audio_path,
    ], capture_output=True, text=True)
    return float(json.loads(result.stdout)["format"]["duration"])

def trim_video(video_path: str, duration: float, out_path: str):
    subprocess.run([
        os.path.join(FFMPEG_PATH, "ffmpeg.exe"),
        "-y", "-i", video_path,
        "-t", str(duration),
        "-c:v", "libx264", "-c:a", "aac",
        "-movflags", "+faststart",
        out_path,
    ], capture_output=True)

def to_gradio_temp(video_path: str) -> str:
    """Gradio Content-Length 오류 방지용 temp 복사."""
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp.close()
    shutil.copy2(video_path, tmp.name)
    return tmp.name

# ─────────────────────────────────────────────
# MuseTalk VRAM 관리
# ─────────────────────────────────────────────
def _offload_musetalk_to_cpu():
    vae.vae = vae.vae.cpu()
    unet.model = unet.model.cpu()
    pe.cpu()
    whisper.cpu()
    torch.cuda.empty_cache()
    print("[MoodTender] MuseTalk 모델 → CPU (VRAM 확보)")

def _reload_musetalk_to_gpu():
    vae.vae = vae.vae.half().to(device)
    unet.model = unet.model.half().to(device)
    pe.half().to(device)
    whisper.to(device=device, dtype=weight_dtype)
    rt.vae = vae; rt.unet = unet; rt.pe = pe; rt.whisper = whisper
    print("[MoodTender] MuseTalk 모델 → GPU 복귀")

# ─────────────────────────────────────────────
# LivePortrait + 커스텀 아바타
# ─────────────────────────────────────────────
custom_avatar = None
CUSTOM_AVATAR_CACHE = f"./results/{args.version}/avatars/custom_avatar"

def run_liveportrait(
    source_image_path: str,
    driving_video_path: str,
    output_dir: str,
    driving_multiplier: float = 0.5,
    animation_region: str = "all",
) -> str:
    env = os.environ.copy()
    env["PATH"] = FFMPEG_PATH + ";" + env.get("PATH", "")
    result = subprocess.run(
        [LP_PYTHON, "inference.py",
         "--source", source_image_path,
         "--driving", driving_video_path,
         "--output_dir", output_dir,
         "--driving_multiplier", str(driving_multiplier),
         "--animation_region", animation_region],
        cwd=LP_DIR, env=env,
        capture_output=True, text=True,
        encoding="utf-8", errors="replace",
        timeout=300,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr[-1000:])
    src_name = os.path.splitext(os.path.basename(source_image_path))[0]
    drv_name = os.path.splitext(os.path.basename(driving_video_path))[0]
    out_path = os.path.join(output_dir, f"{src_name}--{drv_name}.mp4")
    if not os.path.exists(out_path):
        raise FileNotFoundError(f"LivePortrait 출력 파일을 찾을 수 없음: {out_path}")
    return out_path

def prepare_custom_avatar(video_path: str, bbox_shift: int = 0):
    global custom_avatar
    if os.path.exists(CUSTOM_AVATAR_CACHE):
        shutil.rmtree(CUSTOM_AVATAR_CACHE)
    custom_avatar = Avatar(
        avatar_id="custom_avatar",
        video_path=video_path,
        bbox_shift=bbox_shift,
        batch_size=args.batch_size,
        preparation=True,
    )
    return custom_avatar

def try_load_custom_avatar_from_cache():
    global custom_avatar
    info_path = os.path.join(CUSTOM_AVATAR_CACHE, "avator_info.json")
    if not os.path.exists(info_path):
        return
    try:
        with open(info_path, "r") as f:
            saved_info = json.load(f)
        custom_avatar = Avatar(
            avatar_id="custom_avatar",
            video_path=saved_info.get("video_path", ""),
            bbox_shift=saved_info.get("bbox_shift", 0),
            batch_size=args.batch_size,
            preparation=False,
        )
        print(f"[MoodTender] 커스텀 아바타 캐시 로드 완료 (bbox_shift={saved_info.get('bbox_shift', 0)})")
    except Exception as e:
        print(f"[MoodTender] 커스텀 아바타 캐시 로드 실패 (무시): {e}")

try_load_custom_avatar_from_cache()

def initialize_avatar(
    source_image,
    driving_style: str,
    motion_intensity: float,
    anim_region: str,
    bbox_shift: int = 0,
):
    """사진 → LivePortrait → MuseTalk 준비. yields (status, preview_video)"""
    if source_image is None:
        yield "사진을 업로드해주세요.", None
        return

    if isinstance(source_image, str):
        source_image_path = source_image
    elif hasattr(source_image, "name"):
        source_image_path = source_image.name
    else:
        source_image_path = str(source_image)

    driving_video = LP_DRIVING_VIDEOS[driving_style]
    tmp_dir = tempfile.mkdtemp()
    lp_output_dir = os.path.join(tmp_dir, "lp_output")
    os.makedirs(lp_output_dir, exist_ok=True)

    # 대용량 이미지 리사이즈 (1280px 이내)
    source_path = os.path.join(tmp_dir, "source.jpg")
    img = cv2.imread(source_image_path)
    if img is None:
        yield "이미지를 읽을 수 없습니다.", None
        return
    h, w = img.shape[:2]
    if max(h, w) > 1280:
        scale = 1280 / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        print(f"[LP] 이미지 리사이즈: {w}x{h} → {img.shape[1]}x{img.shape[0]}")
    cv2.imwrite(source_path, img)

    yield "VRAM 확보 중...", None
    _offload_musetalk_to_cpu()

    yield f"LivePortrait 실행 중... (강도:{motion_intensity}, 범위:{anim_region})", None
    try:
        lp_video = run_liveportrait(
            source_path, driving_video, lp_output_dir,
            driving_multiplier=motion_intensity,
            animation_region=anim_region,
        )
    except Exception as e:
        _reload_musetalk_to_gpu()
        yield f"LivePortrait 오류: {str(e)}", None
        return

    cap = cv2.VideoCapture(lp_video)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    out_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    print(f"[LP 출력] {out_w}x{out_h} | {frame_count}프레임")
    yield f"아바타 준비 중... ({out_w}x{out_h}, {frame_count}프레임)", lp_video

    try:
        _reload_musetalk_to_gpu()
        prepare_custom_avatar(lp_video, bbox_shift=int(bbox_shift))
    except Exception as e:
        print(f"[MuseTalk 준비 오류]\n{traceback.format_exc()}")
        yield f"MuseTalk 준비 오류: {str(e)}", lp_video
        return

    yield "준비 완료! 텍스트를 입력하고 생성하세요.", lp_video

# ─────────────────────────────────────────────
# 영상 생성 (TTS + MuseTalk)
# ─────────────────────────────────────────────
def run_inference(text: str, voice: str, av):
    """yields (status, video_path)"""
    global TTS_VOICE
    TTS_VOICE = VOICES[voice]
    if not text.strip():
        yield "텍스트를 입력해주세요.", None
        return

    yield "영상 생성 중...", None
    tmp_dir = tempfile.mkdtemp()
    audio_path = os.path.join(tmp_dir, "tts.wav")
    out_name = f"output_{int(time.time())}"
    t_total = time.time()
    try:
        t0 = time.time()
        text_to_audio(text, audio_path)
        duration = get_audio_duration(audio_path)
        print(f"[시간] TTS:            {time.time()-t0:.1f}초  (음성 {duration:.1f}초)")

        t0 = time.time()
        av.inference(audio_path=audio_path, out_vid_name=out_name, fps=args.fps, skip_save_images=False)
        print(f"[시간] MuseTalk:       {time.time()-t0:.1f}초")

        out_vid = os.path.join(av.video_out_path, out_name + ".mp4")
        trimmed_vid = os.path.join(av.video_out_path, out_name + "_trimmed.mp4")
        if os.path.exists(out_vid):
            t0 = time.time()
            trim_video(out_vid, duration, trimmed_vid)
            print(f"[시간] FFmpeg 트리밍: {time.time()-t0:.1f}초")
            print(f"[시간] 합계:          {time.time()-t_total:.1f}초")
            final_vid = trimmed_vid if os.path.exists(trimmed_vid) else out_vid
            yield "완료!", to_gradio_temp(final_vid)
        else:
            yield "영상 생성 실패", None
    except Exception as e:
        yield f"오류: {str(e)}", None
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

# ─────────────────────────────────────────────
# Gradio UI
# ─────────────────────────────────────────────
VOICES = {
    "한국어 여성 (SunHi)": "ko-KR-SunHiNeural",
    "한국어 남성 (InJoon)": "ko-KR-InJoonNeural",
    "영어 여성 (Jenny)": "en-US-JennyNeural",
    "영어 남성 (Guy)": "en-US-GuyNeural",
}

CSS = """
.section-title { font-size: 0.85rem; font-weight: 600; color: #666; margin-bottom: 4px; }
.divider { border-top: 1px solid #e0e0e0; margin: 12px 0; }
"""

with gr.Blocks(title="MoodTender", css=CSS) as demo:
    gr.Markdown("# MoodTender\n### 사일의 감정 바텐더")

    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("**아바타 설정**", elem_classes=["section-title"])
            photo_input = gr.File(
                label="사진 업로드 (JPG/PNG, 생략 시 기본 바텐더 사용)",
                file_types=[".jpg", ".jpeg", ".png", ".webp"],
                file_count="single",
                height=100,
            )
            photo_status = gr.Textbox(
                label="업로드 파일", interactive=False, value="업로드된 파일 없음",
            )
            with gr.Row():
                driving_dropdown = gr.Dropdown(
                    choices=list(LP_DRIVING_VIDEOS.keys()),
                    value="기본 (자연스러운 움직임)",
                    label="움직임 스타일", scale=2,
                )
                anim_region_dropdown = gr.Dropdown(
                    choices=["all", "exp", "pose", "lip", "eyes"],
                    value="all", label="움직임 범위", scale=1,
                )
            motion_slider = gr.Slider(
                minimum=0.2, maximum=1.0, value=0.5, step=0.1,
                label="움직임 강도 (낮을수록 눈·표정 덜 움직임)",
            )
            bbox_shift_slider = gr.Slider(
                minimum=-10, maximum=10, value=0, step=1,
                label="입 위치 조정 bbox_shift (음수=위↑, 양수=아래↓) — 변경 시 아바타 재생성 필요",
            )
            init_btn = gr.Button("아바타 생성", variant="secondary")
            init_status = gr.Textbox(
                label="아바타 상태", value="기본 바텐더 사용 중", interactive=False,
            )
            lp_preview = gr.Video(
                label="LivePortrait 미리보기", autoplay=True, height=180, visible=False,
            )

            gr.Markdown("---", elem_classes=["divider"])

            gr.Markdown("**바텐더에게 말하기**", elem_classes=["section-title"])
            text_input = gr.Textbox(
                label="텍스트 입력",
                placeholder="말할 내용을 입력하세요...",
                lines=4,
            )
            voice_dropdown = gr.Dropdown(
                choices=list(VOICES.keys()), value="한국어 여성 (SunHi)", label="목소리",
            )
            generate_btn = gr.Button("생성", variant="primary", size="lg")
            gen_status = gr.Textbox(label="상태", interactive=False)

        with gr.Column(scale=3):
            video_output = gr.Video(label="바텐더 응답", autoplay=True, height=560)

    # 파일 선택 확인
    def on_file_upload(f):
        if f is None:
            return "업로드된 파일 없음"
        path = f if isinstance(f, str) else (f.name if hasattr(f, "name") else str(f))
        return f"선택됨: {os.path.basename(path)}"

    photo_input.change(fn=on_file_upload, inputs=photo_input, outputs=photo_status)

    # 아바타 초기화
    def on_init(photo, driving_style, motion_intensity, anim_region, bbox_shift):
        for status, preview in initialize_avatar(photo, driving_style, motion_intensity, anim_region, bbox_shift):
            stable_preview = to_gradio_temp(preview) if preview else None
            yield status, gr.update(value=stable_preview, visible=preview is not None)

    init_btn.click(
        fn=on_init,
        inputs=[photo_input, driving_dropdown, motion_slider, anim_region_dropdown, bbox_shift_slider],
        outputs=[init_status, lp_preview],
    )

    # 영상 생성
    def on_generate(text, voice):
        av = custom_avatar if custom_avatar is not None else avatar_long
        print(f"[생성] {'커스텀' if custom_avatar is not None else '기본'} 아바타 사용")
        yield from run_inference(text, voice, av)

    generate_btn.click(
        fn=on_generate,
        inputs=[text_input, voice_dropdown],
        outputs=[gen_status, video_output],
    )

if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7862,
        max_file_size="10mb",
    )
