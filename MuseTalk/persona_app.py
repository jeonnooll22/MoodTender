# coding: utf-8
"""
Real-time persona: Text -> Edge-TTS -> MuseTalk -> Video
"""

import os
import sys
import asyncio
import tempfile
import time
import copy
import glob
import pickle
import queue
import shutil
import threading
import subprocess
import argparse
from argparse import Namespace

import cv2
import numpy as np
import torch
import gradio as gr
from omegaconf import OmegaConf
from transformers import WhisperModel
import edge_tts

# ──────────────────────────────────────────────
# FFmpeg PATH
# ──────────────────────────────────────────────
try:
    from config import FFMPEG_PATH
except ImportError:
    FFMPEG_PATH = os.environ.get("FFMPEG_PATH", "")
    if not FFMPEG_PATH:
        print("[경고] config.py 또는 FFMPEG_PATH 환경변수가 없습니다.")
os.environ["PATH"] = FFMPEG_PATH + ";" + os.environ.get("PATH", "")

# ──────────────────────────────────────────────
# Global args (realtime_inference.Avatar uses global args)
# ──────────────────────────────────────────────
args = Namespace(
    version="v15",
    extra_margin=10,
    parsing_mode="jaw",
    left_cheek_width=90,
    right_cheek_width=90,
    batch_size=8,
    fps=25,
    audio_padding_length_left=2,
    audio_padding_length_right=2,
    skip_save_images=False,
    result_dir="./results",
)

# Inject into realtime_inference module namespace before importing
import scripts.realtime_inference as rt
rt.args = args

from scripts.realtime_inference import Avatar
from musetalk.utils.utils import load_all_model
from musetalk.utils.audio_processor import AudioProcessor
from musetalk.utils.face_parsing import FaceParsing

# ──────────────────────────────────────────────
# Device & model init
# ──────────────────────────────────────────────
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"[persona] Using device: {device}")

print("[persona] Loading models...")
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

fp = FaceParsing(left_cheek_width=90, right_cheek_width=90)

# Inject models into rt module (Avatar.inference uses these globals)
rt.vae = vae
rt.unet = unet
rt.pe = pe
rt.timesteps = timesteps
rt.whisper = whisper
rt.audio_processor = audio_processor
rt.weight_dtype = weight_dtype
rt.device = device
rt.fp = fp

print("[persona] Models loaded.")

# ──────────────────────────────────────────────
# Avatar init (first run: preparation=True)
# ──────────────────────────────────────────────
SOURCE_VIDEO = "data/video/source_720p.mp4"
AVATAR_ID = "persona_1"

print(f"[persona] Preparing avatar from {SOURCE_VIDEO} ...")
avatar = Avatar(
    avatar_id=AVATAR_ID,
    video_path=SOURCE_VIDEO,
    bbox_shift=0,
    batch_size=args.batch_size,
    preparation=True,
)
print("[persona] Avatar ready.")

# ──────────────────────────────────────────────
# TTS helper
# ──────────────────────────────────────────────
TTS_VOICE = "ko-KR-SunHiNeural"  # Korean female voice

async def _tts(text: str, out_path: str):
    communicate = edge_tts.Communicate(text, TTS_VOICE)
    await communicate.save(out_path)

def text_to_audio(text: str, out_path: str):
    asyncio.run(_tts(text, out_path))

# ──────────────────────────────────────────────
# Inference pipeline
# ──────────────────────────────────────────────
def generate(text: str, voice: str):
    global TTS_VOICE
    TTS_VOICE = voice

    if not text.strip():
        return None, "텍스트를 입력해주세요."

    tmp_dir = tempfile.mkdtemp()
    audio_path = os.path.join(tmp_dir, "tts.wav")
    out_name = f"output_{int(time.time())}"

    try:
        # 1. TTS
        print(f"[persona] TTS: {text[:40]}...")
        text_to_audio(text, audio_path)

        # 2. MuseTalk inference
        print("[persona] Running inference...")
        avatar.inference(
            audio_path=audio_path,
            out_vid_name=out_name,
            fps=args.fps,
            skip_save_images=False,
        )

        # 3. Find output video
        out_vid = os.path.join(avatar.video_out_path, out_name + ".mp4")
        if os.path.exists(out_vid):
            return out_vid, "완료!"
        else:
            return None, "영상 생성 실패"

    except Exception as e:
        return None, f"오류: {str(e)}"
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

# ──────────────────────────────────────────────
# Gradio UI
# ──────────────────────────────────────────────
VOICES = {
    "한국어 여성 (SunHi)": "ko-KR-SunHiNeural",
    "한국어 남성 (InJoon)": "ko-KR-InJoonNeural",
    "영어 여성 (Jenny)": "en-US-JennyNeural",
    "영어 남성 (Guy)": "en-US-GuyNeural",
}

with gr.Blocks(title="Real-time Persona") as demo:
    gr.Markdown("# 🎭 Real-time Persona\n텍스트를 입력하면 아바타가 말합니다.")

    with gr.Row():
        with gr.Column(scale=2):
            text_input = gr.Textbox(
                label="텍스트 입력",
                placeholder="여기에 말할 내용을 입력하세요...",
                lines=4,
            )
            voice_dropdown = gr.Dropdown(
                choices=list(VOICES.keys()),
                value="한국어 여성 (SunHi)",
                label="목소리 선택",
            )
            generate_btn = gr.Button("생성", variant="primary")
            status_text = gr.Textbox(label="상태", interactive=False)

        with gr.Column(scale=3):
            video_output = gr.Video(label="결과 영상", autoplay=True)

    generate_btn.click(
        fn=lambda t, v: generate(t, VOICES[v]),
        inputs=[text_input, voice_dropdown],
        outputs=[video_output, status_text],
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7861)
