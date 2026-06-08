import os
import subprocess
import json
import httpx
from config import FFMPEG_PATH, LP_PYTHON, LP_DIR, OPENAI_API_KEY

def tts(text: str, path: str, voice: str):
    with httpx.Client(timeout=60) as client:
        response = client.post(
            "https://api.openai.com/v1/audio/speech",
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
            json={"model": "tts-1", "voice": voice, "input": text, "response_format": "wav"},
        )
        response.raise_for_status()
        with open(path, "wb") as f:
            f.write(response.content)

def get_audio_duration(audio_path: str) -> float:
    r = subprocess.run(
        [os.path.join(FFMPEG_PATH, "ffprobe.exe"), "-v", "quiet", "-print_format", "json", "-show_format", audio_path],
        capture_output=True, text=True,
    )
    return float(json.loads(r.stdout)["format"]["duration"])

def trim_video(src: str, duration: float, dst: str):
    subprocess.run(
        [os.path.join(FFMPEG_PATH, "ffmpeg.exe"), "-y", "-i", src, "-t", str(duration),
         "-c:v", "libx264", "-c:a", "aac", "-movflags", "+faststart", dst],
        capture_output=True,
    )

def run_liveportrait(source: str, driving: str, output_dir: str, multiplier: float = 0.5, region: str = "all") -> str:
    env = os.environ.copy()
    env["PATH"] = FFMPEG_PATH + ";" + env.get("PATH", "")
    r = subprocess.run(
        [LP_PYTHON, "inference.py", "--source", source, "--driving", driving,
         "--output_dir", output_dir, "--driving_multiplier", str(multiplier), "--animation_region", region],
        cwd=LP_DIR, env=env, capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=300,
    )
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-1000:])
    src_name = os.path.splitext(os.path.basename(source))[0]
    drv_name = os.path.splitext(os.path.basename(driving))[0]
    out = os.path.join(output_dir, f"{src_name}--{drv_name}.mp4")
    if not os.path.exists(out):
        raise FileNotFoundError(f"LivePortrait 출력 없음: {out}")
    return out
