# coding: utf-8
"""MoodTender — FastAPI 백엔드"""

import os, sys, asyncio, tempfile, threading, time, json, shutil, subprocess, traceback
from pathlib import Path
from argparse import Namespace

import cv2
import torch
from fastapi import FastAPI, Form, UploadFile, File, Depends, HTTPException
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
import bcrypt as _bcrypt_lib
from jose import jwt
from datetime import datetime, timedelta
from pydantic import BaseModel

# ─── 경로 설정 ────────────────────────────────────────────────
_BACKEND_DIR  = Path(__file__).resolve().parent
_MUSETALK_DIR = _BACKEND_DIR / "MuseTalk"
_FRONTEND_DIR = _BACKEND_DIR.parent / "frontend"

sys.path.insert(0, str(_MUSETALK_DIR))
os.chdir(_MUSETALK_DIR)

try:
    from config import FFMPEG_PATH, LP_DIR, LP_PYTHON
except ImportError:
    FFMPEG_PATH = os.environ.get("FFMPEG_PATH", "")
    LP_DIR      = os.environ.get("LP_DIR", "")
    LP_PYTHON   = os.environ.get("LP_PYTHON", "")
    if not FFMPEG_PATH:
        print("[경고] config.py 또는 환경변수가 없습니다.")

os.environ["PATH"] = FFMPEG_PATH + ";" + os.environ.get("PATH", "")

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

# ─── 인증 설정 ────────────────────────────────────────────────
SECRET_KEY = os.environ.get("SECRET_KEY", "change-me-in-production")
ALGORITHM  = "HS256"
if SECRET_KEY == "change-me-in-production":
    print("[경고] SECRET_KEY 환경변수를 설정하세요.")

_DB_PATH = _BACKEND_DIR / "bar_project.db"
_engine  = create_engine(f"sqlite:///{_DB_PATH}", connect_args={"check_same_thread": False})
_Session = sessionmaker(autocommit=False, autoflush=False, bind=_engine)
_Base    = declarative_base()

class _User(_Base):
    __tablename__ = "users"
    id              = Column(Integer, primary_key=True, index=True)
    username        = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)

_Base.metadata.create_all(bind=_engine)

class _UserCreate(BaseModel):
    username: str
    password: str

class _UserResponse(BaseModel):
    id: int
    username: str
    class Config:
        from_attributes = True

class _Token(BaseModel):
    access_token: str
    token_type: str

def _hash_pw(pw: str) -> str:
    return _bcrypt_lib.hashpw(pw.encode(), _bcrypt_lib.gensalt()).decode()

def _verify_pw(plain: str, hashed: str) -> bool:
    return _bcrypt_lib.checkpw(plain.encode(), hashed.encode())

def _create_token(username: str) -> str:
    expire = datetime.utcnow() + timedelta(minutes=60)
    return jwt.encode({"sub": username, "exp": expire}, SECRET_KEY, algorithm=ALGORITHM)

def _get_db():
    db = _Session()
    try:
        yield db
    finally:
        db.close()

LP_DRIVING_VIDEOS = {
    "기본":  os.path.join(LP_DIR, "assets", "examples", "driving", "d0.mp4"),
    "활발":  os.path.join(LP_DIR, "assets", "examples", "driving", "d9.mp4"),
    "차분":  os.path.join(LP_DIR, "assets", "examples", "driving", "d13.mp4"),
}

VOICES = {
    "ko-KR-SunHiNeural":  "한국어 여성 (SunHi)",
    "ko-KR-InJoonNeural": "한국어 남성 (InJoon)",
    "en-US-JennyNeural":  "영어 여성 (Jenny)",
    "en-US-GuyNeural":    "영어 남성 (Guy)",
}

# ─── MuseTalk 설정 ────────────────────────────────────────────
args = Namespace(
    version="v15", extra_margin=10, parsing_mode="jaw",
    left_cheek_width=40, right_cheek_width=40, batch_size=8, fps=25,
    audio_padding_length_left=2, audio_padding_length_right=2,
    skip_save_images=False, result_dir="./results",
)

import scripts.realtime_inference as rt
rt.args = args

from scripts.realtime_inference import Avatar
from musetalk.utils.utils import load_all_model
from musetalk.utils.audio_processor import AudioProcessor
from musetalk.utils.face_parsing import FaceParsing
from transformers import WhisperModel
import edge_tts

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"[서버] 디바이스: {device}")

# ─── 모델 전역 상태 ───────────────────────────────────────────
vae = unet = pe = timesteps = whisper = audio_processor = weight_dtype = fp = None
avatar_short = avatar_long = custom_avatar = None
CUSTOM_AVATAR_CACHE = f"./results/{args.version}/avatars/custom_avatar"

_models_ready       = False
_loading_status     = "모델 미로드"
_loading_error      = None
_loading_in_progress = False

# ─── 모델 로드 ────────────────────────────────────────────────
def _load_models():
    global vae, unet, pe, timesteps, whisper, audio_processor, weight_dtype, fp
    global avatar_short, avatar_long, _models_ready, _loading_status, _loading_error, _loading_in_progress
    try:
        _loading_status = "MuseTalk 모델 로딩 중..."
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

        rt.vae = vae; rt.unet = unet; rt.pe = pe; rt.timesteps = timesteps
        rt.whisper = whisper; rt.audio_processor = audio_processor
        rt.weight_dtype = weight_dtype; rt.device = device; rt.fp = fp

        _loading_status = "아바타 준비 중..."
        avatar_short = Avatar(avatar_id="bartender",      video_path="data/video/bartender.mp4",     bbox_shift=0, batch_size=args.batch_size, preparation=True)
        avatar_long  = Avatar(avatar_id="bartender_long", video_path="data/video/Bartender_long.mp4", bbox_shift=0, batch_size=args.batch_size, preparation=True)

        _try_load_custom_avatar_from_cache()
        _models_ready   = True
        _loading_status = "준비 완료"
        print("[서버] 모델 준비 완료")
    except Exception as e:
        _loading_error  = str(e)
        _loading_status = f"로딩 실패: {e}"
        print(f"[서버] 로딩 실패: {e}")
    finally:
        _loading_in_progress = False

def _try_load_custom_avatar_from_cache():
    global custom_avatar
    info_path = os.path.join(CUSTOM_AVATAR_CACHE, "avator_info.json")
    if not os.path.exists(info_path):
        return
    try:
        with open(info_path) as f:
            info = json.load(f)
        custom_avatar = Avatar(
            avatar_id="custom_avatar",
            video_path=info.get("video_path", ""),
            bbox_shift=info.get("bbox_shift", 0),
            batch_size=args.batch_size,
            preparation=False,
        )
    except Exception as e:
        print(f"[캐시] 커스텀 아바타 로드 실패: {e}")

# ─── VRAM 관리 ────────────────────────────────────────────────
def _offload_to_cpu():
    vae.vae = vae.vae.cpu(); unet.model = unet.model.cpu()
    pe.cpu(); whisper.cpu()
    torch.cuda.empty_cache()

def _reload_to_gpu():
    vae.vae = vae.vae.half().to(device); unet.model = unet.model.half().to(device)
    pe.half().to(device); whisper.to(device=device, dtype=weight_dtype)
    rt.vae = vae; rt.unet = unet; rt.pe = pe; rt.whisper = whisper

# ─── TTS ──────────────────────────────────────────────────────
async def _tts_async(text: str, path: str, voice: str):
    await edge_tts.Communicate(text, voice).save(path)

def tts(text: str, path: str, voice: str):
    asyncio.run(_tts_async(text, path, voice))

# ─── 영상 유틸 ────────────────────────────────────────────────
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

# ─── LivePortrait ─────────────────────────────────────────────
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

# ─── SSE 헬퍼 ─────────────────────────────────────────────────
def sse(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

# ─── FastAPI ──────────────────────────────────────────────────
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=str(_FRONTEND_DIR)), name="static")

@app.get("/")
async def index():
    return FileResponse(_FRONTEND_DIR / "index.html")

@app.get("/login")
async def login_page():
    return FileResponse(_FRONTEND_DIR / "login.html")

# ─── 인증 API ─────────────────────────────────────────────────
@app.post("/api/signup")
def signup(user: _UserCreate, db: Session = Depends(_get_db)):
    if db.query(_User).filter(_User.username == user.username).first():
        raise HTTPException(status_code=400, detail="이미 존재하는 아이디입니다.")
    new_user = _User(username=user.username, hashed_password=_hash_pw(user.password))
    db.add(new_user); db.commit(); db.refresh(new_user)
    return {"id": new_user.id, "username": new_user.username}

@app.post("/api/login")
def login(user: _UserCreate, db: Session = Depends(_get_db)):
    db_user = db.query(_User).filter(_User.username == user.username).first()
    if not db_user or not _verify_pw(user.password, db_user.hashed_password):
        raise HTTPException(status_code=401, detail="아이디 또는 비밀번호가 틀렸습니다.")
    return {"access_token": _create_token(db_user.username), "token_type": "bearer"}

@app.get("/api/check-username")
def check_username(username: str, db: Session = Depends(_get_db)):
    exists = db.query(_User).filter(_User.username == username).first()
    if exists:
        return {"is_available": False, "message": "이미 사용 중인 아이디입니다."}
    return {"is_available": True, "message": "사용 가능한 아이디입니다."}

# 상태 폴링
@app.get("/api/status")
async def get_status():
    return {"ready": _models_ready, "status": _loading_status, "error": _loading_error, "loading": _loading_in_progress}

# 모델 로드 시작
@app.post("/api/load_model")
async def load_model():
    global _loading_in_progress
    if _models_ready:
        return {"message": "already_loaded"}
    if _loading_in_progress:
        return {"message": "in_progress"}
    _loading_in_progress = True
    threading.Thread(target=_load_models, daemon=True).start()
    return {"message": "started"}

# 모델 로드 SSE 스트림
@app.get("/api/load_model/stream")
async def load_model_stream():
    async def gen():
        while not _models_ready and not _loading_error:
            yield sse({"status": _loading_status, "ready": False, "loading": True})
            await asyncio.sleep(0.8)
        yield sse({"status": _loading_status, "ready": _models_ready, "error": _loading_error, "loading": False})
    return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})

# 목소리 목록
@app.get("/api/voices")
async def get_voices():
    return [{"id": k, "name": v} for k, v in VOICES.items()]

# 영상 생성 SSE
@app.post("/api/generate")
async def generate(text: str = Form(...), voice: str = Form("ko-KR-SunHiNeural")):
    if not _models_ready:
        return JSONResponse({"error": "모델이 로드되지 않았습니다."}, status_code=400)

    loop = asyncio.get_event_loop()
    q: asyncio.Queue = asyncio.Queue()

    def push(data):
        loop.call_soon_threadsafe(q.put_nowait, data)

    def run():
        tmp_dir = tempfile.mkdtemp()
        try:
            push({"status": "TTS 변환 중..."})
            audio_path = os.path.join(tmp_dir, "tts.wav")
            out_name = f"output_{int(time.time())}"

            tts(text, audio_path, voice)
            duration = get_audio_duration(audio_path)

            av = custom_avatar if custom_avatar is not None else avatar_long
            push({"status": f"영상 생성 중... (음성 {duration:.1f}초)"})
            av.inference(audio_path=audio_path, out_vid_name=out_name, fps=args.fps, skip_save_images=False)

            out_vid  = os.path.join(av.video_out_path, out_name + ".mp4")
            trimmed  = os.path.join(av.video_out_path, out_name + "_trimmed.mp4")
            if os.path.exists(out_vid):
                trim_video(out_vid, duration, trimmed)
                final = trimmed if os.path.exists(trimmed) else out_vid
                push({"status": "완료!", "done": True, "video_path": final})
            else:
                push({"error": "영상 생성 실패", "done": True})
        except Exception as e:
            push({"error": str(e), "done": True})
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    threading.Thread(target=run, daemon=True).start()

    async def stream():
        while True:
            item = await asyncio.wait_for(q.get(), timeout=120)
            yield sse(item)
            if item.get("done") or item.get("error"):
                break

    return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})

# 아바타 초기화 SSE
@app.post("/api/init_avatar")
async def init_avatar(
    file: UploadFile = File(...),
    driving_style: str = Form("기본"),
    motion: float = Form(0.5),
    region: str = Form("all"),
    bbox_shift: int = Form(0),
):
    if not _models_ready:
        return JSONResponse({"error": "모델이 로드되지 않았습니다."}, status_code=400)

    loop = asyncio.get_event_loop()
    q: asyncio.Queue = asyncio.Queue()

    def push(data):
        loop.call_soon_threadsafe(q.put_nowait, data)

    tmp_dir = tempfile.mkdtemp()
    upload_path = os.path.join(tmp_dir, file.filename)
    contents = await file.read()
    with open(upload_path, "wb") as f:
        f.write(contents)

    def run():
        global custom_avatar
        try:
            push({"status": "이미지 처리 중..."})
            img = cv2.imread(upload_path)
            if img is None:
                push({"error": "이미지를 읽을 수 없습니다.", "done": True})
                return
            h, w = img.shape[:2]
            if max(h, w) > 1280:
                scale = 1280 / max(h, w)
                img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            src_path = os.path.join(tmp_dir, "source.jpg")
            cv2.imwrite(src_path, img)

            lp_out = os.path.join(tmp_dir, "lp_output")
            os.makedirs(lp_out, exist_ok=True)

            push({"status": "VRAM 확보 중..."})
            _offload_to_cpu()

            driving = LP_DRIVING_VIDEOS.get(driving_style, LP_DRIVING_VIDEOS["기본"])
            push({"status": "LivePortrait 실행 중..."})
            try:
                lp_video = run_liveportrait(src_path, driving, lp_out, motion, region)
            except Exception as e:
                _reload_to_gpu()
                push({"error": f"LivePortrait 오류: {e}", "done": True})
                return

            push({"status": "아바타 준비 중...", "preview_path": lp_video})
            _reload_to_gpu()

            if os.path.exists(CUSTOM_AVATAR_CACHE):
                shutil.rmtree(CUSTOM_AVATAR_CACHE)
            custom_avatar = Avatar(
                avatar_id="custom_avatar", video_path=lp_video,
                bbox_shift=bbox_shift, batch_size=args.batch_size, preparation=True,
            )
            push({"status": "커스텀 아바타 준비 완료!", "done": True})
        except Exception as e:
            push({"error": str(e), "done": True})

    threading.Thread(target=run, daemon=True).start()

    async def stream():
        while True:
            item = await asyncio.wait_for(q.get(), timeout=300)
            yield sse(item)
            if item.get("done") or item.get("error"):
                break

    return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})

# 영상 파일 서빙
@app.get("/api/video")
async def serve_video(path: str):
    if not os.path.exists(path):
        return JSONResponse({"error": "파일 없음"}, status_code=404)
    return FileResponse(path, media_type="video/mp4")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=7862)
