import os
import time
import json
import tempfile
import shutil
import threading
import asyncio
import cv2
from fastapi import APIRouter, Form, UploadFile, File
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse

from config import VOICES, LP_DRIVING_VIDEOS
from services import ml_manager
from services.video_audio import tts, get_audio_duration, trim_video, run_liveportrait
from scripts.realtime_inference import Avatar

router = APIRouter()

def sse(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

@router.get("/voices")
async def get_voices():
    return [{"id": k, "name": v} for k, v in VOICES.items()]

@router.post("/generate")
async def generate(text: str = Form(...), voice: str = Form("onyx")):
    if not ml_manager.models_ready:
        return JSONResponse({"error": "모델이 로드되지 않았습니다."}, status_code=400)

    loop = asyncio.get_event_loop()
    q: asyncio.Queue = asyncio.Queue()

    def push(data):
        loop.call_soon_threadsafe(q.put_nowait, data)

    def run():
        tmp_dir = tempfile.mkdtemp()
        t_total = time.time()
        try:
            push({"status": "TTS 변환 중..."})
            audio_path = os.path.join(tmp_dir, "tts.wav")
            out_name   = f"output_{int(time.time())}"

            t0 = time.time()
            tts(text, audio_path, voice)
            duration = get_audio_duration(audio_path)
            print(f"[시간] TTS:      {time.time()-t0:.1f}초 (음성 {duration:.1f}초)")

            av = ml_manager.custom_avatar if ml_manager.custom_avatar is not None else ml_manager.avatar_long
            push({"status": f"영상 생성 중... (음성 {duration:.1f}초)"})
            t0 = time.time()
            av.inference(audio_path=audio_path, out_vid_name=out_name, fps=ml_manager.args.fps, skip_save_images=False)
            print(f"[시간] MuseTalk: {time.time()-t0:.1f}초")

            out_vid = os.path.join(av.video_out_path, out_name + ".mp4")
            trimmed = os.path.join(av.video_out_path, out_name + "_trimmed.mp4")
            if os.path.exists(out_vid):
                t0 = time.time()
                trim_video(out_vid, duration, trimmed)
                print(f"[시간] FFmpeg:   {time.time()-t0:.1f}초")
                print(f"[시간] 합계:     {time.time()-t_total:.1f}초")
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
            item = await asyncio.wait_for(q.get(), timeout=300)
            yield sse(item)
            if item.get("done") or item.get("error"):
                break

    return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})

@router.post("/init_avatar")
async def init_avatar(
    file: UploadFile = File(...),
    driving_style: str = Form("기본"),
    motion: float = Form(0.5),
    region: str = Form("all"),
    bbox_shift: int = Form(0),
):
    if not ml_manager.models_ready:
        return JSONResponse({"error": "모델이 로드되지 않았습니다."}, status_code=400)

    loop = asyncio.get_event_loop()
    q: asyncio.Queue = asyncio.Queue()

    def push(data):
        loop.call_soon_threadsafe(q.put_nowait, data)

    tmp_dir     = tempfile.mkdtemp()
    upload_path = os.path.join(tmp_dir, file.filename)
    contents    = await file.read()
    with open(upload_path, "wb") as f:
        f.write(contents)

    def run():
        t0 = time.time()
        try:
            push({"status": "이미지 처리 중..."})
            img = cv2.imread(upload_path)
            if img is None:
                push({"error": "이미지를 읽을 수 없습니다.", "done": True}); return
            h, w = img.shape[:2]
            if max(h, w) > 1280:
                scale = 1280 / max(h, w)
                img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            src_path = os.path.join(tmp_dir, "source.jpg")
            cv2.imwrite(src_path, img)

            lp_out = os.path.join(tmp_dir, "lp_output")
            os.makedirs(lp_out, exist_ok=True)

            push({"status": "VRAM 확보 중..."})
            ml_manager.offload_to_cpu()

            driving = LP_DRIVING_VIDEOS.get(driving_style, LP_DRIVING_VIDEOS["기본"])
            push({"status": "LivePortrait 실행 중..."})
            try:
                lp_video = run_liveportrait(src_path, driving, lp_out, motion, region)
            except Exception as e:
                ml_manager.reload_to_gpu()
                push({"error": f"LivePortrait 오류: {e}", "done": True}); return
            print(f"[시간] LivePortrait: {time.time()-t0:.1f}초")

            push({"status": "아바타 준비 중...", "preview_path": lp_video})
            ml_manager.reload_to_gpu()

            if os.path.exists(ml_manager.CUSTOM_AVATAR_CACHE):
                shutil.rmtree(ml_manager.CUSTOM_AVATAR_CACHE)
            t0 = time.time()
            ml_manager.custom_avatar = Avatar(
                avatar_id="custom_avatar", video_path=lp_video,
                bbox_shift=bbox_shift, batch_size=ml_manager.args.batch_size, preparation=True,
            )
            ml_manager.custom_avatar.input_latent_list_cycle = [
                t.to(ml_manager.device) for t in ml_manager.custom_avatar.input_latent_list_cycle
            ]
            print(f"[시간] MuseTalk 아바타 준비: {time.time()-t0:.1f}초")
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

@router.get("/video")
async def serve_video(path: str):
    if not os.path.exists(path):
        return JSONResponse({"error": "파일 없음"}, status_code=404)
    return FileResponse(path, media_type="video/mp4")

@router.post("/generate_stream")
async def generate_stream(text: str = Form(...), voice: str = Form("onyx")):
    if not ml_manager.models_ready:
        return JSONResponse({"error": "모델이 로드되지 않았습니다."}, status_code=400)

    from stream_inference import inference_stream as _infer_stream
    from config import FFMPEG_PATH

    loop    = asyncio.get_event_loop()
    async_q: asyncio.Queue = asyncio.Queue()

    def push(data):
        loop.call_soon_threadsafe(async_q.put_nowait, data)

    def run():
        tmp_dir = tempfile.mkdtemp()
        t0 = time.time()
        try:
            audio_path = os.path.join(tmp_dir, "tts.wav")
            tts(text, audio_path, voice)
            print(f"[Stream] TTS: {time.time()-t0:.1f}초")

            av    = ml_manager.custom_avatar if ml_manager.custom_avatar is not None else ml_manager.avatar_long
            first = True
            for chunk in _infer_stream(
                av, audio_path, ml_manager.args.fps, FFMPEG_PATH,
                ml_manager.pe, ml_manager.unet, ml_manager.vae, ml_manager.timesteps,
                ml_manager.whisper, ml_manager.audio_processor,
                ml_manager.weight_dtype, ml_manager.device,
                ml_manager.args.audio_padding_length_left,
                ml_manager.args.audio_padding_length_right,
            ):
                if first:
                    print(f"[Stream] 첫 청크: {time.time()-t0:.1f}초")
                    first = False
                push(chunk)
            print(f"[Stream] 완료: {time.time()-t0:.1f}초")
        except Exception as e:
            print(f"[Stream] 오류: {e}")
        finally:
            push(None)
            shutil.rmtree(tmp_dir, ignore_errors=True)

    threading.Thread(target=run, daemon=True).start()

    async def stream():
        while True:
            chunk = await asyncio.wait_for(async_q.get(), timeout=300)
            if chunk is None:
                break
            yield chunk

    return StreamingResponse(
        stream(),
        media_type="video/mp4",
        headers={"Cache-Control": "no-cache", "X-Content-Type-Options": "nosniff"},
    )
