import os
import tempfile

from fastapi import APIRouter, UploadFile, File, HTTPException
from openai import AsyncOpenAI, OpenAIError

from config import OPENAI_API_KEY

router = APIRouter()


@router.post("/stt")
async def speech_to_text(audio: UploadFile = File(...)):
    if not OPENAI_API_KEY or OPENAI_API_KEY == "sk-...":
        raise HTTPException(status_code=503, detail="OpenAI API 키가 설정되지 않았습니다.")

    ext = ".webm"
    if audio.content_type and "mp4" in audio.content_type:
        ext = ".mp4"

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(await audio.read())
            tmp_path = tmp.name

        client = AsyncOpenAI(api_key=OPENAI_API_KEY)
        with open(tmp_path, "rb") as f:
            result = await client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                language="ko",
            )
        return {"text": result.text}

    except OpenAIError as e:
        raise HTTPException(status_code=502, detail=f"Whisper API 오류: {e}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
