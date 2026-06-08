from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.ext.asyncio import AsyncSession

# 새로 구성한 backend 패키지 구조에 맞춰 import 경로를 조정했습니다
from backend.database import get_db
from backend.models.schemas import LLMRequest, LLMResponse
from backend.services.openai_llm import OpenAIError, generate_bartender_reply

router = APIRouter()

@router.post("/llm/respond", response_model=LLMResponse)
async def respond(payload: LLMRequest, db: AsyncSession = Depends(get_db)):
    """
    1. Depends(get_db)를 통해 비동기 DB 세션을 주입받습니다.
    2. 생성된 db 세션과 payload의 user_id를 서비스 함수로 전달합니다.
    """
    try:
        # 서비스 함수 호출 (db와 user_id 전달)
        reply = await generate_bartender_reply(payload.user_id, payload.text, db)
    
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    except OpenAIError as exc:
        raise HTTPException(status_code=502, detail=f"OpenAI API error: {exc}")

    return LLMResponse(reply=reply)