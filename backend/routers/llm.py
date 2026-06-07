from fastapi import APIRouter, HTTPException

from models.schemas import LLMRequest, LLMResponse
from services.openai_llm import OpenAIError, generate_bartender_reply

router = APIRouter()


@router.post("/llm/respond", response_model=LLMResponse)
async def respond(payload: LLMRequest):
    try:
        reply = await generate_bartender_reply(payload.text)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    except OpenAIError as exc:
        raise HTTPException(status_code=502, detail=f"OpenAI API error: {exc}")

    return LLMResponse(reply=reply)
