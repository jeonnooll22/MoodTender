from pathlib import Path
import re
from openai import AsyncOpenAI, OpenAIError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import desc

# 기존 설정값 유지
from config import OPENAI_API_KEY, OPENAI_MODEL
from backend.models.domain import ChatMessage, UserMemory, EmotionDictionary

# --- 기존 프롬프트 설정 유지 ---
BASE_DIR = Path(__file__).resolve().parents[2]
AGENT_PROMPT_PATHS = (BASE_DIR / "agent_ko.md", BASE_DIR / "agent.md")

DEFAULT_SYSTEM_PROMPT = """너는 MoodTender다. 사용자의 지친 마음을 위로하는 따뜻한 한국어 AI 바텐더다."""
ENDPOINT_SYSTEM_PROMPT = """
[필수 규칙]
1. 채팅 설명문이 아니라 바로 말하는 대사다.
2. 각 문장은 35자 내외의 짧은 구어체로 쓴다.
3. 마크다운, 따옴표, 이모지, 무대 지시문은 쓰지 않는다.
4. 사용자가 쓴 밈은 설명하지 말고 감정만 먼저 받아준다.
"""

# --- 로직 함수들 ---

def _compose_system_prompt(prompt: str, context: str = "", cocktail_info: str = "") -> str:
    """프롬프트를 조합하여 시스템 메시지 생성"""
    base = f"{prompt.strip()}\n\n[과거 기억 및 대화 맥락]\n{context}\n\n[추천 칵테일 정보]\n{cocktail_info}\n\n{ENDPOINT_SYSTEM_PROMPT.strip()}"
    return base

# [기존 helper 유지]
def _clean_reply(reply: str) -> str:
    reply = reply.strip()
    reply = re.sub(r"^[\s>*#\-•\d.]+", "", reply)
    reply = re.sub(r"\s{2,}", " ", reply)
    reply = reply.replace('"', "").replace("'", "")
    return reply.strip()

def _limit_reply_sentences(reply: str, user_text: str) -> str:
    max_sentences = 3 if any(w in user_text for w in ["죽고", "사라지고", "끝내고"]) else 2
    sentences = re.findall(r"[^.!?。！？]+[.!?。！？]?", reply)
    sentences = [s.strip() for s in sentences if s.strip()]
    return " ".join(sentences[:max_sentences]) if len(sentences) > max_sentences else reply

# --- DB 연동 및 RAG 로직 추가 ---

async def _get_chat_history(db: AsyncSession, user_id: int) -> str:
    """최근 대화 10개를 불러와 텍스트로 변환"""
    stmt = select(ChatMessage).where(ChatMessage.user_id == user_id).order_by(desc(ChatMessage.created_at)).limit(10)
    result = await db.execute(stmt)
    history = result.scalars().all()[::-1]
    return "\n".join([f"{m.role}: {m.content}" for m in history])

async def _get_emotion_category(client: AsyncOpenAI, history: str) -> str:
    """대화 맥락을 읽고 감정 카테고리 추출"""
    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": "대화 맥락을 분석해 감정 카테고리 하나만 말해. [기쁨, 우울, 불안, 분노, 지침, 외로움, 평온]"},
                  {"role": "user", "content": history}]
    )
    return response.choices[0].message.content.strip()

async def _get_cocktail_data(db: AsyncSession, category: str) -> str:
    """DB에서 감정별 칵테일 정보 조회"""
    stmt = select(EmotionDictionary).where(EmotionDictionary.main_category == category)
    result = await db.execute(stmt)
    data = result.scalars().first()
    return f"색상: {data.cocktail_color}, 방향: {data.cocktail_direction}" if data else "일반적인 칵테일"

# --- 메인 실행 함수 ---

async def generate_bartender_reply(user_id: int, user_text: str, db: AsyncSession) -> str:
    text = user_text.strip()
    if not text: return "..."

    client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    
    # 1. DB에 유저 말 저장
    db.add(ChatMessage(user_id=user_id, role="user", content=text))
    await db.commit()

    # 2. 컨텍스트 빌드
    history = await _get_chat_history(db, user_id)
    emotion_cat = await _get_emotion_category(client, history)
    cocktail_info = await _get_cocktail_data(db, emotion_cat)
    
    # 3. 프롬프트 구성
    sys_prompt = _compose_system_prompt(DEFAULT_SYSTEM_PROMPT, history, cocktail_info)
    
    try:
        response = await client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": text}]
        )
    except OpenAIError:
        return "조금 쉬고 싶으신가요? 제가 곁에 있을게요."

    reply = _clean_reply(response.choices[0].message.content or "")
    reply = _limit_reply_sentences(reply, text)
    
    # 4. 바텐더 말 저장
    db.add(ChatMessage(user_id=user_id, role="assistant", content=reply))
    await db.commit()

    return reply