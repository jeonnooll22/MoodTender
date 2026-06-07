from pathlib import Path
import re

from openai import AsyncOpenAI, OpenAIError

from config import OPENAI_API_KEY, OPENAI_MODEL

BASE_DIR = Path(__file__).resolve().parents[2]
AGENT_PROMPT_PATHS = (
    BASE_DIR / "agent_ko.md",
    BASE_DIR / "agent.md",
)

DEFAULT_SYSTEM_PROMPT = """
너는 MoodTender다. 사용자의 지친 마음을 위로하는 따뜻한 한국어 AI 바텐더다.
사용자의 감정을 먼저 받아주고, 영상에서 말하기 좋은 짧고 자연스러운 문장으로 답한다.
마크다운, 이모지, 따옴표, 무대 지시문은 쓰지 않는다.
"""

ENDPOINT_SYSTEM_PROMPT = """

---

## 현재 실행 중인 /llm/respond 최우선 규칙

위 MoodTender Agent Rules를 반드시 따른다.
이 응답은 채팅 설명문이 아니라 바로 TTS와 립싱크 영상에 들어갈 대사다.
사용자가 짧게 말하면 반드시 2문장 이내로 답한다.
각 문장은 짧고 자연스러운 한국어 구어체로 끝낸다.
마크다운 줄바꿈, 번호 목록, 따옴표, 이모지, 무대 지시문은 절대 쓰지 않는다.
사용자가 쓴 밈은 설명하지 말고 감정만 먼저 받아준다.
칵테일은 실제 음주 권유가 아니라 감정의 상징으로만 짧게 말한다.
"""


def _compose_system_prompt(prompt: str) -> str:
    return f"{prompt.strip()}\n\n{ENDPOINT_SYSTEM_PROMPT.strip()}"


def load_system_prompt() -> tuple[str, str]:
    for path in AGENT_PROMPT_PATHS:
        if path.exists():
            prompt = path.read_text(encoding="utf-8").strip()
            if prompt:
                return _compose_system_prompt(prompt), str(path)

    return _compose_system_prompt(DEFAULT_SYSTEM_PROMPT), "default"


SYSTEM_PROMPT, SYSTEM_PROMPT_SOURCE = load_system_prompt()
print(f"[LLM] SYSTEM_PROMPT loaded from {SYSTEM_PROMPT_SOURCE}")


def _fallback_reply(user_text: str) -> str:
    text = user_text.strip()
    if not text:
        return "오늘 마음이 아직 말없이 잔 안에 머물러 있네요. 괜찮아요, 천천히 한 모금씩 풀어봐요."

    if any(word in text for word in ["힘들", "지쳤", "피곤", "우울", "슬퍼"]):
        return "오늘은 마음이 오래 버틴 날 같아요. 따뜻한 잔 하나 내려놓듯, 여기서는 조금 쉬어가도 괜찮아요."
    if any(word in text for word in ["불안", "긴장", "걱정", "무서"]):
        return "마음이 자꾸 앞서 달리고 있네요. 지금은 숨을 한 번 고르고, 오늘의 속도를 조금 낮춰도 괜찮아요."
    if any(word in text for word in ["화", "짜증", "분노", "억울"]):
        return "오늘 안에 뜨거운 게 많이 쌓였나 봐요. 그 마음을 누르기보다, 잠깐 잔 밖으로 내려놓아도 좋아요."
    if any(word in text for word in ["좋", "기뻐", "행복", "성공", "축하"]):
        return "오늘은 기분 좋은 반짝임이 있는 날이네요. 그 순간이 금방 사라지지 않게, 천천히 음미해봐요."

    return "오늘의 마음은 한 가지 맛으로만 설명하기 어렵네요. 제가 조용히 옆에 있을 테니, 그 이야기를 조금 더 들려주세요."


def _clean_reply(reply: str) -> str:
    reply = reply.strip()
    reply = re.sub(r"^[\s>*#\-•\d.]+", "", reply)
    reply = re.sub(r"\s{2,}", " ", reply)
    reply = reply.replace('"', "").replace("'", "")
    return reply.strip()


def _is_safety_sensitive(text: str) -> bool:
    return any(
        word in text
        for word in [
            "죽고",
            "죽을",
            "죽고싶",
            "사라지고",
            "사라지고 싶",
            "리셋",
            "자살",
            "끝내고",
            "해치",
        ]
    )


def _limit_reply_sentences(reply: str, user_text: str) -> str:
    max_sentences = 3 if _is_safety_sensitive(user_text) else 2
    sentences = re.findall(r"[^.!?。！？]+[.!?。！？]?", reply)
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
    if len(sentences) <= max_sentences:
        return reply

    return " ".join(sentences[:max_sentences])


async def generate_bartender_reply(user_text: str) -> str:
    text = user_text.strip()
    if not text:
        raise ValueError("user_text is required.")

    if not OPENAI_API_KEY or OPENAI_API_KEY == "sk-...":
        return _fallback_reply(text)

    client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    try:
        response = await client.responses.create(
            model=OPENAI_MODEL,
            instructions=SYSTEM_PROMPT,
            input=text,
        )
    except OpenAIError as exc:
        message = str(exc).lower()
        if "insufficient_quota" in message or "429" in message:
            return _fallback_reply(text)
        raise

    reply = _clean_reply(response.output_text or "")
    reply = _limit_reply_sentences(reply, text)
    if not reply:
        return _fallback_reply(text)

    return reply


__all__ = ["OpenAIError", "generate_bartender_reply"]
