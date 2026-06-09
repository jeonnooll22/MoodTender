from pydantic import BaseModel
from datetime import date
from typing import Dict, Optional

# --- 기존 유저 및 인증 스키마 ---
class UserCreate(BaseModel):
    username: str
    password: str

class UserResponse(BaseModel):
    id: int
    username: str
    class Config:
        from_attributes = True

class Token(BaseModel):
    access_token: str
    token_type: str

# --- 기존 LLM 및 대화 스키마 ---
class LLMRequest(BaseModel):
    user_id: int
    text: str

class LLMResponse(BaseModel):
    reply: str
    
class ChatRequest(BaseModel):
    user_id: int
    text: str

# --- 🚀 새로 추가된 모바일 건강 데이터 스키마 ---
class HealthDataCreate(BaseModel):
    record_date: date               # 데이터 기록 날짜 (예: "2026-06-09")
    step_count: int                # 걸음 수
    sleep_minutes: int             # 수면 시간 (분 단위)
    screen_time_minutes: int       # 총 스마트폰 사용 시간 (분 단위)
    # 카카오톡, 유튜브, 메시지, 전화 등의 사용 시간을 {"kakao": 120, "phone": 30} 형태로 저장
    app_usage_json: Dict[str, int] 
    depression_score: Optional[int] = None  # 우울 수치 (필요 시 선택적 입력)