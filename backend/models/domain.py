from sqlalchemy import Column, Integer, String, Text, BigInteger, Date, DateTime, ForeignKey, JSON, UniqueConstraint
from sqlalchemy.sql import func
from pgvector.sqlalchemy import Vector
# database.py가 어디에 있는지에 따라 import 경로를 맞춰주세요.
from backend.database import Base

class User(Base):
    __tablename__ = "users"

    id = Column(BigInteger, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    email = Column(String(100), unique=True, index=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    created_at = Column(DateTime, server_default=func.now())

class EmotionDictionary(Base):
    __tablename__ = "emotion_dictionary"

    id = Column(Integer, primary_key=True)
    main_category = Column(String(20))
    sub_category = Column(String(50), unique=True)
    situation_example = Column(Text)
    cocktail_direction = Column(String(100))
    cocktail_color = Column(String(20))

class UserMemory(Base):
    __tablename__ = "user_memories"

    id = Column(BigInteger, primary_key=True)
    user_id = Column(BigInteger, ForeignKey("users.id"))
    memory_text = Column(Text)
    # pgvector 사용을 위한 설정
    embedding = Column(Vector(1536)) 
    main_category = Column(String(20))
    sub_category = Column(String(50))
    emotion_intensity = Column(Integer, default=50)
    created_at = Column(DateTime, server_default=func.now())

class EmotionReceipt(Base):
    __tablename__ = "emotion_receipts"

    id = Column(BigInteger, primary_key=True)
    user_id = Column(BigInteger, ForeignKey("users.id"))
    receipt_date = Column(Date, server_default=func.current_date())
    weather = Column(String(50))
    dominant_sub_category = Column(String(50))
    recommended_cocktail = Column(String(100))
    summary_note = Column(Text)

class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id = Column(BigInteger, primary_key=True)
    user_id = Column(BigInteger, ForeignKey("users.id"))
    role = Column(String(20), nullable=False) # 'user' 또는 'assistant'
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, server_default=func.now())

# --- 🚀 새로 추가된 모바일 건강 데이터 테이블 ---
class HealthMetric(Base):
    __tablename__ = "health_metrics"

    id = Column(BigInteger, primary_key=True)
    user_id = Column(BigInteger, ForeignKey("users.id"), nullable=False) # 기존 users 테이블과 연결!
    record_date = Column(Date, nullable=False)
    step_count = Column(Integer, default=0)
    sleep_minutes = Column(Integer, default=0)
    screen_time_minutes = Column(Integer, default=0)
    app_usage_json = Column(JSON, default=dict)  # 유연한 JSON 형태 저장
    depression_score = Column(Integer, nullable=True)

    # 한 유저당 하루에 하나의 기록만 있도록 제한 (Supabase에 설정한 제약조건과 동일)
    __table_args__ = (
        UniqueConstraint('user_id', 'record_date', name='uq_health_user_date'),
    )