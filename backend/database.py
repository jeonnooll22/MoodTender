import os
from dotenv import load_dotenv # 1. 추가
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base

# .env 파일 불러오기
load_dotenv() 

# 2. .env에서 접속 주소 가져오기
DATABASE_URL = os.getenv("DATABASE_URL")

# 1. 비동기 엔진 생성
engine = create_async_engine(
    DATABASE_URL, 
    echo=True, 
    future=True
)

# 2. 비동기 세션 생성기
AsyncSessionLocal = async_sessionmaker(
    bind=engine, 
    class_=AsyncSession, 
    expire_on_commit=False
)

# 3. 모델 베이스
Base = declarative_base()

# 4. FastAPI 의존성 함수 (비동기)
async def get_db():
    async with AsyncSessionLocal() as session:
        yield session
        await session.close()