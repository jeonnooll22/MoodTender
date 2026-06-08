# coding: utf-8
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

# 1. 경로 수정 (backend 폴더를 참조하도록 변경)
from config import FRONTEND_DIR
from backend.database import engine, Base
from backend.routers import auth, generation, llm, model_status, stt

# ─── FastAPI 앱 ───────────────────────────────────────────────
app = FastAPI(title="MoodTender API")

# 2. 비동기 DB 테이블 초기화 (서버 시작 시 실행)
@app.on_event("startup")
async def startup():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

# ─── 미들웨어 ────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── 정적 파일 ───────────────────────────────────────────────
# (config.py가 루트에 있다면 그대로 두시고, backend 안에 있다면 from backend.config import 로 수정하세요)
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

# ─── 라우터 ───────────────────────────────────────────────────
app.include_router(auth.router,         prefix="/api", tags=["Auth"])
app.include_router(model_status.router, prefix="/api", tags=["Model"])
app.include_router(generation.router,   prefix="/api", tags=["Generation"])
app.include_router(llm.router,          prefix="/api", tags=["LLM"])
app.include_router(stt.router,          prefix="/api", tags=["STT"])

# ─── 프론트엔드 ───────────────────────────────────────────────
@app.get("/")
async def index():
    return FileResponse(FRONTEND_DIR / "index.html")

@app.get("/login")
async def login_page():
    return FileResponse(FRONTEND_DIR / "login.html")

if __name__ == "__main__":
    import uvicorn
    # 3. 포트 및 실행 옵션
    uvicorn.run("main:app", host="127.0.0.1", port=7862, reload=False)