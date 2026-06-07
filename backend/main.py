# coding: utf-8
"""MoodTender — FastAPI 백엔드 (모듈화)"""

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from config import FRONTEND_DIR
from database import engine, Base
from routers import auth, generation, llm, model_status

# ─── DB 테이블 초기화 ─────────────────────────────────────────
Base.metadata.create_all(bind=engine)

# ─── FastAPI 앱 ───────────────────────────────────────────────
app = FastAPI(title="MoodTender API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

# ─── 라우터 ───────────────────────────────────────────────────
app.include_router(auth.router,         prefix="/api", tags=["Auth"])
app.include_router(model_status.router, prefix="/api", tags=["Model"])
app.include_router(generation.router,   prefix="/api", tags=["Generation"])
app.include_router(llm.router,          prefix="/api", tags=["LLM"])

# ─── 프론트엔드 ───────────────────────────────────────────────
@app.get("/")
async def index():
    return FileResponse(FRONTEND_DIR / "index.html")

@app.get("/login")
async def login_page():
    return FileResponse(FRONTEND_DIR / "login.html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=7862, reload=False)
