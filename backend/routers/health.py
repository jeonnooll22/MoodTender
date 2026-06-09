from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import IntegrityError
from backend.database import get_db
from backend.models import schemas, domain

router = APIRouter(
    prefix="/api/mobile",
    tags=["Mobile Data"]
)

@router.post("/health-data")
async def save_health_data(
    data: schemas.HealthDataCreate, 
    db: AsyncSession = Depends(get_db)
):
    # 🚨 임시: JWT 연동 전까지는 테스트를 위해 user_id를 1로 고정합니다.
    # 나중에 JWT 토큰에서 user_id를 빼오는 로직으로 수정할 예정입니다.
    current_user_id = 1 

    new_metric = domain.HealthMetric(
        user_id=current_user_id,
        record_date=data.record_date,
        step_count=data.step_count,
        sleep_minutes=data.sleep_minutes,
        screen_time_minutes=data.screen_time_minutes,
        app_usage_json=data.app_usage_json,
        depression_score=data.depression_score
    )

    db.add(new_metric)
    
    try:
        await db.commit()
        return {"status": "success", "message": "모바일 건강 데이터가 성공적으로 저장되었습니다."}
    except IntegrityError:
        await db.rollback()
        # UNIQUE(user_id, record_date) 제약 조건 때문에 같은 날짜에 두 번 보내면 에러가 납니다.
        raise HTTPException(status_code=400, detail="오늘 날짜의 데이터가 이미 존재합니다.")
    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=500, detail=str(e))