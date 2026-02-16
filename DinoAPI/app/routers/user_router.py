from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.models.user_model import UserModel
from app.services.db_connection import get_db

# Same old Router Setup
router = APIRouter(
    prefix="/users",
    tags=["users"]
)

# All of these functions use DEPENDENCY INJECTION to get access to a DB connection

# Insert User
@router.post("/")
async def create_user(new_user:UserModel, db: Session = Depends(get_db)):
    # You can check the sql_ops router in the EvilScientist API V2 for examples
    pass

