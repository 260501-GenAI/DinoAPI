from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.models.user_db_model import UserDBModel, CreateUserModel
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
async def create_user(new_user:CreateUserModel, db: Session = Depends(get_db)):

    # Extract the incoming user data into a format that the DB can accept
    # **? this unpacks the data into a dict which we convert to a UserDBModel
    user = UserDBModel(**new_user.model_dump())

    # Add and commit the new user to the DB
    db.add(user)
    db.commit()

    # Refresh the user variable, which overwrites it with what went into the DB
    db.refresh(user)

    return user # Send the new User back to the client (SwaggerUI in this case)


# Get all users
@router.get("/")
async def get_all_users(db: Session = Depends(get_db)):
    return db.query(UserDBModel).all()