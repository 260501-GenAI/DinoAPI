from fastapi import APIRouter, Depends, HTTPException
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

# Get one user by ID (path param)
@router.get("/by_id/{user_id}")
async def get_user_by_id(user_id:int, db: Session = Depends(get_db)):
    # Filter the ResultSet to only include the user with the given id.
    # first() returns the first result of the query... there should only be one
    user = db.query(UserDBModel).filter(UserDBModel.id == user_id).first()

    # Some basic error handling for user not found
    if not user:
        raise HTTPException(status_code=404, detail=f"User with ID {user_id} not found!")

    # If the user is found, return it
    return user