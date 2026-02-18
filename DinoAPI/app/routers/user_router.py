from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.models.user_db_model import UserDBModel, CreateUserModel
from app.models.user_model import UserModel
from app.services.db_connection import get_db
from app.services.langchain_service import get_basic_chain

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

# Update user by ID
@router.put("/{user_id}")
async def update_user(user_id:int, updated_user: CreateUserModel, db: Session = Depends(get_db)):

    # First, we need to check if the user exists
    user = db.query(UserDBModel).filter(UserDBModel.id == user_id).first()

    if not user:
        raise HTTPException(status_code=404, detail=f"User with ID {user_id} not found!")

    # If the user exists, we can update their info with the incoming data
    user.username = updated_user.username
    user.password = updated_user.password

    # Commit the changes to the DB
    db.commit()
    db.refresh(user)

    # Return the user!
    return user


# Delete user by ID
@router.delete("/{user_id}")
async def delete_user(user_id:int, db: Session = Depends(get_db)):
    # Check if the user exists
    user = db.query(UserDBModel).filter(UserDBModel.id == user_id).first()

    if not user:
        raise HTTPException(status_code=404, detail=f"User with ID {user_id} not found!")

    # If the user exists, delete them
    db.delete(user)
    db.commit()

    return {"message": f"User with ID {user_id} deleted!"}


# RAG (Retrieval Augmented Generated) with our LLM and user data
# AUGMENTING the GENERATED response based on some data we're RETRIEVING
@router.post("/rag")
async def users_rag(user_input:str, db: Session = Depends(get_db)):

    # NOTE: we didn't make user_input a Pydantic model
    # ...which is fine, but the user's question will come in as a query param
        # (Instead of a value in the request body)

    # Get all users, convert the info to a string (easier for the LLM)
    users = db.query(UserDBModel).all()
    user_info = "\n".join([f"ID: {user.id}, Username: {user.username}" for user in users])
    # This^ looks like: ID: 1, Username: user1

    # Get the basic chain from the langchain service
    chain = get_basic_chain()

    # Ask the LLM a question based on that user info
    response = chain.invoke(
        {"input": f"""Here is some information about users in our database: {user_info}
            Based on this information, answer the user's query: {user_input} """}
    )

    # Return the LLM's response!
    return response

