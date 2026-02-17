from pydantic import BaseModel
from sqlalchemy import Column, Integer, String

from app.services.db_connection import Base


# This is the Class that defines and builds the table in the DB
class UserDBModel(Base):

    # Name the table
    __tablename__ = "users"

    # Define the columns, complete with constraints
    id = Column(Integer, primary_key=True) # Setting primary_key = True makes this autoincrement!
    username = Column(String, unique=True, nullable=False)
    password = Column(String, nullable=False)


# A Pydantic model for inserting users (no ID, string password instead of SecretStr)
class CreateUserModel(BaseModel):
    username:str
    password:str


    # TODO: another model for returning users (leave out the password)


