from typing import Annotated

from pydantic import BaseModel, Field


# This MODEL class helps us MODEL data that we can use in the app.
# A model class is kind of like making a custom data type.

# FastAPI models are based on Pydantic's BaseModel.
# It comes with a bunch of useful features like data validation
    # We also need to use BaseModel for compatibility with FastAPI's routers

class DinoModel(BaseModel):
    # gt = 0: this value must be greater than 0
    # = None: this field is options when creating a dino
    id: Annotated[int, Field(gt=0)] = None
    species: str
    period: str

    # I won't validate the other fields super hard - look at the user_model for more examples
