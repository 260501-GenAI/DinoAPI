# Check dino_model for more detailed notes on what a Model is
from typing import Annotated

from pydantic import BaseModel, Field, SecretStr


class UserModel(BaseModel):

    id: Annotated[int, Field(gt=0)] = None
    username: Annotated[str, Field(min_length=5, max_length=15)]
    password: Annotated[SecretStr, Field(min_length=8)]

    # SecretStr is a cool Pydantic type that hides the value when printed