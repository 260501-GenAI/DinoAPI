from fastapi import APIRouter

from app.models.dino_model import DinoModel

# Remember, routers are how we EXPOSE HTTP ENDPOINTS
# So this router will be full of functions that:
    # Take in HTTP requests
    # Return HTTP responses

# Every Router needs to get set up like this
router = APIRouter(
    prefix="/dinos", # HTTP requests ending in /dinos will get directed to this router
    tags=["dinos"] # This routers endpoints will be under "dinos" in the SwaggerUI docs
)

# Dinky Python map database (check the user_router for endpoints that hit a REAL DB)
dino_database = {
    1: DinoModel(id=1, species="T Rex", period="Cretaceous"),
    2: DinoModel(id=2, species="Velociraptor", period="Cretaceous")
}

# Some endpoints-------------------

# GET all dinos
@router.get("/")
async def get_all_dinos():
    return dino_database


# POST a new dino to the DB - note the use of our DinoModel in the args
@router.post("/", status_code=201) # 201 CREATED - good for successful data insertion
async def create_dino(dino:DinoModel):

    # TODO: some input validation would be good. Unique names only?

    # Give the dino an auto-incremented ID (just length of map + 1)
    dino.id = len(dino_database) + 1

    # Store the dino in the DB
    dino_database[dino.id] = dino

    return {
        "message": dino.species + " created!",
        "inserted_dino": dino
    }

# GET a certain amount of dinos (query param)
# This will return 2 dinos unless specified in the request
@router.get("/some_dinos")
async def get_some_dinos(limit:int=2):

    # Turn the map into a list and return the specified amount
    return list(dino_database.values())[:limit]