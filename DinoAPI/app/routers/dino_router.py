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


# Some endpoints-------------------

# GET all dinos
@router.get("/")
async def get_all_dinos():
    return "[here's where I'd return my dinos... IF I HAD SOME]"


# POST a new dino to the DB - note the use of our DinoModel in the args
@router.post("/", status_code=201) # 201 CREATED - good for successful data insertion
async def create_dino(dino:DinoModel):
    return dino.species + " created!"

# GET dino by ID (path param)