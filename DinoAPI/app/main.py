from fastapi import FastAPI

from app.routers import dino_router, user_router
from app.services.db_connection import Base, engine

# Create the DB tables on startup (if they don't already exist)
Base.metadata.create_all(bind=engine)

# Set up our FastAPI instance.
# This "app" variable will be used to do FastAPI stuff like defining endpoints and routers
app = FastAPI()

# REGISTER my routers (so they actually show up in SwaggerUI)
app.include_router(dino_router.router)
app.include_router(user_router.router)

# Generic sample endpoint (greeting GET request)
@app.get("/")
async def sample_endpoint():
    return {"message":"Hello from FastAPI!"}