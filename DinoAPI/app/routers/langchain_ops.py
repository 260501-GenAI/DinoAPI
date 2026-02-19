from fastapi import APIRouter
from langchain_community.document_loaders import TextLoader
from pydantic import BaseModel

from app.services.langchain_service import get_basic_chain, get_sequential_chain, get_memory_chain

# Same old router setup
router = APIRouter(
    prefix="/langchain",
    tags=["langchain"]
)

# I'm going to make a quick Pydantic model that will represent the user's input
# This helps it play nice with FastAPI
class ChatInputModel(BaseModel):
    input:str

# Import the chains we defined in the Service for use in the endpoints below
basic_chain = get_basic_chain()
refined_answer_chain = get_sequential_chain()
memory_chain = get_memory_chain()

# General chat endpoint with no memory or any other fancy features
@router.post("/chat")
async def general_chat(chat:ChatInputModel):
    # Now we just invoke the chain with the user's input!
    return basic_chain.invoke(input={"input":chat.input})

# A DOCUMENT LOADING EXAMPLE - summarizing a txt file about a hypothetical dino fight
@router.get("/summarize")
async def summarize_dino_fight():

    # Use LangChain's TextLoader to load in the .txt file
    loader = TextLoader("app/DinoFightToSummarize.txt")

    # Extract the text into a LangChain Document object
    doc = loader.load()
    text = doc[0].page_content # Just a string with the .txt file's content

    # Invoke the LLM and return the summary thanks to a basic prompt
    return basic_chain.invoke(input={"input": f"Summarize this text: {text}"})

# This endpoint is for the more professional chat using our sequential chain
@router.post("/refined-chat")
async def refined_chat(chat:ChatInputModel):
    return refined_answer_chain.invoke(input={"input":chat.input})

# This endpoint is just a chat endpoint WITH MEMORY!
@router.post("/memory-chat")
async def memory_chat(chat:ChatInputModel):
    # Just a one liner - The chain will remember the last "k" interactions automatically
    return memory_chain.invoke(input={"input":chat.input})