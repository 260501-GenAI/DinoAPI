from typing import TypedDict

from langchain_ollama import ChatOllama

# This Service will define the State, Nodes, and Graph for our LangGraph implementation

# First, just wanna define the LLM we'll use
llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0.5
)

# This is the State object for our Graph
# Like in React, State holds data that we want to keep track of
# Each Node in the Graph can read from and write to the State
class GraphState(TypedDict, total=False): #total=False makes all fields optional

