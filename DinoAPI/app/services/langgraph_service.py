from typing import TypedDict, Any

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
    query:str # The user's input to the graph
    route:str # The "routing decision" we make. This tells the app what to invoke next
    docs:list[dict[str, Any]] # Results returned from VectorDB searches
    answer:str # The LLM's answer to the user's query
    #TODO: memory manager field

# ========================(NODE DEFINITIONS)============================

# Think of Nodes like steps in our Graph. Each Node has a specific responsibility.
# Nodes have read/write access to the fields in State

# Our first node - The ROUTING Node
# The user will pass in a query, and depending on what they're asking, go to:
    # A node that searches the dino_docs VectorDB collection
    # A node that searches the plans_docs VectorDB collection
    # A node that just does general chat (query unrelated to dinos/plans)
def route_node(state:GraphState) -> GraphState:

    # Get the user's query from state (stored for us when the graph is invoked)
    query = state.get("query", "") # Default to empty string if query is not set

    # VERY basic keyword matching (for now) to decide the route
    # Later, we'll let the LLM decide which route to go down
    if any(word in query for word in ["dino", "dinosaur", "dinosaurs"]):
        return {"route":"dinos"}

    if any(word in query for word in ["plan", "plans", "boss", "digs"]):
        return {"route":"plans"}

# Node that gets Dino data from VectorDB


# Node that gets Plans data from VectorDB


# =====================(END OF NODE DEFINITIONS)=======================