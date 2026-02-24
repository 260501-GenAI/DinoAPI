from typing import TypedDict, Any

from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph

from app.services.vectordb_service import search

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

    # Route to general chat node if no keywords get matched
    return {"route":"chat"}


# Node that gets Dino data from VectorDB
def search_dinos(state:GraphState) -> GraphState:

    # Simple similarity search like we've done before, using the query stored in state
    query = state.get("query", "")
    results = search("dino_docs", query, k=5)

    # Save the results in state!
    return {"docs":results}

# Node that gets Plans data from VectorDB
def search_plans(state:GraphState) -> GraphState:

    query = state.get("query", "")
    results = search("plans_docs", query, k=5)

    # Save the results in state!
    return {"docs":results}

# Node that uses the stored vectorDB docs to respond to the user
def answer_with_docs(state:GraphState) -> GraphState:

    # Ultimately, this node just invokes the LLM
    # The only difference is it's using the docs stored in state

    # First, extract the query and docs from state
    query = state.get("query", "")
    docs = state.get("docs", [])

    # Set up a prompt - basic, no real personality
    prompt=(
    f"""
    You are a friendly chatbot that takes search results from a VectorDB
    Answer the user's query in a concise but thorough way
    
    Search Results: {docs}
    User Query: {query}
    Answer: 
    """
    )

    # Invoke the LLM! And save the answer in state
    response = llm.invoke(prompt)
    return {"answer":response.text}

# Here's the general chat node that we fall back to if the query isn't related to vector data
def general_chat_node(state:GraphState) -> GraphState:

    # Get the user's query from state
    query = state.get("query", "")

    # Define the prompt
    prompt=(
        f"""
        You are a friendly chatbot that responds to general queries
        Answer the user's query in a concise but thorough way
        
        User Query: {query}
        Answer: 
        """
    )

    # Return the invocation and store it in State
    response = llm.invoke(prompt)
    return {"answer":response.text}



# =====================(END OF NODE DEFINITIONS)=======================

# The function that BUILDS OUR GRAPH - a Graph is just a series of steps
# The Nodes make up this "series of steps"
# So we have to define the execution order and branch points for these nodes
def build_graph():

    # First, define the graph builder using the State Graph
    build = StateGraph(GraphState)

    # Register each node
    build.add_node("route", route_node)
    build.add_node("search_dinos", search_dinos)
    build.add_node("search_plans", search_plans)
    build.add_node("answer_with_docs", answer_with_docs)

    # Set the node that starts the graph (router node in this case)
    build.set_entry_point("route")

    # Set up the branching nodes (conditional edges)
    # A conditional edge is a way to define nodes that MAY run based on a condition
    build.add_conditional_edges(
        "route",
        # This lambda function returns the key we use to choose an edge
        # We're just saying "use the route key" while avoiding writing a whole new function
        lambda state: state["route"],

        # Map that connects the possible "route" values to the appropriate note
        {
            "dinos":"search_dinos",
            "plans":"search_plans"
        }
    )

    # After either retrieval node, we ALWAYS want to go to the answer node
    build.add_edge("search_dinos", "answer_with_docs")
    build.add_edge("search_plans", "answer_with_docs")

    # Define potential terminal node (stopping points) for the graph
    build.set_finish_point("answer_with_docs")

    # Return the built graph!
    return build.compile()

# Make a single graph instance using the build_graph function
# This is what we'll invoke in our endpoints!
langgraph = build_graph()
