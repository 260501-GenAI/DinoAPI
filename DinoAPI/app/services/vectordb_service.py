from langchain_community.embeddings import OllamaEmbeddings

# This service will help us initialize and interact with a ChromaDB vector store
# Remember, chromaDB is just a type VectorDB. There are other like pinecone

PERSIST_DIRECTORY = "app/chroma_store" # This is where our vectorDB will live

# The vector embedding model we installed
# DIFFERENT from our LLM! This one specializes in turning text into vectors
EMBEDDING = OllamaEmbeddings(model="nomic-embed-text")

