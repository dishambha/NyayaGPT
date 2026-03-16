import os
from functools import lru_cache
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any

# --- All your existing LangChain imports ---
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.tools.retriever import create_retriever_tool
from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor

# --- FastAPI App ---
app = FastAPI(
    title="Legal Chat Agent",
    description="Query the BNS and BNSS documents using an advanced AI agent.",
)

# --- API Key Setup ---
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise RuntimeError(
        "Groq API key not found. Please create a .env file with GROQ_API_KEY='your_key'."
    )


# --- Function to check if FAISS index files exist ---
def index_exists(index_path):
    return (
        os.path.isdir(index_path)
        and os.path.isfile(os.path.join(index_path, "index.faiss"))
        and os.path.isfile(os.path.join(index_path, "index.pkl"))
    )


# --- Caching the Agent Creation ---
@lru_cache(maxsize=1)
def build_agent():
    """Builds the agent by loading pre-built FAISS indexes."""

    # Define paths and embedding model
    BNS_INDEX_PATH = "faiss_index_bns"
    BNSS_INDEX_PATH = "faiss_index_bnss"
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"

    # Check if FAISS index files exist
    if not index_exists(BNS_INDEX_PATH) or not index_exists(BNSS_INDEX_PATH):
        raise RuntimeError(
            "FAISS index files not found. Please run the `build_index.py` script first."
        )

    # Load the embedding model
    embeddings_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    # Load the pre-built FAISS indexes
    bns_vector_store = FAISS.load_local(
        BNS_INDEX_PATH, embeddings_model, allow_dangerous_deserialization=True
    )
    bnss_vector_store = FAISS.load_local(
        BNSS_INDEX_PATH, embeddings_model, allow_dangerous_deserialization=True
    )

    # Create retriever tools from the loaded indexes
    bns_retriever = bns_vector_store.as_retriever(search_kwargs={"k": 4})
    bnss_retriever = bnss_vector_store.as_retriever(search_kwargs={"k": 4})

    bns_tool = create_retriever_tool(
        bns_retriever,
        "bns_law_search",
        "Searches the Bharatiya Nyaya Sanhita (BNS) for definitions of crimes and punishments.",
    )
    bnss_tool = create_retriever_tool(
        bnss_retriever,
        "bnss_procedure_search",
        "Searches the Bharatiya Nagarik Suraksha Sanhita (BNSS) for legal procedures.",
    )
    tools = [bns_tool, bnss_tool]

    # Create the agent prompt
    prompt = hub.pull("hwchase17/react-chat")
    new_rules = """
**VERY IMPORTANT RULES FOR YOUR FINAL ANSWER**:
1. Structure your response clearly with markdown. Use bold headings for each part of the user's question (e.g., **Punishment for Assault (BNS)** and **Reporting Procedure (BNSS)**).
2. Under each heading, first provide a concise, one-sentence summary of the answer.
3. After the summary, add a bullet point labeled "**Source Quote:**" and provide the single, most relevant quote from the source text that supports your summary. Keep the quote as short as possible.
4. Only answer based on the information retrieved from the provided documents (BNS and BNSS). Do not use any external knowledge or assumptions. If no relevant information is found in the retrieved data, respond with: "There is no data available for this query."
"""
    prompt.template += new_rules

    # Create the agent
    llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0)
    agent = create_react_agent(llm, tools, prompt)

    return AgentExecutor(
        agent=agent, tools=tools, verbose=False, handle_parsing_errors=True
    )


# --- Pydantic Models ---
class ChatRequest(BaseModel):
    message: str
    history: List[Dict[str, Any]] = []


class ChatResponse(BaseModel):
    response: str


# --- Endpoints ---
@app.on_event("startup")
async def startup_event():
    # Build the agent on startup
    global agent_executor
    agent_executor = build_agent()


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    # Prepare chat history (exclude user messages as per original logic)
    agent_chat_history = [msg for msg in request.history if msg.get("role") != "user"]

    try:
        response = agent_executor.invoke(
            {"input": request.message, "chat_history": agent_chat_history}
        )
        output = response["output"]
        return ChatResponse(response=output)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error processing request: {str(e)}"
        )


@app.get("/")
async def root():
    return {
        "message": "Welcome to the Legal Chat Agent API. Use POST /chat to interact."
    }
