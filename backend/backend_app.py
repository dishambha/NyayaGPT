import os
from pathlib import Path
from functools import lru_cache
from typing import Any, AsyncGenerator, Dict, List

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools.retriever import create_retriever_tool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS

# --- Token and context controls ---
RETRIEVER_K = int(os.getenv("RETRIEVER_K", "1"))
MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "300"))
MAX_INPUT_CHARS = int(os.getenv("MAX_INPUT_CHARS", "2200"))
MAX_HISTORY_MESSAGES = int(os.getenv("MAX_HISTORY_MESSAGES", "3"))
MAX_TOTAL_INPUT_TOKENS = int(os.getenv("MAX_TOTAL_INPUT_TOKENS", "2800"))
TOKEN_CHARS_RATIO = int(os.getenv("TOKEN_CHARS_RATIO", "4"))
MODEL_NAME = os.getenv("MODEL_NAME", "deepseek-chat").strip()
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
AGENT_MAX_ITERATIONS = int(os.getenv("AGENT_MAX_ITERATIONS", "8"))
AGENT_MAX_EXECUTION_SECONDS = float(os.getenv("AGENT_MAX_EXECUTION_SECONDS", "0"))
AGENT_STOP_SENTINEL = "Agent stopped due to iteration limit or time limit."
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- FastAPI App ---
app = FastAPI(
    title="Legal Chat Agent",
    description="Query the BNS and BNSS documents using an advanced AI agent.",
)

# --- API Key Setup ---
load_dotenv()
deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
if not deepseek_api_key:
    raise RuntimeError(
        "DeepSeek API key not found. Please create a .env file with DEEPSEEK_API_KEY='your_key'."
    )

# Allow your separate frontend to call this API.
allowed_origins = [
    origin.strip()
    for origin in os.getenv("ALLOWED_ORIGINS", "*").split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def index_exists(index_path: str) -> bool:
    return (
        os.path.isdir(index_path)
        and os.path.isfile(os.path.join(index_path, "index.faiss"))
        and os.path.isfile(os.path.join(index_path, "index.pkl"))
    )


@lru_cache(maxsize=1)
def build_agent() -> AgentExecutor:
    """Build retriever tools and a single model-backed agent executor."""

    bns_index_path = PROJECT_ROOT / "faiss_index_bns"
    bnss_index_path = PROJECT_ROOT / "faiss_index_bnss"
    embedding_model = "all-MiniLM-L6-v2"

    if not index_exists(str(bns_index_path)) or not index_exists(str(bnss_index_path)):
        raise RuntimeError(
            "FAISS index files not found. Please run `python build_index.py` from the project root."
        )

    embeddings_model = HuggingFaceEmbeddings(model_name=embedding_model)

    bns_vector_store = FAISS.load_local(
        str(bns_index_path), embeddings_model, allow_dangerous_deserialization=True
    )
    bnss_vector_store = FAISS.load_local(
        str(bnss_index_path), embeddings_model, allow_dangerous_deserialization=True
    )

    bns_retriever = bns_vector_store.as_retriever(search_kwargs={"k": RETRIEVER_K})
    bnss_retriever = bnss_vector_store.as_retriever(search_kwargs={"k": RETRIEVER_K})

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

    prompt = hub.pull("hwchase17/react-chat")
    new_rules = """


SYSTEM INSTRUCTION: RESPONSE FORMAT AND QUALITY RULES

You are a legal retrieval assistant that MUST answer strictly from the retrieved BNS and BNSS context.

Follow these rules exactly:
1. Use clean markdown with clear section headings.
2. Split the answer by user intent. If the question has multiple parts, create one heading per part.
3. Under each heading, output the following blocks in this exact order:
   - Summary: one short sentence.
   - Key Points: 2 to 4 concise bullet points.
   - Source Quote: one short, verbatim quote from retrieved text.
4. After all sections, add a final heading named "Conclusion" with a 1 to 2 sentence wrap-up.
5. Do not include chain-of-thought, tool usage, or internal reasoning.
6. Do not use external knowledge. If retrieval is missing or insufficient, write exactly:
   "There is no data available for this query."
7. When possible, mention whether the quote comes from BNS or BNSS in plain text.
8. Keep language simple, precise, and non-speculative.
9. Do not loop on tool calls. Use at most one or two tool calls before writing the final answer.

Output template:
## <Part of the user question>
Summary: <one sentence>
Key Points:
- <point 1>
- <point 2>
Source Quote: "<short exact quote>" (<BNS or BNSS>)

## Conclusion
<1 to 2 sentence final answer>
"""
    prompt.template += new_rules

    llm = ChatOpenAI(
        model=MODEL_NAME,
        temperature=0,
        max_tokens=MAX_OUTPUT_TOKENS,
        api_key=deepseek_api_key,
        base_url=DEEPSEEK_BASE_URL,
    )
    agent = create_react_agent(llm, tools, prompt)

    max_execution_time = (
        AGENT_MAX_EXECUTION_SECONDS if AGENT_MAX_EXECUTION_SECONDS > 0 else None
    )

    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,
        handle_parsing_errors=True,
        max_iterations=AGENT_MAX_ITERATIONS,
        max_execution_time=max_execution_time,
        early_stopping_method="force",
    )


class ChatRequest(BaseModel):
    message: str
    history: List[Dict[str, Any]] = []


class ChatResponse(BaseModel):
    response: str


def normalize_agent_output(output: str) -> str:
    if not isinstance(output, str):
        return "There is no data available for this query."
    if AGENT_STOP_SENTINEL in output:
        return (
            "I could not complete this query within the current agent limits. "
            "Please retry with a more specific question."
        )
    return output


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, (len(text) + TOKEN_CHARS_RATIO - 1) // TOKEN_CHARS_RATIO)


def trim_to_tokens(text: str, max_tokens: int) -> str:
    if max_tokens <= 0:
        return ""
    max_chars = max_tokens * TOKEN_CHARS_RATIO
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


def build_payload(request: ChatRequest) -> Dict[str, Any]:
    user_message = request.message.strip()
    if not user_message:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    user_message = user_message[:MAX_INPUT_CHARS]
    max_user_tokens = max(300, int(MAX_TOTAL_INPUT_TOKENS * 0.5))
    user_message = trim_to_tokens(user_message, max_user_tokens)

    raw_history = [msg for msg in request.history if msg.get("role") != "user"]
    raw_history = raw_history[-MAX_HISTORY_MESSAGES:]

    # Keep only as much history as fits under the total input token budget.
    remaining_tokens = MAX_TOTAL_INPUT_TOKENS - estimate_tokens(user_message)
    agent_chat_history: List[Dict[str, Any]] = []

    for item in reversed(raw_history):
        content = item.get("content")
        if not isinstance(content, str):
            continue

        trimmed = trim_to_tokens(content[:MAX_INPUT_CHARS], min(remaining_tokens, 300))
        needed = estimate_tokens(trimmed)
        if needed <= 0 or needed > remaining_tokens:
            continue

        msg = dict(item)
        msg["content"] = trimmed
        agent_chat_history.append(msg)
        remaining_tokens -= needed

        if remaining_tokens <= 0:
            break

    agent_chat_history.reverse()

    return {"input": user_message, "chat_history": agent_chat_history}


@app.on_event("startup")
async def startup_event() -> None:
    global agent_executor
    agent_executor = build_agent()


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    payload = build_payload(request)

    try:
        response = await agent_executor.ainvoke(payload)
        return ChatResponse(response=normalize_agent_output(response.get("output", "")))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Model {MODEL_NAME} failed: {str(e)}",
        )


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest) -> StreamingResponse:
    payload = build_payload(request)

    async def event_stream() -> AsyncGenerator[str, None]:
        streamed_any = False
        try:
            async for event in agent_executor.astream_events(payload, version="v1"):
                if event.get("event") != "on_chat_model_stream":
                    continue

                chunk = event.get("data", {}).get("chunk")
                content = getattr(chunk, "content", "") if chunk else ""
                if isinstance(content, list):
                    content = "".join(
                        part.get("text", "") if isinstance(part, dict) else str(part)
                        for part in content
                    )
                if not content:
                    continue

                if AGENT_STOP_SENTINEL in content:
                    content = (
                        "I could not complete this query within the current agent limits. "
                        "Please retry with a more specific question."
                    )

                streamed_any = True
                yield f"data: {content}\n\n"

            if not streamed_any:
                response = await agent_executor.ainvoke(payload)
                output = normalize_agent_output(response.get("output", ""))
                yield f"data: {output}\n\n"

            yield "event: done\ndata: [DONE]\n\n"
        except Exception as e:
            yield f"event: error\ndata: Model {MODEL_NAME} failed: {str(e)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


@app.get("/")
async def root() -> Dict[str, str]:
    return {
        "message": "Welcome to the Legal Chat Agent API. Use POST /chat to interact."
    }
