import os
import asyncio
import shutil
import tempfile
import json
import queue
import threading
from pathlib import Path
from functools import lru_cache
from typing import Any, AsyncGenerator, Dict, List
import importlib

from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools.retriever import create_retriever_tool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS

try:
    from backend.rlm_agent import SUPPORTED_EXTENSIONS, run_rlm_agent
except ModuleNotFoundError:
    from rlm_agent import SUPPORTED_EXTENSIONS, run_rlm_agent

# Load environment variables before reading config constants.
load_dotenv()

# --- Token and context controls ---
RETRIEVER_K = int(os.getenv("RETRIEVER_K", "1"))
MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "300"))
MAX_INPUT_CHARS = int(os.getenv("MAX_INPUT_CHARS", "2200"))
MAX_HISTORY_MESSAGES = int(os.getenv("MAX_HISTORY_MESSAGES", "3"))
MAX_TOTAL_INPUT_TOKENS = int(os.getenv("MAX_TOTAL_INPUT_TOKENS", "2800"))
TOKEN_CHARS_RATIO = int(os.getenv("TOKEN_CHARS_RATIO", "4"))
MODEL_NAME = os.getenv("MODEL_NAME", "deepseek-chat").strip()
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()
AGENT_MAX_ITERATIONS = int(os.getenv("AGENT_MAX_ITERATIONS", "8"))
AGENT_MAX_EXECUTION_SECONDS = float(os.getenv("AGENT_MAX_EXECUTION_SECONDS", "0"))
AGENT_STOP_SENTINEL = "Agent stopped due to iteration limit or time limit."
PROJECT_ROOT = Path(__file__).resolve().parent.parent

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "").strip()
PINECONE_INDEX_HOST = os.getenv("PINECONE_INDEX_HOST", "").strip()
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "__default__").strip()
PINECONE_FIELDS = [
    field.strip()
    for field in os.getenv(
        "PINECONE_FIELDS", "chunk_text,text,content,category,source,title"
    ).split(",")
    if field.strip()
]
SMARTCHAT_TOP_K = int(os.getenv("SMARTCHAT_TOP_K", "4"))
SMARTCHAT_MAX_CONTEXT_CHARS = int(os.getenv("SMARTCHAT_MAX_CONTEXT_CHARS", "6000"))
SMARTCHAT_MAX_OUTPUT_TOKENS = int(os.getenv("SMARTCHAT_MAX_OUTPUT_TOKENS", "180"))
SMARTCHAT_MAX_HITS = int(os.getenv("SMARTCHAT_MAX_HITS", "3"))
SMARTCHAT_HIT_TEXT_CHARS = int(os.getenv("SMARTCHAT_HIT_TEXT_CHARS", "240"))
RLM_CHAT_MAX_TURNS = int(os.getenv("RLM_CHAT_MAX_TURNS", "10"))

RESPONSE_RULES = """
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
"""

# --- FastAPI App ---
app = FastAPI(
    title="Legal Chat Agent",
    description="Query the BNS and BNSS documents using an advanced AI agent.",
)

# --- API Key Setup ---
deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
if not deepseek_api_key:
    raise RuntimeError(
        "DeepSeek API key not found. Please create a .env file with DEEPSEEK_API_KEY='your_key'."
    )

gemini_api_key = os.getenv("GEMINI_API_KEY", "").strip()

# Allow your separate frontend to call this API.
default_allowed_origins = {
    "https://nyayagptbot.vercel.app",
}
configured_origins = {
    origin.strip()
    for origin in os.getenv("ALLOWED_ORIGINS", "").split(",")
    if origin.strip()
}
if os.getenv("ALLOWED_ORIGINS", "").strip() == "*":
    allowed_origins = ["*"]
else:
    allowed_origins = sorted(default_allowed_origins | configured_origins)
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
def build_retrievers():
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

    return (
        bns_vector_store.as_retriever(search_kwargs={"k": RETRIEVER_K}),
        bnss_vector_store.as_retriever(search_kwargs={"k": RETRIEVER_K}),
    )


@lru_cache(maxsize=1)
def build_query_embedder() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


@lru_cache(maxsize=1)
def build_agent() -> AgentExecutor:
    """Build retriever tools and a single model-backed agent executor."""

    bns_retriever, bnss_retriever = build_retrievers()

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
    prompt.template += RESPONSE_RULES + "\n9. Do not loop on tool calls. Use at most one or two tool calls before writing the final answer.\n"

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


class SmartChatRequest(BaseModel):
    message: str
    history: List[Dict[str, Any]] = []


class SmartChatResponse(BaseModel):
    response: str
    hits: List[Dict[str, Any]]


class ModelChatRequest(BaseModel):
    message: str
    history: List[Dict[str, Any]] = []
    model: str = "deepseek"


class ModelChatResponse(BaseModel):
    model: str
    response: str
    sources: List[Dict[str, Any]]
    hits: List[Dict[str, Any]]


class RLMChatResponse(BaseModel):
    model: str
    response: str
    files: List[str]
    max_turns: int


def compact_hits(hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    compact: List[Dict[str, Any]] = []
    for hit in hits[:SMARTCHAT_MAX_HITS]:
        if not isinstance(hit, dict):
            continue
        fields = hit.get("fields", {}) if isinstance(hit.get("fields", {}), dict) else {}
        text_value = ""
        for key in ("chunk_text", "text", "content"):
            value = fields.get(key)
            if isinstance(value, str) and value.strip():
                text_value = value.strip()
                break

        compact.append(
            {
                "id": hit.get("_id", ""),
                "score": hit.get("_score", 0),
                "source": fields.get("source") or fields.get("title") or "unknown",
                "page": fields.get("page"),
                "text": text_value[:SMARTCHAT_HIT_TEXT_CHARS],
            }
        )
    return compact


def sanitize_uploaded_filename(filename: str, fallback_index: int) -> str:
    safe_name = Path(filename or "").name.strip()
    if not safe_name:
        return f"upload_{fallback_index}"
    return safe_name


async def save_uploaded_files_to_temp_dir(files: List[UploadFile], temp_dir: str) -> List[str]:
    saved_paths: List[str] = []
    used_names: set[str] = set()

    for index, upload in enumerate(files, start=1):
        extension = Path(upload.filename or "").suffix.lower()
        if extension not in SUPPORTED_EXTENSIONS:
            supported_list = ", ".join(sorted(SUPPORTED_EXTENSIONS))
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {upload.filename or 'unknown'}. Supported types are: {supported_list}.",
            )

        base_name = sanitize_uploaded_filename(upload.filename or "", index)
        candidate_name = base_name
        stem = Path(base_name).stem
        suffix = Path(base_name).suffix
        counter = 1
        while candidate_name in used_names:
            candidate_name = f"{stem}_{counter}{suffix}"
            counter += 1

        used_names.add(candidate_name)
        destination_path = os.path.join(temp_dir, candidate_name)

        content = await upload.read()
        with open(destination_path, "wb") as handle:
            handle.write(content)

        saved_paths.append(destination_path)

    return saved_paths


def build_sources_from_hits(hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    sources: List[Dict[str, Any]] = []
    for hit in hits[:SMARTCHAT_MAX_HITS]:
        if not isinstance(hit, dict):
            continue
        fields = hit.get("fields", {}) if isinstance(hit.get("fields", {}), dict) else {}
        sources.append(
            {
                "source": fields.get("source") or fields.get("title") or "unknown",
                "page": fields.get("page"),
                "score": hit.get("_score", 0),
            }
        )
    return sources


def normalize_model_name(model_name: str) -> str:
    value = (model_name or "deepseek").strip().lower()
    if value in {"deepseek", "gemini"}:
        return value
    raise HTTPException(status_code=400, detail="model must be either 'deepseek' or 'gemini'.")


def flatten_messages_for_gemini(messages: List[Dict[str, str]]) -> str:
    lines: List[str] = []
    for msg in messages:
        role = (msg.get("role") or "user").strip().upper()
        content = msg.get("content", "")
        if isinstance(content, str) and content.strip():
            lines.append(f"[{role}] {content.strip()}")
    return "\n\n".join(lines)


def complete_with_selected_model(messages: List[Dict[str, str]], model_name: str) -> str:
    selected = normalize_model_name(model_name)

    if selected == "deepseek":
        completion = deepseek_client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            stream=False,
            temperature=0,
            max_tokens=min(MAX_OUTPUT_TOKENS, SMARTCHAT_MAX_OUTPUT_TOKENS),
        )
        return completion.choices[0].message.content or ""

    if not gemini_api_key:
        raise HTTPException(
            status_code=503,
            detail="Gemini is not configured. Set GEMINI_API_KEY in backend/.env.",
        )

    genai_module = importlib.import_module("google.genai")
    types_module = importlib.import_module("google.genai.types")
    client = genai_module.Client(api_key=gemini_api_key)

    prompt = flatten_messages_for_gemini(messages)
    config = types_module.GenerateContentConfig(
        temperature=0,
        max_output_tokens=min(MAX_OUTPUT_TOKENS, SMARTCHAT_MAX_OUTPUT_TOKENS),
    )
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=config,
    )
    return (getattr(response, "text", "") or "").strip()


def build_stream_messages(payload: Dict[str, Any]) -> List[Dict[str, str]]:
    user_query = payload.get("input", "")
    bns_retriever, bnss_retriever = build_retrievers()

    bns_docs = bns_retriever.invoke(user_query)
    bnss_docs = bnss_retriever.invoke(user_query)

    context_parts: List[str] = []
    for doc in bns_docs:
        context_parts.append(f"[BNS] {doc.page_content}")
    for doc in bnss_docs:
        context_parts.append(f"[BNSS] {doc.page_content}")

    context_text = "\n\n".join(context_parts)
    if not context_text.strip():
        context_text = "No relevant context retrieved from BNS/BNSS."

    history = payload.get("chat_history", [])
    messages: List[Dict[str, str]] = [
        {
            "role": "system",
            "content": RESPONSE_RULES
            + "\nUse only this retrieved context:\n"
            + context_text,
        }
    ]

    for item in history:
        role = item.get("role", "assistant")
        content = item.get("content", "")
        if role not in {"system", "user", "assistant"}:
            role = "assistant"
        if isinstance(content, str) and content.strip():
            messages.append({"role": role, "content": content})

    messages.append({"role": "user", "content": user_query})
    return messages


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


def extract_pinecone_hits(results: Any) -> List[Dict[str, Any]]:
    if hasattr(results, "to_dict"):
        data = results.to_dict()
    elif isinstance(results, dict):
        data = results
    else:
        data = {}

    result = data.get("result", {}) if isinstance(data, dict) else {}
    hits = result.get("hits", []) if isinstance(result, dict) else []
    if isinstance(hits, list) and hits:
        return hits

    # query() shape fallback: {"matches": [...]}.
    matches = data.get("matches", []) if isinstance(data, dict) else []
    normalized: List[Dict[str, Any]] = []
    if isinstance(matches, list):
        for m in matches:
            if not isinstance(m, dict):
                continue
            normalized.append(
                {
                    "_id": m.get("id", ""),
                    "_score": m.get("score", 0),
                    "fields": m.get("metadata", {}) if isinstance(m.get("metadata", {}), dict) else {},
                }
            )
    return normalized


def build_context_from_hits(hits: List[Dict[str, Any]]) -> str:
    parts: List[str] = []
    for idx, hit in enumerate(hits, start=1):
        fields = hit.get("fields", {}) if isinstance(hit, dict) else {}
        chunk = ""
        for key in ("chunk_text", "text", "content"):
            value = fields.get(key)
            if isinstance(value, str) and value.strip():
                chunk = value.strip()
                break
        if not chunk:
            continue

        source = fields.get("source") or fields.get("title") or "unknown"
        parts.append(f"[{idx}] source={source}\n{chunk}")

    context = "\n\n".join(parts)
    return context[:SMARTCHAT_MAX_CONTEXT_CHARS]


@app.post("/rlm/chat", response_model=RLMChatResponse)
async def rlm_chat(
    question: str = Form(...),
    files: List[UploadFile] = File(...),
    max_turns: int = Form(RLM_CHAT_MAX_TURNS),
    model: str = Form(MODEL_NAME),
) -> RLMChatResponse:
    if not files:
        raise HTTPException(status_code=400, detail="At least one file must be uploaded.")

    temp_dir = tempfile.mkdtemp(prefix="nyayagpt_rlm_")
    saved_paths: List[str] = []

    try:
        saved_paths = await save_uploaded_files_to_temp_dir(files, temp_dir)
        response_text = run_rlm_agent(saved_paths, question, max_turns=max_turns, model=model)
        return RLMChatResponse(
            model=model,
            response=response_text,
            files=[Path(path).name for path in saved_paths],
            max_turns=max_turns,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RLM document chat failed: {str(e)}")
    finally:
        for upload in files:
            await upload.close()
        shutil.rmtree(temp_dir, ignore_errors=True)


@app.post("/rlm/chat/stream")
async def rlm_chat_stream(
    question: str = Form(...),
    files: List[UploadFile] = File(...),
    max_turns: int = Form(RLM_CHAT_MAX_TURNS),
    model: str = Form(MODEL_NAME),
) -> StreamingResponse:
    if not files:
        raise HTTPException(status_code=400, detail="At least one file must be uploaded.")

    temp_dir = tempfile.mkdtemp(prefix="nyayagpt_rlm_")
    saved_paths: List[str] = []
    try:
        saved_paths = await save_uploaded_files_to_temp_dir(files, temp_dir)
    finally:
        for upload in files:
            await upload.close()

    event_queue: queue.Queue = queue.Queue()
    sentinel = object()

    def push_event(event_type: str, message: str) -> None:
        event_queue.put(
            {
                "event": event_type,
                "message": message,
            }
        )

    def worker() -> None:
        try:
            event_queue.put({"event": "status", "message": f"Saved {len(saved_paths)} files. Starting RLM."})
            final_answer = run_rlm_agent(
                saved_paths,
                question,
                max_turns=max_turns,
                model=model,
                progress_callback=push_event,
            )
            event_queue.put(
                {
                    "event": "done",
                    "message": final_answer,
                    "model": model,
                    "files": [Path(path).name for path in saved_paths],
                    "max_turns": max_turns,
                }
            )
        except Exception as exc:
            event_queue.put({"event": "error", "message": str(exc)})
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
            event_queue.put(sentinel)

    threading.Thread(target=worker, daemon=True).start()

    async def event_stream() -> AsyncGenerator[str, None]:
        while True:
            item = await asyncio.to_thread(event_queue.get)
            if item is sentinel:
                break
            yield f"data: {json.dumps(item, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.on_event("startup")
async def startup_event() -> None:
    global agent_executor, deepseek_client, pinecone_index
    agent_executor = build_agent()
    deepseek_client = OpenAI(api_key=deepseek_api_key, base_url=DEEPSEEK_BASE_URL)
    pinecone_index = None
    if PINECONE_API_KEY and PINECONE_INDEX_HOST:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        pinecone_index = pc.Index(host=PINECONE_INDEX_HOST)


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
    messages = build_stream_messages(payload)

    async def event_stream() -> AsyncGenerator[str, None]:
        try:
            stream = deepseek_client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                stream=True,
                temperature=0,
                max_tokens=MAX_OUTPUT_TOKENS,
            )

            pending = ""
            for chunk in stream:
                delta = ""
                if chunk.choices and chunk.choices[0].delta:
                    delta = chunk.choices[0].delta.content or ""
                if not delta:
                    continue
                if AGENT_STOP_SENTINEL in delta:
                    delta = (
                        "I could not complete this query within the current agent limits. "
                        "Please retry with a more specific question."
                    )

                pending += delta
                # Flush in larger readable chunks to avoid fragmented words in the client.
                if len(pending) >= 48 or pending.endswith((" ", "\n", ".", ",", ":", ";", "?", "!")):
                    yield f"data: {pending}\n\n"
                    pending = ""

            if pending:
                yield f"data: {pending}\n\n"

            yield "event: done\ndata: [DONE]\n\n"
        except Exception as e:
            yield f"event: error\ndata: Model {MODEL_NAME} failed: {str(e)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/smartchat", response_model=SmartChatResponse)
async def smartchat(request: SmartChatRequest) -> SmartChatResponse:
    if pinecone_index is None:
        raise HTTPException(
            status_code=503,
            detail="Pinecone is not configured. Set PINECONE_API_KEY and PINECONE_INDEX_HOST.",
        )

    payload = build_payload(ChatRequest(message=request.message, history=request.history))

    top_k = max(1, min(SMARTCHAT_TOP_K, 20))
    namespace = (PINECONE_NAMESPACE or "__default__").strip()

    try:
        results = pinecone_index.search(
            namespace=namespace,
            query={"inputs": {"text": payload["input"]}, "top_k": top_k},
            fields=PINECONE_FIELDS,
        )
    except Exception as e:
        # If index has no integrated embedding, fallback to vector query.
        if "Integrated inference is not configured for this index" in str(e):
            try:
                query_vector = build_query_embedder().embed_query(payload["input"])
                results = pinecone_index.query(
                    namespace=namespace,
                    vector=query_vector,
                    top_k=top_k,
                    include_metadata=True,
                    include_values=False,
                )
            except Exception as q_err:
                raise HTTPException(
                    status_code=500,
                    detail=f"Pinecone vector query failed: {str(q_err)}",
                )
        else:
            raise HTTPException(status_code=500, detail=f"Pinecone search failed: {str(e)}")

    hits = extract_pinecone_hits(results)
    context_text = build_context_from_hits(hits)
    if not context_text:
        return SmartChatResponse(
            response="There is no data available for this query.",
            hits=hits,
        )

    messages: List[Dict[str, str]] = [
        {
            "role": "system",
            "content": RESPONSE_RULES
            + "\nFor /smartchat keep the final answer concise: maximum 120 words."
            + " If the user asks a single question, answer in one short section only."
            + "\nUse only this retrieved Pinecone context:\n"
            + context_text,
        }
    ]

    for item in payload.get("chat_history", []):
        role = item.get("role", "assistant")
        content = item.get("content", "")
        if role not in {"system", "user", "assistant"}:
            role = "assistant"
        if isinstance(content, str) and content.strip():
            messages.append({"role": role, "content": content})

    messages.append({"role": "user", "content": payload["input"]})

    try:
        content = complete_with_selected_model(messages, "deepseek")
        return SmartChatResponse(
            response=normalize_agent_output(content),
            hits=compact_hits(hits),
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Model {MODEL_NAME} failed during /smartchat: {str(e)}",
        )


@app.post("/modelchat", response_model=ModelChatResponse)
async def modelchat(request: ModelChatRequest) -> ModelChatResponse:
    if pinecone_index is None:
        raise HTTPException(
            status_code=503,
            detail="Pinecone is not configured. Set PINECONE_API_KEY and PINECONE_INDEX_HOST.",
        )

    payload = build_payload(ChatRequest(message=request.message, history=request.history))
    top_k = max(1, min(SMARTCHAT_TOP_K, 20))
    namespace = (PINECONE_NAMESPACE or "__default__").strip()

    try:
        results = pinecone_index.search(
            namespace=namespace,
            query={"inputs": {"text": payload["input"]}, "top_k": top_k},
            fields=PINECONE_FIELDS,
        )
    except Exception as e:
        if "Integrated inference is not configured for this index" in str(e):
            try:
                query_vector = build_query_embedder().embed_query(payload["input"])
                results = pinecone_index.query(
                    namespace=namespace,
                    vector=query_vector,
                    top_k=top_k,
                    include_metadata=True,
                    include_values=False,
                )
            except Exception as q_err:
                raise HTTPException(
                    status_code=500,
                    detail=f"Pinecone vector query failed: {str(q_err)}",
                )
        else:
            raise HTTPException(status_code=500, detail=f"Pinecone search failed: {str(e)}")

    hits = extract_pinecone_hits(results)
    context_text = build_context_from_hits(hits)
    if not context_text:
        return ModelChatResponse(
            model=normalize_model_name(request.model),
            response="There is no data available for this query.",
            sources=[],
            hits=compact_hits(hits),
        )

    messages: List[Dict[str, str]] = [
        {
            "role": "system",
            "content": RESPONSE_RULES
            + "\nFor /modelchat keep the final answer concise: maximum 120 words."
            + " Add a final 'Sources' section listing source file and page from provided context."
            + "\nUse only this retrieved Pinecone context:\n"
            + context_text,
        }
    ]

    for item in payload.get("chat_history", []):
        role = item.get("role", "assistant")
        content = item.get("content", "")
        if role not in {"system", "user", "assistant"}:
            role = "assistant"
        if isinstance(content, str) and content.strip():
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": payload["input"]})

    selected = normalize_model_name(request.model)
    try:
        content = complete_with_selected_model(messages, selected)
        return ModelChatResponse(
            model=selected,
            response=normalize_agent_output(content),
            sources=build_sources_from_hits(hits),
            hits=compact_hits(hits),
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Model {selected} failed during /modelchat: {str(e)}",
        )


@app.get("/")
async def root() -> Dict[str, str]:
    return {
        "message": "Welcome to the Legal Chat Agent API. Use POST /chat to interact."
    }
