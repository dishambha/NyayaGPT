import argparse
import concurrent.futures
import io
import os
import re
import sys
from pathlib import Path
from typing import Callable, Optional

import fitz  # PyMuPDF  # type: ignore[import-not-found]
from dotenv import load_dotenv  # type: ignore[import-not-found]
from openai import OpenAI  # type: ignore[import-not-found]


load_dotenv()

DEFAULT_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEFAULT_MODEL = os.getenv("MODEL_NAME", "deepseek-chat")
ACTIVE_MODEL = DEFAULT_MODEL
_client = None

# ==========================================
# 1. SETUP & DEEPSEEK CLIENT
# ==========================================


def get_client():
    """Return a cached DeepSeek client configured from environment variables."""
    global _client

    if _client is None:
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError(
                "DEEPSEEK_API_KEY is not set. Add it to your .env file before running the script."
            )

        _client = OpenAI(api_key=api_key, base_url=DEFAULT_BASE_URL)

    return _client

def call_deepseek(prompt, system_message="You are a helpful assistant.", model=None):
    """Basic wrapper to call the DeepSeek API."""
    response = get_client().chat.completions.create(
        model=model or ACTIVE_MODEL,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ],
        temperature=0.1 # Low temperature for analytical tasks
    )
    return response.choices[0].message.content

# ==========================================
# 2. HELPER TOOLS FOR THE AGENT
# ==========================================
def llm_batch(prompts, system_message="You are a sub-LLM. Extract concise, relevant information based on the prompt."):
    """Processes a list of prompts in parallel using Sub-LLMs."""
    if not prompts:
        return []

    results = [None] * len(prompts)
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = {
            executor.submit(call_deepseek, prompt, system_message): index
            for index, prompt in enumerate(prompts)
        }

        for future in concurrent.futures.as_completed(futures):
            index = futures[future]
            try:
                results[index] = future.result()
            except Exception as e:
                results[index] = f"Error in sub-LLM: {str(e)}"
    return results

SUPPORTED_EXTENSIONS = {".pdf", ".md", ".txt", ".doc", ".docx"}


def read_document_text(file_path):
    """Read text from a supported local document file."""
    suffix = Path(file_path).suffix.lower()

    if suffix == ".pdf":
        try:
            doc = fitz.open(file_path)
            full_text = ""
            for page in doc:
                full_text += page.get_text("text") + "\n"

            if not full_text.strip():
                return "", "Error: No readable text found. The PDF might be a scanned image."

            return full_text, None
        except Exception as e:
            return "", f"Error reading PDF: {str(e)}"

    if suffix in {".md", ".txt"}:
        try:
            with open(file_path, "r", encoding="utf-8") as handle:
                return handle.read(), None
        except UnicodeDecodeError:
            try:
                with open(file_path, "r", encoding="utf-8-sig") as handle:
                    return handle.read(), None
            except Exception as e:
                return "", f"Error reading text file: {str(e)}"
        except Exception as e:
            return "", f"Error reading text file: {str(e)}"

    if suffix in {".doc", ".docx"}:
        try:
            doc = fitz.open(file_path)
            full_text = ""
            for page in doc:
                full_text += page.get_text("text") + "\n"

            if not full_text.strip():
                return "", "Error: No readable text found in the Office document."

            return full_text, None
        except Exception as e:
            return "", f"Error reading Office document: {str(e)}"

    return "", f"Unsupported file type: {suffix or 'unknown'}"


def chunk_text(text, chunk_size=4000, overlap=500):
    """Split text into overlapping chunks."""
    if not text.strip():
        return []

    step = max(1, chunk_size - overlap)
    return [text[i:i + chunk_size] for i in range(0, len(text), step)]


def get_document_chunks(file_path, chunk_size=4000, overlap=500):
    """Extract text from a supported file and return overlapping chunks with metadata."""
    text, error = read_document_text(file_path)
    if error:
        return [
            {
                "file_path": file_path,
                "file_name": os.path.basename(file_path),
                "file_type": Path(file_path).suffix.lower().lstrip("."),
                "chunk_index": 0,
                "chunk_text": error,
                "text": error,
                "source": os.path.basename(file_path),
                "page": None,
                "error": True,
            }
        ]

    chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
    return [
        {
            "file_path": file_path,
            "file_name": os.path.basename(file_path),
            "file_type": Path(file_path).suffix.lower().lstrip("."),
            "chunk_index": index,
            "chunk_text": chunk,
            "text": chunk,
            "source": os.path.basename(file_path),
            "page": None,
            "error": False,
        }
        for index, chunk in enumerate(chunks)
    ]


def get_document_chunks_for_files(file_paths, chunk_size=4000, overlap=500):
    """Extract chunks for many supported files and preserve source metadata."""
    all_chunks = []

    for file_path in file_paths:
        all_chunks.extend(get_document_chunks(file_path, chunk_size=chunk_size, overlap=overlap))

    return all_chunks

# ==========================================
# 3. MAIN RLM AGENT
# ==========================================
def run_rlm_agent(
    file_paths,
    user_question,
    max_turns=10,
    model=DEFAULT_MODEL,
    progress_callback: Optional[Callable[[str, str], None]] = None,
):
    global ACTIVE_MODEL
    ACTIVE_MODEL = model

    api_client = get_client()

    def emit(event_type: str, message: str) -> None:
        if progress_callback is not None:
            progress_callback(event_type, message)

    repl_locals = {
        "fitz": fitz,
        "llm_batch": llm_batch,
        "read_document_text": read_document_text,
        "chunk_text": chunk_text,
        "get_document_chunks": get_document_chunks,
        "get_document_chunks_for_files": get_document_chunks_for_files,
        "file_paths": file_paths,
        "answer": {"content": "", "ready": False}
    }

    system_prompt = f"""You are a Recursive Language Model (RLM).
You are tasked with answering a user's question about one or more local documents.
You DO NOT have the PDF text in your context window. Instead, you have access to a Python environment.

Available variables in your environment:
- `file_paths`: A list of strings containing the paths to the local files.
- `fitz`: The PyMuPDF library (already imported).
- `llm_batch(prompts)`: A function that takes a list of string prompts and runs them in parallel through sub-LLMs.
- `read_document_text(file_path)`: Extract text from a supported local file.
- `chunk_text(text, chunk_size=4000, overlap=500)`: Split text into overlapping chunks.
- `get_document_chunks(file_path, chunk_size=4000, overlap=500)`: Extract chunks from one supported file with metadata.
- `get_document_chunks_for_files(file_paths, chunk_size=4000, overlap=500)`: Extract chunks from many supported files with metadata.
- `answer`: A dictionary {{ "content": "", "ready": False }}.

YOUR STRATEGY:
1. Use `chunks = get_document_chunks_for_files(file_paths)` to safely extract the text.
2. Keep track of which chunk came from which file using the metadata fields in each chunk dictionary.
3. Construct prompts for the sub-LLMs to search for the answer in those chunks.
4. Call `results = llm_batch(prompts)`.
5. Print the results so you can see them in the next turn (`print(results)`).
6. Once you know the final answer, write it to `answer["content"]` and set `answer["ready"] = True`.

Always output your Python code inside a ```python ``` block.
Only write code. Wait for the execution output before writing the next step.
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"The user asks: {user_question}"}
    ]

    start_message = f"--- Starting RLM Agent for: '{user_question}' ---"
    print(start_message)
    emit("status", start_message)

    for turn in range(max_turns):
        turn_message = f"\n[Turn {turn+1}] Main RLM is thinking..."
        print(turn_message)
        emit("turn", turn_message)

        response = api_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.0
        ).choices[0].message.content

        messages.append({"role": "assistant", "content": response})

        code_match = re.search(r"```python\n(.*?)\n```", response, re.DOTALL)
        if not code_match:
            print("Main LLM did not output Python code. Ending early.")
            break

        code_to_run = code_match.group(1)
        print(f"\n[Executing Code]:\n{code_to_run}\n")
        emit("status", f"Executing code for turn {turn + 1}.")

        old_stdout = sys.stdout
        redirected_output = sys.stdout = io.StringIO()

        try:
            exec(code_to_run, globals(), repl_locals)
            execution_output = redirected_output.getvalue()
        except Exception as e:
            execution_output = f"Python Error: {str(e)}"
        finally:
            sys.stdout = old_stdout

        if execution_output.strip():
            emit("output", execution_output)

        if repl_locals["answer"]["ready"]:
            finish_message = "\n[RLM Finished] Answer Ready!"
            print(finish_message)
            emit("final", repl_locals["answer"]["content"])
            return repl_locals["answer"]["content"]

        print(f"[Execution Output]:\n{execution_output[:500]}... (truncated if long)")
        messages.append({"role": "user", "content": f"Execution Output:\n{execution_output}\nWhat is your next step?"})

    failure_message = "Failed to find answer within maximum turns."
    emit("error", failure_message)
    return failure_message

# ==========================================
# 4. EXECUTION
# ==========================================
def parse_args():
    """Parse command-line arguments for the RLM runner."""
    parser = argparse.ArgumentParser(description="Run the recursive language model over one or more local documents.")
    parser.add_argument(
        "--files",
        nargs="+",
        required=True,
        help="One or more local files to analyze. Supported types: .pdf, .md, .txt.",
    )
    parser.add_argument(
        "--question",
        required=True,
        help="Question to answer from the provided files.",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=10,
        help="Maximum recursive turns before giving up.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="DeepSeek chat model to use for the main loop and sub-LLM calls.",
    )
    return parser.parse_args()


def main():
    """CLI entry point for the RLM runner."""
    args = parse_args()

    missing_files = [file_path for file_path in args.files if not os.path.exists(file_path)]
    if missing_files:
        print(f"ERROR: Cannot find: {', '.join(missing_files)}. Please provide valid local file paths.")
        return 1

    unsupported_files = [file_path for file_path in args.files if Path(file_path).suffix.lower() not in SUPPORTED_EXTENSIONS]
    if unsupported_files:
        supported_list = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        print(f"ERROR: Unsupported file type in: {', '.join(unsupported_files)}. Supported types are: {supported_list}.")
        return 1

    final_answer = run_rlm_agent(
        args.files,
        args.question,
        max_turns=args.max_turns,
        model=args.model,
    )

    print("\n================ FINAL ANSWER ================")
    print(final_answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())