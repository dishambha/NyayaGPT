import streamlit as st
import os
import time
from dotenv import load_dotenv

# --- All your existing LangChain imports ---
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.tools.retriever import create_retriever_tool
from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor

# --- App Configuration ---
st.set_page_config(page_title="Legal Chat Agent", page_icon="⚖️", layout="wide")
st.title("⚖️ Agentic Legal Assistant")
st.caption("Query the BNS and BNSS documents using an advanced AI agent.")

# --- API Key Setup ---
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    st.error(
        "Groq API key not found. Please create a .env file with GROQ_API_KEY='your_key'."
    )
    st.stop()


# --- Function to check if FAISS index files exist ---
def index_exists(index_path):
    return (
        os.path.isdir(index_path)
        and os.path.isfile(os.path.join(index_path, "index.faiss"))
        and os.path.isfile(os.path.join(index_path, "index.pkl"))
    )


# --- Caching the Agent Creation ---
@st.cache_resource
def build_agent():
    """Builds the agent by loading pre-built FAISS indexes."""
    st.info("Loading knowledge base... This should be quick!")

    # Define paths and embedding model
    BNS_INDEX_PATH = "faiss_index_bns"
    BNSS_INDEX_PATH = "faiss_index_bnss"
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"

    # Check if FAISS index files exist
    if not index_exists(BNS_INDEX_PATH) or not index_exists(BNSS_INDEX_PATH):
        st.error(
            "FAISS index files not found. Please run the `build_index.py` script first."
        )
        return None

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
\n\nSYSTEM INSTRUCTION: RESPONSE FORMAT AND QUALITY RULES

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

    # Create the agent
    llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0)
    agent = create_react_agent(llm, tools, prompt)
    st.success("Agent is ready!")

    return AgentExecutor(
        agent=agent, tools=tools, verbose=False, handle_parsing_errors=True
    )


# Build the agent
agent_executor = build_agent()

if agent_executor is None:
    st.stop()

# --- Chat History Management ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello! How can I help you with the BNS or BNSS today?",
        }
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Handle User Input ---
if prompt := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("The agent is thinking..."):
            agent_chat_history = [
                msg for msg in st.session_state.messages if msg["role"] != "user"
            ]
            response = agent_executor.invoke(
                {"input": prompt, "chat_history": agent_chat_history}
            )

            output = response["output"]
            full_response = ""
            message_placeholder = st.empty()
            for chunk in output.split():
                full_response += chunk + " "
                time.sleep(0.02)
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": output})
