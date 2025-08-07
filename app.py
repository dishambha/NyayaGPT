import streamlit as st
import os
import time
from dotenv import load_dotenv

# --- All your existing LangChain imports ---
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.tools.retriever import create_retriever_tool
from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.messages import HumanMessage, AIMessage

# --- App Configuration ---
st.set_page_config(page_title="Legal Chat Agent", page_icon="⚖️", layout="wide")
st.title("⚖️ Agentic Legal Assistant")
st.caption("Query the BNS and BNSS documents using an advanced AI agent.")

# --- API Key Setup ---
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    st.error("Groq API key not found. Please create a .env file with GROQ_API_KEY='your_key'.")
    st.stop()

# --- Caching the Agent Creation ---
@st.cache_resource
def build_agent():
    """Builds the agent by loading pre-built FAISS indexes."""
    st.info("Loading knowledge base... This should be quick!")

    # Define paths and embedding model
    BNS_INDEX_PATH = "faiss_index_bns"
    BNSS_INDEX_PATH = "faiss_index_bnss"
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"

    # Check if index files exist
    if not os.path.exists(BNS_INDEX_PATH) or not os.path.exists(BNSS_INDEX_PATH):
        st.error("FAISS index files not found. Please run the `build_index.py` script first.")
        return None
    
    # Load the embedding model
    embeddings_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    # Load the pre-built FAISS indexes
    bns_vector_store = FAISS.load_local(BNS_INDEX_PATH, embeddings_model, allow_dangerous_deserialization=True)
    bnss_vector_store = FAISS.load_local(BNSS_INDEX_PATH, embeddings_model, allow_dangerous_deserialization=True)

    # Create retriever tools from the loaded indexes
    bns_retriever = bns_vector_store.as_retriever(search_kwargs={"k": 4})
    bnss_retriever = bnss_vector_store.as_retriever(search_kwargs={"k": 4})
    
    bns_tool = create_retriever_tool(bns_retriever, "bns_law_search", "Searches the Bharatiya Nyaya Sanhita (BNS) for definitions of crimes and punishments.")
    bnss_tool = create_retriever_tool(bnss_retriever, "bnss_procedure_search", "Searches the Bharatiya Nagarik Suraksha Sanhita (BNSS) for legal procedures.")
    tools = [bns_tool, bnss_tool]
    
    # Create the agent prompt
    prompt = hub.pull("hwchase17/react-chat")
    new_rules = """

**VERY IMPORTANT RULES FOR YOUR FINAL ANSWER**:
1. Structure your response clearly with markdown. Use bold headings for each part of the user's question (e.g., **Punishment for Assault (BNS)** and **Reporting Procedure (BNSS)**).
2. Under each heading, first provide a concise, one-sentence summary of the answer.
3. After the summary, add a bullet point labeled "**Source Quote:**" and provide the single, most relevant quote from the source text that supports your summary. Keep the quote as short as possible.
"""
    prompt.template += new_rules
    
    # Create the agent
    llm = ChatGroq(model_name="llama3-70b-8192", temperature=0)
    agent = create_react_agent(llm, tools, prompt)
    st.success("Agent is ready!")
    
    # --- THIS IS THE FIX ---
    # Add handle_parsing_errors=True to make the agent more robust
    return AgentExecutor(agent=agent, tools=tools, verbose=False, handle_parsing_errors=True)

# Build the agent
agent_executor = build_agent()

if agent_executor is None:
    st.stop()

# --- Chat History Management ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! How can I help you with the BNS or BNSS today?"}]

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
            agent_chat_history = [msg for msg in st.session_state.messages if msg['role'] != 'user']
            response = agent_executor.invoke({"input": prompt, "chat_history": agent_chat_history})
            
            output = response["output"]
            full_response = ""
            message_placeholder = st.empty()
            for chunk in output.split():
                full_response += chunk + " "
                time.sleep(0.02)
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": output})