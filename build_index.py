import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# --- Configuration ---
BNS_PDF_PATH = "BNS.pdf"
BNSS_PDF_PATH = "BNSS.pdf"
BNS_INDEX_PATH = "faiss_index_bns"
BNSS_INDEX_PATH = "faiss_index_bnss"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

def build_and_save_index(pdf_path, index_path):
    """Loads a PDF, processes it, and saves the FAISS index to disk."""
    if not os.path.exists(pdf_path):
        print(f"Error: PDF file not found at {pdf_path}")
        return

    print(f"Loading document: {pdf_path}...")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    print("Splitting text into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)

    print(f"Loading embedding model: {EMBEDDING_MODEL}...")
    # Note: The first time this runs, it will download the model.
    embeddings_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    print("Creating FAISS index from documents... (This may take a while)")
    vector_store = FAISS.from_documents(docs, embeddings_model)

    print(f"Saving index to: {index_path}...")
    vector_store.save_local(index_path)
    print(f"Index for {os.path.basename(pdf_path)} saved successfully!")

if __name__ == "__main__":
    print("--- Starting Knowledge Base Build Process ---")
    
    # Build and save the index for the BNS document
    build_and_save_index(BNS_PDF_PATH, BNS_INDEX_PATH)
    
    print("-" * 20)
    
    # Build and save the index for the BNSS document
    build_and_save_index(BNSS_PDF_PATH, BNSS_INDEX_PATH)
    
    print("\n--- All indexes have been built and saved. You can now run the Streamlit app. ---")