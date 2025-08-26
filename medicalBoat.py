# streamlit_app.py
# -----------------------------------------------------------
# RAG Chat over your PDFs with FAISS + HuggingFace Endpoint
# (Only Chat UI visible; config + indexing run in backend)
# -----------------------------------------------------------

import asyncio
try:
    asyncio.set_event_loop(asyncio.new_event_loop())
except:
    pass



import os
from pathlib import Path
from typing import List, Optional

import streamlit as st
from dotenv import load_dotenv

# LangChain / HF
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import (
    HuggingFaceEmbeddings,
    HuggingFaceEndpoint,
    ChatHuggingFace,
)
from langchain.chains import RetrievalQA
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import Document


# -----------------------------
# Page config + hide chrome/sidebar
# -----------------------------
st.set_page_config(page_title="RAG Chat", page_icon="💬", layout="wide")

st.markdown(
    """
    <style>
      /* Hide Streamlit sidebar entirely (no config visible) */
      [data-testid="stSidebar"] {display: none !important;}
      /* Optional: hide default menu/footer/header if you want a clean chat surface */
      #MainMenu, header, footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("""
    <style>
        .header-container {
            text-align: center;
            margin-bottom: 20px;
        }
        .logo-circle {
            width: 120px;
            height: 120px;
            border-radius: 50%;
            background: radial-gradient(circle, #04d9ff, #0077ff);
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 0 auto 15px;
            box-shadow: 0 0 30px rgba(0, 119, 255, 0.6);
            font-size: 50px; /* Emoji size */
        }
        .gradient-text {
            font-size: 28px;
            font-weight: 800;
            background: linear-gradient(90deg, #00d4ff, #7f00ff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            display: inline-block;
        }
    </style>

    <div class="header-container">
        <div class="logo-circle">
            🧑‍⚕️
        </div>
        <div class="gradient-text">🔎 RAG Chat over your PDFst</div>
    </div>
""", unsafe_allow_html=True)




# -----------------------------
# Paths
# -----------------------------
DATA_DIR = Path("data")
UPLOAD_DIR = DATA_DIR / "uploaded"
VECTOR_DIR = Path("vectorstore/db_faiss")
for p in [DATA_DIR, UPLOAD_DIR, VECTOR_DIR.parent]:
    p.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Defaults / ENV config (backend only)
# -----------------------------
load_dotenv()  # loads .env if present

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_HF_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"

def get_hf_token() -> Optional[str]:
    # Safe access: if secrets is missing, fall back to env
    try:
        return st.secrets["HF_TOKEN"]
    except Exception:
        return os.getenv("HF_TOKEN")

HF_TOKEN = get_hf_token()
HF_REPO_ID = os.getenv("HF_REPO_ID", DEFAULT_HF_REPO_ID)
MODEL_TEMP = float(os.getenv("MODEL_TEMP", "0.10"))
MAX_NEW_TOKENS = int(os.getenv("MAX_TOKENS", "512"))
TOP_P = float(os.getenv("TOP_P", "0.9"))
REP_PENALTY = float(os.getenv("REP_PENALTY", "1.05"))
TOP_K = int(os.getenv("TOP_K", "3"))

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))

SYSTEM_MSG = os.getenv(
    "SYSTEM_MSG",
    "You are a helpful medical RAG assistant. Use ONLY the provided context. "
    "If the answer is not in the context, say you don't know. Be concise."
)

AUTO_BUILD_INDEX = os.getenv("AUTO_BUILD_INDEX", "1") == "1"  # build if missing


# if not HF_TOKEN:
#     st.error("HF_TOKEN is missing! Set it in .env or Streamlit Secrets.")
# else:
#     st.success("HF_TOKEN loaded successfully.")


# -----------------------------
# Caching helpers (backend)
# -----------------------------
@st.cache_resource(show_spinner=False)
def get_embeddings():
    return HuggingFaceEmbeddings(model_name=DEFAULT_EMBEDDING_MODEL)

@st.cache_resource(show_spinner=False)
def get_text_splitter(csize: int, coverlap: int):
    return RecursiveCharacterTextSplitter(chunk_size=csize, chunk_overlap=coverlap)

@st.cache_resource(show_spinner=False)
def build_prompt(system_msg: str):
    return ChatPromptTemplate.from_messages([
        ("system", system_msg),
        ("human", "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:")
    ])

@st.cache_resource(show_spinner=False)
def init_llm(repo_id: str, token: str, temperature: float,
             max_new_tokens: int, top_p: float, repetition_penalty: float):
    if not token:
        raise ValueError("HF_TOKEN is required via .env / secrets / environment.")
    endpoint_llm = HuggingFaceEndpoint(
        repo_id=repo_id,
        task="conversational",
        huggingfacehub_api_token=token,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
    )
    return ChatHuggingFace(llm=endpoint_llm)

@st.cache_resource(show_spinner=False)
def load_vectorstore(path: Path, _embeddings: HuggingFaceEmbeddings):
    """Load FAISS if exists; else return None.
       `_embeddings` name prevents Streamlit hashing error.
    """
    try:
        if path.exists():
            return FAISS.load_local(str(path), _embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        # Keep UI clean; log server-side
        print(f"[load_vectorstore] warning: {e}")
    return None


# -----------------------------
# Backend: PDF ingestion & (auto)index build — NO UI
# -----------------------------
def load_all_local_pdfs() -> List[Path]:
    return sorted([*UPLOAD_DIR.glob("*.pdf"), *DATA_DIR.glob("*.pdf")])

def load_documents_from_paths(paths: List[Path]) -> List[Document]:
    docs: List[Document] = []
    for p in paths:
        try:
            loader = PyPDFLoader(str(p))
            docs.extend(loader.load())
        except Exception as e:
            print(f"[PDF read error] {p.name}: {e}")
    return docs

def ensure_index() -> Optional[FAISS]:
    """Build index if not present (backend only), return loaded FAISS db."""
    embeddings = get_embeddings()
    db = load_vectorstore(VECTOR_DIR, embeddings)
    if db is not None:
        return db

    if not AUTO_BUILD_INDEX:
        return None

    pdfs = load_all_local_pdfs()
    if not pdfs:
        print("[ensure_index] No PDFs found in data/ or data/uploaded/.")
        return None

    # Build without exposing any UI components
    documents = load_documents_from_paths(pdfs)
    if not documents:
        print("[ensure_index] No readable PDF pages.")
        return None

    splitter = get_text_splitter(CHUNK_SIZE, CHUNK_OVERLAP)
    chunks = splitter.split_documents(documents)
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(str(VECTOR_DIR))
    # Clear cached loader so subsequent calls pick fresh index
    load_vectorstore.clear()
    print(f"[ensure_index] Built FAISS index with {len(chunks)} chunks from {len(pdfs)} PDF(s).")
    return db


# -----------------------------
# Retrieval / QA chain (backend)
# -----------------------------
embeddings = get_embeddings()
db = load_vectorstore(VECTOR_DIR, embeddings) or ensure_index()

if db is None:
    st.warning("No index found and no PDFs to build from. "
               "Place PDFs in `data/` or `data/uploaded/` and reload.")
    st.stop()

retriever = db.as_retriever(search_kwargs={"k": int(TOP_K)})

chat_llm = init_llm(HF_REPO_ID, HF_TOKEN or "", MODEL_TEMP, int(MAX_NEW_TOKENS), TOP_P, REP_PENALTY)
prompt = build_prompt(SYSTEM_MSG)

qa_chain = RetrievalQA.from_chain_type(
    llm=chat_llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt},
)


# -----------------------------
# Chat UI ONLY
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Optional tiny clear button inside chat area
if st.button("Clear Chat"):
    st.session_state.messages = []
    st.experimental_rerun()

# Show history
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("sources"):
            with st.expander("Sources"):
                for i, s in enumerate(m["sources"], 1):
                    src_name = s.metadata.get("source") or s.metadata.get("file_path") or "Unknown"
                    page = s.metadata.get("page", "N/A")
                    preview = s.page_content[:500].strip().replace("\n", " ")
                    st.markdown(f"**{i}. {src_name} (page {page})**\n\n> {preview}...")

# Input + response
user_query = st.chat_input("Ask your question…")
if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        try:
            resp = qa_chain.invoke({"query": user_query})
            answer = (resp.get("result") or "").strip()
            sources = resp.get("source_documents", []) or []
        except Exception as e:
            answer, sources = f"Error: {e}", []
        st.markdown(answer or "_No answer returned._")
        if sources:
            with st.expander("Sources"):
                for i, s in enumerate(sources, 1):
                    src_name = s.metadata.get("source") or s.metadata.get("file_path") or "Unknown"
                    page = s.metadata.get("page", "N/A")
                    preview = s.page_content[:500].strip().replace("\n", " ")
                    st.markdown(f"**{i}. {src_name} (page {page})**\n\n> {preview}...")

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
