import os
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_core.prompts import ChatPromptTemplate

# -----------------------------
# 1) Env & token
# -----------------------------
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("❌ HF_TOKEN missing. Add HF_TOKEN=<your_hf_pat> in .env")

# -----------------------------
# 2) HF Inference (Mistral chat)
#    NOTE: Mistral-7B-Instruct-v0.3 => task='conversational'
# -----------------------------
HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"

endpoint_llm = HuggingFaceEndpoint(
    repo_id=HUGGINGFACE_REPO_ID,
    task="conversational",                 # ✅ critical: this model supports conversational
    huggingfacehub_api_token=HF_TOKEN,
    temperature=0.1,
    max_new_tokens=512,
    top_p=0.9,
    repetition_penalty=1.05,
)

# Wrap endpoint into a ChatModel LangChain understands
chat_llm = ChatHuggingFace(llm=endpoint_llm)

# -----------------------------
# 3) Prompt (chat style)
# -----------------------------
SYSTEM_MSG = (
    "You are a helpful medical RAG assistant. "
    "Use ONLY the provided context to answer the question. "
    "If the answer is not in the context, say you don't know. "
    "Be concise and clinically careful."
)

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_MSG),
    ("human", "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:")
])

# -----------------------------
# 4) Load FAISS vector DB
# -----------------------------
DB_FAISS_PATH = "vectorstore/db_faiss"  # ensure this folder exists & built with same embeddings
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# allow_dangerous_deserialization because FAISS save/load uses pickle metadata
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
retriever = db.as_retriever(search_kwargs={"k": 3})

# -----------------------------
# 5) Build RetrievalQA chain
#    IMPORTANT: prompt is ChatPromptTemplate (works with ChatHuggingFace)
# -----------------------------
qa_chain = RetrievalQA.from_chain_type(
    llm=chat_llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt}
)

# -----------------------------
# 6) Run
# -----------------------------
if __name__ == "__main__":
    user_query = input("Write Query Here: ").strip()
    # RetrievalQA expects {'query': "..."} by default
    resp = qa_chain.invoke({"query": user_query})

    # LangChain normalizes outputs; for chat models we still get "result" text
    answer_text = resp.get("result", "")
    sources = resp.get("source_documents", [])

    print("\n🟢 RESULT:\n", answer_text)
    print("\n📄 SOURCE DOCUMENTS:")
    if not sources:
        print("  (no sources returned)")
    else:
        for i, doc in enumerate(sources, 1):
            src = doc.metadata.get("source") or doc.metadata.get("file_path") or "Unknown"
            print(f"  {i}. {src}")
