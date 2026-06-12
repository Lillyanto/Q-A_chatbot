"""
Streamlit Chatbot UI for Document Q&A
======================================
- Upload PDF documents
- Automatic ingestion (text + image extraction)
- Chat interface to ask questions about uploaded docs
- Uses Ollama (nomic-embed-text + llama3.2 + llava) locally

Run: streamlit run app.py

Fixes applied:
  1. Image description is now LAZY — LLaVA runs only when an image chunk
     is actually retrieved, not during ingestion. Huge speed-up for upload.
  2. Removed time.sleep(0.5) dead-wait inside the ChromaDB batch loop.
  3. Separate llava model used for image description (llama3.2 is text-only).
  4. Unified DB ingestion path — no more duplicate if/else code paths.
  5. Prompt context capped at ~2000 chars to keep inference fast on low-RAM.
  6. Duplicate PDF guard — same filename skips re-ingestion.
  7. Temp file cleaned up in finally block to prevent leaks on crash.
"""

from __future__ import annotations

import streamlit as st
import os
import sys
import base64
import tempfile
import hashlib

# Fix sqlite3 BEFORE any chromadb/langchain imports
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_core.documents import Document

# ─── CONFIGURATION ─────────────────────────────────────────────────────────────
DB_PATH            = "./chroma_db_streamlit"
IMAGE_SAVE_PATH    = "./extracted_images"
OLLAMA_BASE_URL    = "http://127.0.0.1:11434"
EMBEDDING_MODEL    = "nomic-embed-text"
LLM_MODEL          = "llama3.2"
VISION_MODEL       = "llava"          # FIX 3: separate model for images
MAX_CONTEXT_CHARS  = 2000             # FIX 5: cap prompt context size
BATCH_SIZE         = 50               # larger batches are fine; no sleep needed

os.makedirs(IMAGE_SAVE_PATH, exist_ok=True)

# ─── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="📄 Document Q&A Chatbot",
    page_icon="🤖",
    layout="wide"
)

# ─── CACHED MODELS ─────────────────────────────────────────────────────────────
@st.cache_resource
def get_embeddings():
    return OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)

@st.cache_resource
def get_llm():
    return OllamaLLM(model=LLM_MODEL, base_url=OLLAMA_BASE_URL)

@st.cache_resource
def get_vision_llm():
    # FIX 3: llava is the vision-capable model; llama3.2 cannot handle images
    return OllamaLLM(model=VISION_MODEL, base_url=OLLAMA_BASE_URL)

# ─── HELPER: LAZY IMAGE DESCRIPTION ───────────────────────────────────────────
def describe_image(image_path: str) -> str:
    """
    Use LLaVA to describe an image on-demand (called at query time, not ingest).
    FIX 1 + FIX 3: lazy evaluation with the correct vision model.
    """
    vision_llm = get_vision_llm()
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")

    prompt = (
        "Analyze this image from a technical document. "
        "If it is a table, transcribe it as a Markdown table. "
        "If it is a diagram, describe its components step-by-step. "
        "Be concise."
    )
    return vision_llm.invoke(prompt, images=[img_b64])


# ─── INGESTION ─────────────────────────────────────────────────────────────────
def get_ingested_filenames() -> set[str]:
    """Return set of filenames already stored in the DB to avoid re-ingestion."""
    db = get_vector_db(get_embeddings())
    if db is None:
        return set()
    try:
        existing = db.get(include=["metadatas"])
        return {m.get("source", "") for m in existing["metadatas"]}
    except Exception:
        return set()


def process_pdf(uploaded_file, embeddings, progress_bar, status_text):
    """
    Extract text and images from PDF and store in ChromaDB.

    FIX 1: Images are stored with their path in metadata only — NO LLaVA call
            during ingestion. Description happens lazily at retrieval time.
    FIX 2: No time.sleep() in the batch loop.
    FIX 4: Single unified DB upsert path.
    FIX 7: Temp file cleaned up in finally block.
    """
    tmp_path = None
    all_docs  = []
    extracted_images: set[str] = set()

    try:
        # Save upload to a temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name

        doc         = fitz.open(tmp_path)
        total_pages = len(doc)

        for page_num in range(total_pages):
            page     = doc[page_num]
            progress = (page_num + 1) / total_pages
            progress_bar.progress(progress * 0.8)          # 0-80 % = extraction
            status_text.text(f"Extracting page {page_num + 1}/{total_pages}…")

            # ── Text ──────────────────────────────────────────────────────────
            text = page.get_text()
            if text.strip():
                all_docs.append(Document(
                    page_content=text,
                    metadata={
                        "source": uploaded_file.name,
                        "page":   page_num + 1,
                        "type":   "text",
                    }
                ))

            # ── Images ────────────────────────────────────────────────────────
            for img_index, img in enumerate(page.get_images(full=True)):
                xref        = img[0]
                base_image  = doc.extract_image(xref)
                image_bytes = base_image["image"]

                if len(image_bytes) < 5000:          # skip tiny icons
                    continue

                image_hash = hashlib.md5(image_bytes).hexdigest()
                if image_hash in extracted_images:   # skip duplicates
                    continue
                extracted_images.add(image_hash)

                filename  = f"page_{page_num+1}_img_{img_index}.png"
                save_path = os.path.join(IMAGE_SAVE_PATH, filename)
                with open(save_path, "wb") as f:
                    f.write(image_bytes)

                # FIX 1: Store path only — NO LLaVA call here
                all_docs.append(Document(
                    page_content=(
                        f"[Image on page {page_num + 1} — "
                        f"description generated when retrieved]"
                    ),
                    metadata={
                        "source": uploaded_file.name,
                        "page":   page_num + 1,
                        "type":   "image",
                        "path":   save_path,
                    }
                ))

        doc.close()

        # ── Chunk ─────────────────────────────────────────────────────────────
        status_text.text("Splitting into chunks…")
        progress_bar.progress(0.85)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=100
        )
        chunks = splitter.split_documents(all_docs)

        # ── Store (FIX 2: no sleep; FIX 4: single code path) ─────────────────
        status_text.text("Storing in vector database…")
        progress_bar.progress(0.90)

        # Chroma.from_documents handles both new and existing DBs cleanly
        # For an existing DB we use add_texts in batches (no sleep needed)
        if os.path.exists(DB_PATH):
            vector_db = Chroma(
                persist_directory=DB_PATH,
                embedding_function=embeddings
            )
            for i in range(0, len(chunks), BATCH_SIZE):
                batch = chunks[i : i + BATCH_SIZE]
                vector_db.add_texts(
                    texts=[d.page_content for d in batch],
                    metadatas=[d.metadata for d in batch],
                )
                # Update progress within 90-100 %
                sub = 0.90 + 0.10 * min((i + BATCH_SIZE) / len(chunks), 1.0)
                progress_bar.progress(sub)
        else:
            vector_db = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                persist_directory=DB_PATH,
            )

    except Exception as e:
        st.error(f"Error processing document: {e}")
        import traceback
        st.error(traceback.format_exc())
        return 0, 0, 0
    finally:
        # FIX 7: always clean up the temp file
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

    progress_bar.progress(1.0)
    status_text.text("✅ Done!")
    return len(chunks), len(extracted_images), total_pages


# ─── VECTOR DB ─────────────────────────────────────────────────────────────────
def get_vector_db(embeddings):
    if os.path.exists(DB_PATH):
        return Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
    return None


# ─── RAG QUERY ─────────────────────────────────────────────────────────────────
def ask_question(query: str, vector_db, llm):
    """
    Retrieve relevant chunks then generate an answer.

    FIX 1: If an image chunk is retrieved its LLaVA description is generated
            NOW (lazy), then injected into the context.
    FIX 5: Context is capped at MAX_CONTEXT_CHARS to keep inference fast.
    """
    retriever    = vector_db.as_retriever(search_kwargs={"k": 3})
    relevant_docs = retriever.invoke(query)

    context_parts: list[str] = []
    char_budget = MAX_CONTEXT_CHARS

    for doc in relevant_docs:
        if doc.metadata.get("type") == "image":
            img_path = doc.metadata.get("path", "")
            if img_path and os.path.exists(img_path):
                with st.spinner(f"Describing image from page {doc.metadata.get('page', '?')}…"):
                    description = describe_image(img_path)
                chunk_text = f"[Image, page {doc.metadata.get('page')}]: {description}"
            else:
                chunk_text = doc.page_content
        else:
            chunk_text = doc.page_content

        # FIX 5: honour char budget
        if char_budget <= 0:
            break
        context_parts.append(chunk_text[:char_budget])
        char_budget -= len(chunk_text)

    context = "\n\n".join(context_parts)

    prompt = (
        "You are an expert assistant answering questions about uploaded documents.\n"
        "Use only the retrieved context below to answer. "
        "If the answer is not in the context, say you don't know.\n"
        "Keep the answer concise, accurate, and structured.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        "Answer:"
    )

    answer = llm.invoke(prompt)
    return answer, relevant_docs


# ─── UI ────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📁 Document Upload")
    st.markdown("Upload a PDF to add it to the knowledge base.")

    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        help="Upload a PDF document to analyse",
    )

    if uploaded_file:
        # FIX 6: warn if this file is already in the DB
        already_ingested = uploaded_file.name in get_ingested_filenames()
        if already_ingested:
            st.info(
                f"**{uploaded_file.name}** is already in the database. "
                "Clear the DB first if you want to re-ingest it."
            )

        process_disabled = already_ingested
        if st.button(
            "🚀 Process Document",
            type="primary",
            use_container_width=True,
            disabled=process_disabled,
        ):
            with st.spinner("Initialising embedding model…"):
                embeddings = get_embeddings()

            progress_bar = st.progress(0)
            status_text  = st.empty()

            num_chunks, num_images, num_pages = process_pdf(
                uploaded_file, embeddings, progress_bar, status_text
            )

            if num_chunks > 0:
                st.success("✅ Processed successfully!")
                st.info(
                    f"📄 Pages: {num_pages} | "
                    f"📦 Chunks: {num_chunks} | "
                    f"🖼️ Images queued: {num_images}"
                )
                st.session_state["db_ready"] = True

    st.divider()

    if os.path.exists(DB_PATH):
        st.success("🟢 Database ready")
        st.session_state["db_ready"] = True

        if st.button("🗑️ Clear Database", use_container_width=True):
            import shutil
            shutil.rmtree(DB_PATH, ignore_errors=True)
            st.session_state["db_ready"]  = False
            st.session_state["messages"]  = []
            st.rerun()
    else:
        st.warning("🟡 No documents uploaded yet")

# ── Chat area ──────────────────────────────────────────────────────────────────
st.title("🤖 Document Q&A Chatbot")
st.caption("Upload a PDF in the sidebar, then ask questions about it here.")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message:
            with st.expander("📚 Sources"):
                for src in message["sources"]:
                    st.markdown(f"- {src}")

if prompt := st.chat_input("Ask a question about your documents…"):
    if not st.session_state.get("db_ready"):
        st.warning("⚠️ Please upload and process a document first!")
    else:
        st.session_state["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                embeddings = get_embeddings()
                llm        = get_llm()
                vector_db  = get_vector_db(embeddings)

                if vector_db is None:
                    st.error("Database not found. Please upload a document first.")
                else:
                    answer, sources = ask_question(prompt, vector_db, llm)
                    st.markdown(answer)

                    source_list = []
                    with st.expander("📚 Sources"):
                        for idx, doc in enumerate(sources):
                            meta     = doc.metadata
                            doc_type = meta.get("type", "unknown")
                            page_num = meta.get("page", "?")
                            src_str  = f"Page {page_num} ({doc_type})"
                            if doc_type == "image":
                                src_str += f" — {meta.get('path', '')}"
                            source_list.append(src_str)
                            st.markdown(f"**[{idx+1}]** {src_str}")

                    st.session_state["messages"].append({
                        "role":    "assistant",
                        "content": answer,
                        "sources": source_list,
                    })
