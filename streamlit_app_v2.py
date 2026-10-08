import hashlib
import io
import math
import time

import ollama
import streamlit as st

try:
    from pypdf import PdfReader
    HAS_PDF = True
except ImportError:
    HAS_PDF = False

EMBED_MODEL = "nomic-embed-text"
SYSTEM_PROMPT = (
    "You answer questions about a document using ONLY the passages below. "
    "Each passage starts with its location in [brackets]. "
    "If the answer is not in the passages, say you could not find it in the document. "
    "Keep answers short and clear.\n\n"
)

st.set_page_config(page_title="Private Doc Chat", page_icon="🔒", layout="centered")

st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
.stDeployButton {display: none;}
.hero {
    padding: 1.4rem 1.6rem;
    border-radius: 16px;
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 50%, #db2777 100%);
    color: white;
    margin-bottom: 1.2rem;
}
.hero h1 { margin: 0; font-size: 2.1rem; color: white; padding: 0; }
.hero p { margin: 0.4rem 0 0.7rem 0; opacity: 0.92; font-size: 1.02rem; }
.chip {
    display: inline-block;
    padding: 2px 11px;
    margin: 2px 6px 0 0;
    border-radius: 999px;
    background: rgba(255,255,255,0.20);
    font-size: 0.78rem;
}
.cite {
    display: inline-block;
    padding: 1px 10px;
    margin: 0 6px 4px 0;
    border-radius: 999px;
    background: rgba(124,58,237,0.18);
    border: 1px solid rgba(124,58,237,0.45);
    font-size: 0.8rem;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <h1>🔒 Private Doc Chat</h1>
  <p>Ask questions about your documents. Runs 100% on your laptop with tiny open source models.</p>
  <span class="chip">Open source</span><span class="chip">Offline</span>
  <span class="chip">Tiny LLM</span><span class="chip">Ollama</span>
</div>
""", unsafe_allow_html=True)

st.session_state.setdefault("messages", [])


# ---------- helpers ----------
def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


def installed_models():
    try:
        names = [m.model for m in ollama.list().models]
        names = [n for n in names if "embed" not in n]
        return names or ["gemma3:1b"]
    except Exception:
        return ["gemma3:1b"]


def split_text(text, size=800, overlap=150):
    text = " ".join(text.split())
    pieces, start = [], 0
    while start < len(text):
        piece = text[start:start + size].strip()
        if piece:
            pieces.append(piece)
        if start + size >= len(text):
            break
        start += size - overlap
    return pieces


def load_document(uploaded, pasted):
    if uploaded is not None:
        data = uploaded.getvalue()
        if uploaded.name.lower().endswith(".pdf"):
            reader = PdfReader(io.BytesIO(data))
            pages = [(f"Page {i}", p.extract_text() or "") for i, p in enumerate(reader.pages, 1)]
            return uploaded.name, pages
        return uploaded.name, [("Text", data.decode("utf-8", errors="ignore"))]
    if pasted.strip():
        return "Pasted text", [("Text", pasted)]
    return None, []


def make_chunks(sections):
    chunks = []
    for label, text in sections:
        for piece in split_text(text):
            lab = label if label != "Text" else f"Section {len(chunks) + 1}"
            chunks.append({"label": lab, "text": piece})
    return chunks


def embed_all(chunks):
    vecs = []
    bar = st.sidebar.progress(0.0, text="Reading document...")
    for i in range(0, len(chunks), 16):
        batch = [c["text"] for c in chunks[i:i + 16]]
        vecs += ollama.embed(model=EMBED_MODEL, input=batch)["embeddings"]
        bar.progress(min(1.0, (i + 16) / len(chunks)), text="Reading document...")
    bar.empty()
    return vecs


def show_sources(sources):
    labels = []
    for s in sources:
        if s["label"] not in labels:
            labels.append(s["label"])
    st.markdown("".join(f'<span class="cite">📍 {l}</span>' for l in labels), unsafe_allow_html=True)
    with st.expander("View the passages used"):
        for s in sources:
            st.markdown(f"**{s['label']}** · match {s['score']:.0%}")
            st.write(s["text"])
            st.divider()


# ---------- sidebar ----------
with st.sidebar:
    st.header("📄 Document")
    types = ["pdf", "txt", "md"] if HAS_PDF else ["txt", "md"]
    uploaded = st.file_uploader("Upload a file", type=types, label_visibility="collapsed")
    pasted = st.text_area("...or paste text", height=120)

    st.header("⚙️ Settings")
    models = installed_models()
    default = models.index("gemma3:1b") if "gemma3:1b" in models else 0
    model = st.selectbox("Chat model", models, index=default)
    top_k = st.slider("Passages to retrieve", 1, 6, 3)
    if st.button("🗑️ Clear chat"):
        st.session_state.messages = []
        st.rerun()

    with st.expander("How it works"):
        st.markdown(
            "1. Your document is split into small passages.\n"
            "2. **nomic-embed-text** turns each passage into numbers that capture its meaning.\n"
            "3. Your question is matched to the closest passages.\n"
            "4. **gemma3:1b** writes an answer using only those passages.\n\n"
            "Everything runs on this laptop through Ollama. No cloud, no API keys."
        )

# ---------- load and index the document ----------
name, sections = load_document(uploaded, pasted)

if not any(t.strip() for _, t in sections):
    st.info("👈 Upload a PDF or text file in the sidebar, or paste some text, to begin.")
    st.stop()

key = hashlib.md5((name + "".join(t for _, t in sections)).encode("utf-8", errors="ignore")).hexdigest()
if st.session_state.get("doc_key") != key:
    chunks = make_chunks(sections)
    if not chunks:
        st.error("No readable text found. This PDF may be a scan.")
        st.stop()
    try:
        vecs = embed_all(chunks)
    except Exception:
        st.error("Couldn't reach Ollama. Open the Ollama app, then refresh this page.")
        st.stop()
    st.session_state.update(doc_key=key, chunks=chunks, vecs=vecs, messages=[])

chunks = st.session_state["chunks"]
vecs = st.session_state["vecs"]
st.success(f"📄 **{name}** is ready · {len(chunks)} passages indexed")

# ---------- chat history ----------
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m["role"] == "assistant":
            if m.get("sources"):
                show_sources(m["sources"])
            if m.get("stats"):
                st.caption(m["stats"])

# ---------- new question ----------
if prompt := st.chat_input("Ask a question about your document"):
    history = st.session_state.messages[-6:]
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            q_vec = ollama.embed(model=EMBED_MODEL, input=prompt)["embeddings"][0]
            ranked = sorted(zip(chunks, vecs), key=lambda p: cosine(q_vec, p[1]), reverse=True)[:top_k]
            sources = [{"label": c["label"], "text": c["text"], "score": cosine(q_vec, v)} for c, v in ranked]
            context = "\n\n".join(f"[{s['label']}] {s['text']}" for s in sources)
            msgs = [{"role": "system", "content": SYSTEM_PROMPT + context}]
            msgs += [{"role": m["role"], "content": m["content"]} for m in history]
            msgs.append({"role": "user", "content": prompt})

            start = time.time()
            first = {"t": None}

            def stream():
                for part in ollama.chat(model=model, messages=msgs, stream=True):
                    if first["t"] is None:
                        first["t"] = time.time()
                    yield part["message"]["content"]

            answer = st.write_stream(stream())
            total = time.time() - start
            ttf = (first["t"] - start) if first["t"] else total
            stats = f"⚡ {model} · {total:.1f}s total · first word in {ttf:.1f}s · 100% offline"

            show_sources(sources)
            st.caption(stats)
            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "sources": sources, "stats": stats}
            )
        except Exception as e:
            st.error(f"Something went wrong talking to Ollama: {e}")
