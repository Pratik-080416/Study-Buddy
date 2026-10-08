import math
import ollama
import streamlit as st

try:
    from pypdf import PdfReader
    HAS_PDF = True
except ImportError:
    HAS_PDF = False

st.set_page_config(page_title="Private Doc Chat", page_icon="🔒")
st.title("🔒 Private Doc Chat")
st.caption("Runs fully offline with gemma3:1b and nomic-embed-text. Nothing leaves your laptop.")

def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))

def make_chunks(text, size=800):
    chunks = []
    for para in text.split("\n\n"):
        para = para.strip()
        for i in range(0, len(para), size):
            piece = para[i:i + size].strip()
            if piece:
                chunks.append(piece)
    return chunks

types = ["txt", "md"] + (["pdf"] if HAS_PDF else [])
uploaded = st.file_uploader("Upload a document", type=types)
pasted = st.text_area("...or paste text here", height=150)

text = ""
if uploaded is not None:
    if uploaded.name.lower().endswith(".pdf"):
        reader = PdfReader(uploaded)
        text = "\n\n".join((p.extract_text() or "") for p in reader.pages)
    else:
        text = uploaded.read().decode("utf-8", errors="ignore")
elif pasted.strip():
    text = pasted

if text.strip():
    if st.session_state.get("source") != text:
        chunks = make_chunks(text)
        with st.spinner("Reading your document..."):
            vecs = []
            for i in range(0, len(chunks), 32):
                vecs += ollama.embed(model="nomic-embed-text", input=chunks[i:i + 32])["embeddings"]
        st.session_state.update(source=text, chunks=chunks, vecs=vecs)

    chunks = st.session_state["chunks"]
    vecs = st.session_state["vecs"]
    st.success(f"Document ready ({len(chunks)} pieces). Ask away!")

    q = st.text_input("Ask a question about your document")
    if q:
        q_vec = ollama.embed(model="nomic-embed-text", input=q)["embeddings"][0]
        ranked = sorted(zip(chunks, vecs), key=lambda p: cosine(q_vec, p[1]), reverse=True)[:3]
        context = "\n\n".join(c for c, _ in ranked)
        with st.spinner("Thinking..."):
            reply = ollama.chat(
                model="gemma3:1b",
                messages=[
                    {"role": "system", "content": "Answer using only this context. If the answer isn't there, say so.\n\n" + context},
                    {"role": "user", "content": q},
                ],
            )
        st.markdown("### Answer")
        st.write(reply["message"]["content"])
        with st.expander("Sources used"):
            for c, _ in ranked:
                st.write(c)
                st.divider()
else:
    st.info("Upload a file or paste some text to begin.")
