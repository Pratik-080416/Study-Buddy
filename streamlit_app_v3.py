import hashlib
import html
import io
import json
import math
import time

import ollama
import streamlit as st

try:
    from pypdf import PdfReader, PdfWriter
    HAS_PDF = True
except ImportError:
    HAS_PDF = False

EMBED_MODEL = "nomic-embed-text"
MAX_RANGE_PAGES = 15

LEVELS = {
    "Standard": "Explain clearly and concisely.",
    "Simple": "Use simple words and short sentences. Avoid jargon.",
    "Like I'm 5": "Explain as if to a 5-year-old, using a friendly everyday comparison.",
    "Exam-ready": "Give short bullet points a student can memorize, with key terms in bold.",
    "With an example": "Explain briefly, then give one concrete example.",
}

SYSTEM_PROMPT = (
    "You answer questions about a document using ONLY the passages below. "
    "Each passage starts with its location in [brackets]. "
    "If the answer is not in the passages, say you could not find it in the document. "
    "Keep answers short and clear.\n"
)

QUIZ_SCHEMA = {
    "type": "object",
    "properties": {
        "questions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "options": {"type": "array", "items": {"type": "string"}},
                    "answer_index": {"type": "integer"},
                    "explanation": {"type": "string"},
                },
                "required": ["question", "options", "answer_index", "explanation"],
            },
        }
    },
    "required": ["questions"],
}

CARDS_SCHEMA = {
    "type": "object",
    "properties": {
        "cards": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"front": {"type": "string"}, "back": {"type": "string"}},
                "required": ["front", "back"],
            },
        }
    },
    "required": ["cards"],
}

st.set_page_config(page_title="Private Study Buddy", page_icon="🔒", layout="centered")

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
    display: inline-block; padding: 2px 11px; margin: 2px 6px 0 0;
    border-radius: 999px; background: rgba(255,255,255,0.20); font-size: 0.78rem;
}
.cite {
    display: inline-block; padding: 1px 10px; margin: 0 6px 4px 0;
    border-radius: 999px; background: rgba(124,58,237,0.18);
    border: 1px solid rgba(124,58,237,0.45); font-size: 0.8rem;
}
.flashcard {
    padding: 2rem 1.5rem; border-radius: 16px; text-align: center; font-size: 1.25rem;
    min-height: 150px; display: flex; align-items: center; justify-content: center;
    border: 1px solid rgba(124,58,237,0.5); background: rgba(124,58,237,0.12);
    margin: 0.5rem 0 1rem 0;
}
.flashcard.back { background: rgba(219,39,119,0.12); border-color: rgba(219,39,119,0.5); }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <h1>🔒 Private Study Buddy</h1>
  <p>Chat with your textbook, get quizzes, flashcards and summaries, and build new PDFs. 100% on your laptop.</p>
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
    """Returns (name, sections, pdf_bytes). sections = list of (label, text)."""
    if uploaded is not None:
        data = uploaded.getvalue()
        if uploaded.name.lower().endswith(".pdf"):
            reader = PdfReader(io.BytesIO(data))
            pages = [(f"Page {i}", p.extract_text() or "") for i, p in enumerate(reader.pages, 1)]
            return uploaded.name, pages, data
        return uploaded.name, [("Text", data.decode("utf-8", errors="ignore"))], None
    if pasted.strip():
        return "Pasted text", [("Text", pasted)], None
    return None, [], None


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


def stream_reply(model, messages, timing):
    for part in ollama.chat(model=model, messages=messages, stream=True, options={"num_ctx": 8192}):
        if timing["first"] is None:
            timing["first"] = time.time()
        yield part["message"]["content"]


def llm_json(model, prompt, schema, tries=2):
    for _ in range(tries):
        try:
            r = ollama.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                format=schema,
                options={"temperature": 0.3, "num_ctx": 8192},
            )
            return json.loads(r["message"]["content"])
        except Exception:
            continue
    return None


def range_text(start, end, max_chars=6500):
    pages = st.session_state["page_texts"][start - 1:end]
    text = "\n\n".join(" ".join(p.split()) for p in pages if p.strip())
    if len(text) <= max_chars:
        return text, False
    n = 5
    win = max_chars // n
    step = (len(text) - win) // (n - 1)
    return "\n...\n".join(text[i * step:i * step + win] for i in range(n)), True


def study_range():
    total = len(st.session_state["page_texts"])
    if total <= 1:
        return 1, 1
    c1, c2 = st.columns(2)
    start = int(c1.number_input("From page", min_value=1, max_value=total, value=1, step=1, key="r_start"))
    end = int(c2.number_input("To page", min_value=1, max_value=total, value=min(total, 5), step=1, key="r_end"))
    if end < start:
        start, end = end, start
    if end - start + 1 > MAX_RANGE_PAGES:
        end = start + MAX_RANGE_PAGES - 1
        st.caption(f"Limited to {MAX_RANGE_PAGES} pages at a time, so using pages {start}-{end}.")
    return start, end


def clean_quiz(data):
    out = []
    for q in (data or {}).get("questions", []):
        try:
            opts = [str(o) for o in q["options"]]
            idx = int(q["answer_index"])
            question = str(q["question"]).strip()
            if len(opts) >= 2 and 0 <= idx < len(opts) and question:
                out.append({"question": question, "options": opts, "answer_index": idx,
                            "explanation": str(q.get("explanation", ""))})
        except Exception:
            continue
    return out


def clean_cards(data):
    out = []
    for c in (data or {}).get("cards", []):
        try:
            front, back = str(c["front"]).strip(), str(c["back"]).strip()
            if front and back:
                out.append({"front": front, "back": back})
        except Exception:
            continue
    return out


def parse_ranges(spec, total):
    pages = set()
    for part in spec.replace(" ", "").split(","):
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            a, b = int(a), int(b)
            pages.update(range(min(a, b), max(a, b) + 1))
        else:
            pages.add(int(part))
    return sorted(p for p in pages if 1 <= p <= total)


def build_pdf(page_numbers):
    reader = PdfReader(io.BytesIO(st.session_state["pdf_bytes"]))
    writer = PdfWriter()
    for p in page_numbers:
        writer.add_page(reader.pages[p - 1])
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


def pages_for_topic(topic, n):
    q = ollama.embed(model=EMBED_MODEL, input=topic)["embeddings"][0]
    best = {}
    for c, v in zip(st.session_state["chunks"], st.session_state["vecs"]):
        page = int(c["label"].split()[1])
        best[page] = max(best.get(page, 0.0), cosine(q, v))
    top = sorted(sorted(best, key=best.get, reverse=True)[:n])
    return top


def card_move(delta, n):
    st.session_state.card_i = (st.session_state.get("card_i", 0) + delta) % n
    st.session_state.flipped = False


def card_flip():
    st.session_state.flipped = not st.session_state.get("flipped", False)


def build_notes():
    ss = st.session_state
    lines = [f"# Study notes: {ss.get('doc_name', 'document')}", ""]
    if ss.get("summary"):
        lines += ["## Summary", ss["summary"], ""]
    if ss.get("quiz"):
        lines.append("## Quiz")
        for i, q in enumerate(ss["quiz"], 1):
            lines.append(f"**{i}. {q['question']}**")
            for j, o in enumerate(q["options"]):
                lines.append(f"- {chr(65 + j)}. {o}")
            lines.append(f"Answer: {chr(65 + q['answer_index'])}. {q['explanation']}")
            lines.append("")
    if ss.get("cards"):
        lines.append("## Flashcards")
        for c in ss["cards"]:
            lines.append(f"- **{c['front']}** → {c['back']}")
        lines.append("")
    qa = [m for m in ss.get("messages", [])]
    if qa:
        lines.append("## Questions and answers")
        for m in qa:
            who = "Q" if m["role"] == "user" else "A"
            lines.append(f"**{who}:** {m['content']}")
            lines.append("")
    return "\n".join(lines)


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
    level = st.selectbox("Explain it", list(LEVELS.keys()))
    top_k = st.slider("Passages to retrieve", 1, 6, 3)
    if st.button("🗑️ Clear chat"):
        st.session_state.messages = []
        st.rerun()

    with st.expander("How it works"):
        st.markdown(
            "1. Your document is split into small passages.\n"
            "2. **nomic-embed-text** turns each passage into numbers that capture its meaning.\n"
            "3. Your question is matched to the closest passages.\n"
            "4. A tiny local model writes the answer, quiz, or summary from those passages.\n\n"
            "Everything runs on this laptop through Ollama. No cloud, no API keys."
        )

# ---------- load and index the document ----------
name, sections, pdf_bytes = load_document(uploaded, pasted)

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
    for k in ["r_start", "r_end"] + [f"quiz_{i}" for i in range(12)]:
        st.session_state.pop(k, None)
    st.session_state.update(
        doc_key=key, doc_name=name, chunks=chunks, vecs=vecs, messages=[],
        page_texts=[t for _, t in sections], pdf_bytes=pdf_bytes,
        summary=None, quiz=None, quiz_checked=False, cards=None,
        card_i=0, flipped=False, topic_pages=None,
    )

chunks = st.session_state["chunks"]
vecs = st.session_state["vecs"]
total_pages = len(st.session_state["page_texts"])
is_pdf = st.session_state["pdf_bytes"] is not None
st.success(f"📄 **{name}** is ready · {len(chunks)} passages indexed")

modes = ["💬 Chat", "📝 Summary", "🧠 Quiz", "🃏 Flashcards"] + (["✂️ PDF tools"] if is_pdf else [])
mode = st.radio("Mode", modes, horizontal=True, label_visibility="collapsed")

# ---------- CHAT ----------
if mode == "💬 Chat":
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m["role"] == "assistant":
                if m.get("sources"):
                    show_sources(m["sources"])
                if m.get("stats"):
                    st.caption(m["stats"])

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
                msgs = [{"role": "system", "content": SYSTEM_PROMPT + "Style: " + LEVELS[level] + "\n\n" + context}]
                msgs += [{"role": m["role"], "content": m["content"]} for m in history]
                msgs.append({"role": "user", "content": prompt})

                timing = {"first": None}
                start = time.time()
                answer = st.write_stream(stream_reply(model, msgs, timing))
                total = time.time() - start
                ttf = (timing["first"] - start) if timing["first"] else total
                stats = f"⚡ {model} · {total:.1f}s total · first word in {ttf:.1f}s · 100% offline"

                show_sources(sources)
                st.caption(stats)
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer, "sources": sources, "stats": stats}
                )
            except Exception as e:
                st.error(f"Something went wrong talking to Ollama: {e}")

# ---------- SUMMARY ----------
elif mode == "📝 Summary":
    st.subheader("📝 Summarize a chapter")
    start, end = study_range()
    if st.button("Summarize", type="primary"):
        text, sampled = range_text(start, end)
        prompt = (
            "Summarize the study material below as key points a college student can review before an exam. "
            + LEVELS[level]
            + " Use bullet points and put key terms in bold. Use only the material.\n\nMATERIAL:\n" + text
        )
        timing = {"first": None}
        t0 = time.time()
        try:
            summary = st.write_stream(stream_reply(model, [{"role": "user", "content": prompt}], timing))
            st.session_state.summary = summary
            st.caption(f"⚡ {model} · {time.time() - t0:.1f}s · 100% offline")
            if sampled:
                st.caption("This range is long, so the summary covers sampled parts. Use a smaller range for more detail.")
        except Exception as e:
            st.error(f"Something went wrong talking to Ollama: {e}")
    elif st.session_state.get("summary"):
        st.markdown(st.session_state["summary"])

# ---------- QUIZ ----------
elif mode == "🧠 Quiz":
    st.subheader("🧠 Quiz me")
    start, end = study_range()
    n_q = st.slider("Number of questions", 3, 8, 5)
    if st.button("Generate quiz", type="primary"):
        text, _ = range_text(start, end)
        prompt = (
            f"Write {n_q} multiple-choice questions that test understanding of the study material below. "
            "Each question has exactly 4 options and exactly one correct answer. "
            "answer_index is the position of the correct option (0 to 3). "
            "Add a one-sentence explanation. Use only the material.\n\nMATERIAL:\n" + text
        )
        with st.spinner("Writing your quiz..."):
            quiz = clean_quiz(llm_json(model, prompt, QUIZ_SCHEMA))
        if quiz:
            for i in range(12):
                st.session_state.pop(f"quiz_{i}", None)
            st.session_state.quiz = quiz
            st.session_state.quiz_checked = False
        else:
            st.error("The model didn't return a usable quiz. Try again, a smaller page range, or a bigger model.")

    quiz = st.session_state.get("quiz")
    if quiz:
        if st.session_state.get("quiz_checked"):
            score = sum(
                1 for i, q in enumerate(quiz)
                if st.session_state.get(f"quiz_{i}") == q["options"][q["answer_index"]]
            )
            st.metric("Your score", f"{score} / {len(quiz)}")
        for i, q in enumerate(quiz):
            st.markdown(f"**Q{i + 1}. {q['question']}**")
            st.radio("Choose one", q["options"], index=None, key=f"quiz_{i}", label_visibility="collapsed")
            if st.session_state.get("quiz_checked"):
                correct = q["options"][q["answer_index"]]
                if st.session_state.get(f"quiz_{i}") == correct:
                    st.success("Correct! " + q["explanation"])
                else:
                    st.error(f"Correct answer: {correct}. " + q["explanation"])
        if st.button("✅ Check answers"):
            st.session_state.quiz_checked = True
            st.rerun()
        st.caption("Small models can make mistakes. Check the page in your book if an answer looks wrong.")

# ---------- FLASHCARDS ----------
elif mode == "🃏 Flashcards":
    st.subheader("🃏 Flashcards")
    start, end = study_range()
    n_c = st.slider("Number of cards", 4, 12, 8)
    if st.button("Make flashcards", type="primary"):
        text, _ = range_text(start, end)
        prompt = (
            f"Make {n_c} flashcards from the study material below. "
            "Each card has a short front (a term or question) and a short back (the answer or definition). "
            "Use only the material.\n\nMATERIAL:\n" + text
        )
        with st.spinner("Making your cards..."):
            cards = clean_cards(llm_json(model, prompt, CARDS_SCHEMA))
        if cards:
            st.session_state.cards = cards
            st.session_state.card_i = 0
            st.session_state.flipped = False
        else:
            st.error("The model didn't return usable cards. Try again or use a smaller page range.")

    cards = st.session_state.get("cards")
    if cards:
        i = st.session_state.get("card_i", 0) % len(cards)
        flipped = st.session_state.get("flipped", False)
        side = "back" if flipped else "front"
        st.caption(f"Card {i + 1} of {len(cards)} · showing the {side}")
        st.markdown(f'<div class="flashcard {side}">{html.escape(cards[i][side])}</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        c1.button("⬅️ Previous", on_click=card_move, args=(-1, len(cards)), use_container_width=True)
        c2.button("🔄 Flip", on_click=card_flip, use_container_width=True)
        c3.button("Next ➡️", on_click=card_move, args=(1, len(cards)), use_container_width=True)

# ---------- PDF TOOLS ----------
elif mode == "✂️ PDF tools":
    st.subheader("✂️ Build a new PDF")
    st.caption(f"Your original has {total_pages} pages and is never changed. Downloads are new files.")
    method = st.radio("How do you want to choose pages?", ["By page numbers", "By topic (AI-picked)"], horizontal=True)
    base = name.rsplit(".", 1)[0]

    if method == "By page numbers":
        action = st.radio("Action", ["Keep only these pages", "Remove these pages"], horizontal=True)
        spec = st.text_input("Pages (example: 1-3, 7, 20-35)")
        if spec:
            try:
                nums = parse_ranges(spec, total_pages)
            except ValueError:
                st.error("Use numbers like 1-3, 7, 20-35.")
                nums = []
            if nums:
                keep = nums if action.startswith("Keep") else [p for p in range(1, total_pages + 1) if p not in nums]
                if keep:
                    st.success(f"The new PDF will have {len(keep)} pages.")
                    st.download_button(
                        "📥 Download new PDF", build_pdf(keep),
                        file_name=f"{base}_edited.pdf", mime="application/pdf",
                    )
                else:
                    st.warning("That would remove every page.")
    else:
        topic = st.text_input("Topic (example: entanglement, Bayes theorem)")
        n_pages = st.slider("How many pages", 1, 15, 5)
        if topic and st.button("Find pages", type="primary"):
            with st.spinner("Searching by meaning..."):
                st.session_state.topic_pages = (topic, pages_for_topic(topic, n_pages))
        tp = st.session_state.get("topic_pages")
        if tp:
            st.success(f"Most relevant pages for “{tp[0]}”: " + ", ".join(map(str, tp[1])))
            st.download_button(
                "📥 Download these pages as a PDF", build_pdf(tp[1]),
                file_name=f"{base}_{tp[0].replace(' ', '_')}.pdf", mime="application/pdf",
            )

# ---------- study notes download ----------
with st.sidebar:
    st.header("📥 Export")
    st.download_button(
        "Download study notes", build_notes(),
        file_name="study_notes.md", mime="text/markdown",
    )
