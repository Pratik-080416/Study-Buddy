# 📚 Study Buddy

An **offline learning companion for students**, powered by tiny open-source
LLMs. Upload your notes or textbook and learn from them with chat, summaries,
quizzes and flashcards. No cloud, no API keys, no data leaving your laptop.

> Screenshot or GIF goes here

## Why tiny models?
Many students don't have reliable internet or can't pay for AI subscriptions.
Study Buddy runs a 1B-parameter model on an ordinary laptop, so anyone can use it.

## Features
- 💬 Chat with your document, with page citations
- 📝 Chapter summaries at different levels (simple, exam-ready, like I'm 5)
- 🧠 Auto-generated quizzes
- 🃏 3D flip flashcards with auto-play
- ✂️ Build a custom PDF by page numbers or by topic

## Roadmap
- [ ] Model lab: compare tiny models side by side
- [ ] Study progress tracking
- [ ] More file types (.docx)
- [ ] Your ideas! See CONTRIBUTING.md

## Quick start
1. Install [Ollama](https://ollama.com), then run:
```
   ollama pull gemma3:1b
   ollama pull nomic-embed-text
```
2. Install and run:
```
   pip install -r requirements.txt
   streamlit run streamlit_app.py
```

## Built with
Streamlit, Ollama, gemma3:1b, nomic-embed-text, pypdf

## Contributing
See [CONTRIBUTING.md](CONTRIBUTING.md). Issues labeled `good first issue` are a great start.

## License
MIT
