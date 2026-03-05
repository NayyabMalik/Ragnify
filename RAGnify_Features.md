# ⚡ RAGnify — Complete Feature Reference
**Retrieval-Augmented Intelligence · By Nayyab Malik**

---

## 📋 Project Overview

| Field | Details |
|-------|---------|
| Project Name | RAGnify — Retrieval-Augmented Intelligence |
| Developer | Nayyab Malik (AI Engineer) |
| LLM Provider | OpenRouter → GPT-4o-mini |
| Embeddings | OpenAI text-embedding-3-small (via OpenRouter) |
| Framework | LangGraph (StateGraph) + LangChain + Streamlit |
| Memory Store | SQLite (persistent across sessions) |
| Vector Store | FAISS (in-memory, multi-document) |
| Web Search | Tavily API (direct HTTP, no wrappers) |
| Interface | Streamlit — wide layout, dark theme, custom CSS |

---

## 🤖 Core AI Features

### 🔍 Retrieval-Augmented Generation (RAG)
Upload documents and ask questions grounded in their content. Automatically switches between **Chat Mode** and **RAG Mode** based on whether documents are loaded.

### 🌐 Live Web Search (Tavily)
Direct `requests.post()` call to Tavily fetches real-time search results and injects them into the system prompt. Triggered automatically for current events, news, conflicts, prices, and any time-sensitive queries. No LangChain wrappers — results are guaranteed to reach the LLM.

### 🧠 Persistent Memory (SQLite)
All conversation history stored in `ragnify_memory.db` using LangGraph's `SqliteSaver`. Threads persist across restarts. Auto-summarization kicks in after 20 messages to keep context manageable.

### 🌍 Multi-Language Support
Detects the language of every user message (ISO 639-1 code) and responds in the same language. Covers English, Urdu, Arabic, French, Chinese, and more.

### 📡 Streaming Responses
Answers stream token-by-token using LangGraph `stream_mode="messages"`. An animated spinning web-search badge appears while Tavily is fetching results.

### ⚡ Async Parallel Retrieval
RAG queries are expanded into 3 variants and retrieved in parallel using `asyncio.gather()`. Results are deduplicated by content hash before being passed to the generator node.

### 📊 Embedding Confidence Score
Cosine similarity between the query embedding and the top retrieved chunk gives a 0–100% confidence score, displayed as a live colour-coded progress bar (green/amber/red).

---

## 📄 Document Features

### 📂 Multi-Document Upload
Upload and index multiple documents simultaneously. Each file is chunked (600 chars, 80 overlap) and added to a shared FAISS vector store. Remove individual documents without affecting others.

### 📝 10 Supported File Types
`PDF`, `TXT`, `DOCX`, `CSV`, `JSON`, `MD`, `HTML`, `HTM`, `XML`, `EPUB`

Smart parsers extract clean text from each format using PyPDF2, python-docx, BeautifulSoup, and ebooklib.

### 🔬 AI Document Summary
On upload, GPT-4o-mini analyses the first 2,500 characters and returns structured JSON:
- Document title
- Detected language
- Document type (Resume, Research Paper, Contract, etc.)
- 4–6 key topics
- 2-sentence summary

All shown instantly in the sidebar.

### 🧩 Quiz Generator
Generate 3–10 multiple-choice questions from any uploaded document. Each question includes:
- 4 answer options (A/B/C/D)
- The correct answer letter
- A one-sentence explanation

Scored live in the UI with feedback on submission.

---

## 🎨 UI & UX Features

### 💬 Session Management
Create unlimited named chat sessions. Each session gets an **AI-generated title** (5 words max) based on the first message. Switch between sessions from the sidebar — history loads instantly from SQLite.

### 💡 Follow-up Suggestions
After every response, 3 contextual follow-up questions are generated and shown as clickable buttons below the answer. Clicking one auto-submits it as the next message.

### 📋 Copy to Clipboard
Every assistant response has a **📋 Copy** button. Uses `navigator.clipboard` API with fallback to `execCommand` for older browsers. A toast notification confirms the copy.

### ⭐ Response Rating
👍 / 👎 rating buttons appear **directly under the last assistant message** (not floating above). Ratings are logged with timestamp, confidence score, mode, and question preview.

### 📤 Export Chat (Markdown)
Download any conversation as a `.md` file with the session title, all messages, and a timestamp. Available from the Sessions panel in the sidebar.

### 🎨 Custom Dark Theme
Full custom CSS with:
- **Syne** (headings) + **DM Mono** (body) fonts
- Teal `#00C9A7` and Amber `#F5A623` accent palette
- Animated spinning web-search badge
- Styled file uploader with dashed teal border
- Confidence progress bar with dynamic colour
- Source citation tags
- Custom scrollbar styling

---

## 📊 Analytics & Evaluation

### 📈 Session Analytics Dashboard
Live metrics panel with:
- Total turns, RAG turns, Chat turns
- Average confidence score
- Satisfaction rate (% positive ratings)

**Plotly charts:**
- Confidence score over turns (line chart)
- RAG vs Chat mode split (pie chart)
- Source document usage frequency (bar chart)
- Confidence distribution — High / Medium / Low (column chart)

### 🧪 Evaluation Log
Every turn is automatically logged with:
- Question preview
- 👍/👎 rating
- Confidence percentage
- Mode (RAG or Chat)
- Timestamp

Full log displayed in a table. Export as **CSV**. Cumulative satisfaction trend line with 80% target marker.

### 🗑 Reset Controls
Separate reset buttons for Analytics and Evaluation log — clears session metrics without affecting conversation history.

---

## 🔗 LangGraph Architecture

```
START
  └─► router_node       — Detects Chat/RAG mode, detects language,
                          auto-summarises history if > 20 messages
        ├─► chat_node   — (no documents loaded)
        │     └─► Checks _SEARCH_TRIGGERS
        │           ├─► Calls tavily_search() directly (Python)
        │           │     └─► Injects results into system prompt
        │           │           └─► LLM answers from results
        │           └─► Normal chat (stable facts / math / code)
        │
        └─► retrieve_node  — (documents loaded)
              ├─► generate_node  — High-confidence RAG context found
              └─► chat_node      — No relevant chunks → fallback
```

**5 Graph Nodes:**

| Node | Responsibility |
|------|---------------|
| `router_node` | Mode detection, language detection, history summarisation |
| `chat_node` | Web search trigger detection, Tavily injection, LLM response |
| `tools` | Stub ToolNode (kept for graph compatibility) |
| `retrieve_node` | Parallel FAISS retrieval, confidence scoring, source extraction |
| `generate_node` | RAG answer generation grounded in retrieved chunks |

---

## 🛠 Technical Stack

| Component | Technology |
|-----------|-----------|
| LangGraph | `StateGraph` — 5 nodes, conditional routing, SQLite checkpointer |
| LangChain | `ChatOpenAI`, `OpenAIEmbeddings`, `RecursiveCharacterTextSplitter`, `FAISS` |
| Streamlit | Frontend UI, session state, tabs, file uploader, `st.chat_input` |
| Tavily | Direct `requests.post()` to `api.tavily.com/search` — no deprecated wrappers |
| SQLite | LangGraph `SqliteSaver` for persistent cross-session memory |
| FAISS | In-memory vector store, similarity search, multi-doc merging |
| Plotly | Interactive charts in Analytics tab |
| OpenRouter | Routes all LLM and embedding calls to OpenAI models |
| python-dotenv | Loads `OPENAI_API_KEY`, `TAVILY_API_KEY` from `.env` |

---

## ⚙️ Setup & Dependencies

### Install packages
```bash
pip install langchain langchain-community langchain-openai langchain-text-splitters
pip install langgraph faiss-cpu streamlit plotly python-dotenv requests
pip install PyPDF2 python-docx beautifulsoup4 ebooklib lxml numpy
```

### .env file
```
OPENAI_API_KEY=sk-...          # Your OpenRouter key
TAVILY_API_KEY=tvly-...        # Free at app.tavily.com (1000 searches/month)
```

### Run
```bash
streamlit run Chatbot\chatbot_complete_frontend.py
```

### First-time setup note
Delete `ragnify_memory.db` if upgrading from an older version to clear stale checkpoints.

---

*RAGnify — Built by Nayyab Malik · All features listed above are implemented and functional*
