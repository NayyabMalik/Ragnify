"""
RAGnify — Backend
By Nayyab Malik
Features: multi-doc RAG, async parallel retrieval, persistent memory,
          multi-language, quiz generation, doc summary, confidence scoring,
          streaming, follow-up suggestions, conversation titling.
"""

from __future__ import annotations
import asyncio, csv, io, json, os, sqlite3
from typing import Annotated, Literal, TypedDict

from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import (
    AIMessage, BaseMessage, HumanMessage, SystemMessage,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

load_dotenv()

# ── LLMs ──────────────────────────────────────────────────────────────────────
def _llm(temp=0.7):
    return ChatOpenAI(
        model="openai/gpt-4o-mini",
        base_url="https://openrouter.ai/api/v1",
        temperature=temp,
        max_retries=3,
        streaming=True,
    )

chat_llm      = _llm(0.7)
precise_llm   = _llm(0.2)   # summaries, titles, quizzes
lang_llm      = _llm(0.5)   # language detection + translation routing

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    base_url="https://openrouter.ai/api/v1",
)

# ── Web Search — Tavily direct API (no LangChain wrappers) ───────────────────
# Get free key at https://app.tavily.com → add to .env: TAVILY_API_KEY=tvly-xxx
import requests as _requests

def tavily_search(query: str) -> str:
    """Fetch live search results and return as plain text for prompt injection."""
    api_key = os.getenv("TAVILY_API_KEY", "")
    if not api_key:
        return ""
    try:
        resp = _requests.post(
            "https://api.tavily.com/search",
            json={
                "api_key":             api_key,
                "query":               query,
                "search_depth":        "advanced",
                "include_answer":      True,
                "include_raw_content": False,
                "max_results":         6,
            },
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        parts = []
        if data.get("answer"):
            parts.append(f"SUMMARY: {data['answer']}")
        for r in data.get("results", []):
            title = r.get("title", "")
            body  = r.get("content", "")
            url   = r.get("url", "")
            date  = r.get("published_date", "")
            date_str = f" ({date})" if date else ""
            parts.append(f"- {title}{date_str}\n  {body}\n  Source: {url}")
        return "\n\n".join(parts)
    except Exception as e:
        print(f"[Tavily error] {e}")
        return ""

_has_tavily = bool(os.getenv("TAVILY_API_KEY", ""))
print(f"✅ Web search: {'Tavily active' if _has_tavily else 'WARNING — add TAVILY_API_KEY to .env'}")

# Minimal stub so ToolNode import doesn't break
from langchain_core.tools import tool as _tool
@_tool
def search_tool(query: str) -> str:
    """Stub — search handled directly in chat_node."""
    return ""
tools = [search_tool]
llm_with_tools = chat_llm  # search handled directly, no tool binding needed

# ── Supported file types ───────────────────────────────────────────────────────
SUPPORTED_EXTENSIONS = [
    "pdf", "txt", "docx", "csv", "json",
    "md", "html", "htm", "xml", "epub",
]

# ── File parsers ───────────────────────────────────────────────────────────────
def parse_file(file_bytes: bytes, filename: str, mime_type: str = "") -> str:
    ext = filename.rsplit(".", 1)[-1].lower()

    if ext == "pdf":
        import PyPDF2
        reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
        return "\n".join(p.extract_text() or "" for p in reader.pages)

    if ext == "docx":
        import docx as _docx
        doc = _docx.Document(io.BytesIO(file_bytes))
        return "\n".join(p.text for p in doc.paragraphs)

    if ext == "csv":
        text = file_bytes.decode("utf-8", errors="ignore")
        rows = list(csv.reader(io.StringIO(text)))
        return "\n".join(", ".join(r) for r in rows)

    if ext == "json":
        text = file_bytes.decode("utf-8", errors="ignore")
        try:
            return json.dumps(json.loads(text), indent=2)
        except Exception:
            return text

    if ext in ("html", "htm"):
        from bs4 import BeautifulSoup
        return BeautifulSoup(file_bytes, "html.parser").get_text("\n")

    if ext == "xml":
        from bs4 import BeautifulSoup
        return BeautifulSoup(file_bytes, "xml").get_text("\n")

    if ext == "epub":
        import ebooklib
        from ebooklib import epub
        from bs4 import BeautifulSoup
        book = epub.read_epub(io.BytesIO(file_bytes))
        parts = []
        for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
            parts.append(BeautifulSoup(item.get_content(), "html.parser").get_text("\n"))
        return "\n".join(parts)

    return file_bytes.decode("utf-8", errors="ignore")


# ── Vector store ───────────────────────────────────────────────────────────────
vectorstore:  FAISS | None = None
retriever                  = None
indexed_docs: dict[str, int] = {}   # filename → chunk count
TOP_K = 5

def build_vectorstore(file_bytes: bytes, filename: str, mime_type: str = "") -> int:
    global vectorstore, retriever, indexed_docs
    raw   = parse_file(file_bytes, filename, mime_type)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600, chunk_overlap=80,
        separators=["\n\n", "\n", ".", " "],
    )
    chunks = splitter.create_documents(
        [raw],
        metadatas=[{"source": filename, "chunk": i} for i in range(99999)],
    )
    for i, c in enumerate(chunks):
        c.metadata["chunk"] = i

    if vectorstore is None:
        vectorstore = FAISS.from_documents(chunks, embeddings)
    else:
        vectorstore.add_documents(chunks)

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K},
    )
    indexed_docs[filename] = len(chunks)
    return len(chunks)


def rebuild_vectorstore_without(removed: str, file_store: dict):
    global vectorstore, retriever, indexed_docs
    vectorstore  = None
    retriever    = None
    indexed_docs = {}
    for name, meta in file_store.items():
        if name != removed:
            build_vectorstore(meta["bytes"], name, meta["mime"])


def is_rag_mode() -> bool:
    return vectorstore is not None

def get_indexed_docs() -> dict:
    return indexed_docs


# ── Async parallel retrieval ───────────────────────────────────────────────────
async def _async_retrieve(query: str) -> list[Document]:
    """Run retrieval in thread pool so it doesn't block the event loop."""
    loop = asyncio.get_event_loop()
    docs = await loop.run_in_executor(None, retriever.invoke, query)
    return docs

def parallel_retrieve(queries: list[str]) -> list[Document]:
    """Retrieve for multiple query variants and deduplicate by page_content."""
    async def _gather():
        results = await asyncio.gather(*[_async_retrieve(q) for q in queries])
        return results

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, _gather())
                all_docs = future.result()
        else:
            all_docs = loop.run_until_complete(_gather())
    except Exception:
        all_docs = [retriever.invoke(q) for q in queries]

    seen, merged = set(), []
    for batch in all_docs:
        for doc in batch:
            key = doc.page_content[:120]
            if key not in seen:
                seen.add(key)
                merged.append(doc)
    return merged[:TOP_K + 2]


# ── Utility LLM chains ─────────────────────────────────────────────────────────
_SUMMARIZE_CONV = ChatPromptTemplate.from_messages([
    ("system", "Summarize this conversation concisely, preserving key facts:"),
    ("human",  "{history}"),
]) | precise_llm

_TITLE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Generate a short, descriptive title (5 words max) for a conversation "
     "whose first user message is below. Return ONLY the title, no quotes."),
    ("human", "{first_message}"),
]) | precise_llm

_DOC_SUMMARY_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a document analyst. Given the first ~2000 chars of a document, return a JSON object with:\n"
     "- title: string\n"
     "- language: detected language name\n"
     "- topics: list of 4-6 key topics (strings)\n"
     "- summary: 2-sentence summary\n"
     "- doc_type: e.g. Resume, Research Paper, Contract, Report, etc.\n"
     "Return ONLY valid JSON, no markdown."),
    ("human", "{preview}"),
]) | precise_llm

_FOLLOWUP_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Based on this Q&A exchange, suggest exactly 3 short follow-up questions the user might want to ask next.\n"
     "Return ONLY a JSON array of 3 strings. No markdown, no explanation."),
    ("human", "Question: {question}\nAnswer: {answer}"),
]) | precise_llm

_QUIZ_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Create a quiz from the document excerpt below.\n"
     "Return ONLY a JSON array of objects, each with:\n"
     "- question: string\n"
     "- options: list of 4 strings (A/B/C/D)\n"
     "- answer: correct option letter (A/B/C/D)\n"
     "- explanation: one sentence\n"
     "Generate exactly {count} questions. No markdown."),
    ("human", "{content}"),
]) | precise_llm

_LANG_DETECT_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Detect the language of the user message. "
     "Return ONLY the ISO 639-1 two-letter language code (e.g. en, ur, fr, ar, zh). "
     "Nothing else."),
    ("human", "{text}"),
]) | precise_llm

_RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are RAGnify, a precise document assistant created by Nayyab Malik.\n"
     "Answer using ONLY the context below. Cite sources as [Source: filename, chunk N].\n"
     "If the answer is not in the context, say so and answer from general knowledge.\n"
     "{lang_instruction}\n\n"
     "Context:\n{context}"),
    ("placeholder", "{messages}"),
]) | chat_llm

_CHAT_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are RAGnify, a smart AI assistant created by Nayyab Malik.\n\n"
     "You have access to a web search tool. Follow these rules strictly:\n"
     "- ALWAYS call search_tool for: current events, news, politics, conflicts, "
     "ongoing situations, prices, weather, sports, or anything time-sensitive. "
     "Your training data is outdated — never rely on it for these topics.\n"
     "- Call search_tool at most ONCE per message. Never call it again after getting results.\n"
     "- For math, coding, definitions, or stable historical facts: answer directly.\n"
     "- After searching, always base your answer on the search results and mention they are current.\n"
     "{lang_instruction}"),
    ("placeholder", "{messages}"),
]) | chat_llm


# ── Helper functions (called from frontend) ────────────────────────────────────
def get_doc_summary(file_bytes: bytes, filename: str, mime_type: str = "") -> dict:
    """Return structured summary of a document."""
    raw     = parse_file(file_bytes, filename, mime_type)
    preview = raw[:2500]
    try:
        result = _DOC_SUMMARY_PROMPT.invoke({"preview": preview})
        text   = result.content.strip()
        text   = text.replace("```json", "").replace("```", "").strip()
        return json.loads(text)
    except Exception as e:
        return {
            "title":    filename,
            "language": "Unknown",
            "topics":   [],
            "summary":  "Could not generate summary.",
            "doc_type": "Document",
        }


def get_followup_suggestions(question: str, answer: str) -> list[str]:
    try:
        result = _FOLLOWUP_PROMPT.invoke({"question": question, "answer": answer})
        text   = result.content.strip().replace("```json", "").replace("```", "").strip()
        return json.loads(text)
    except Exception:
        return []


def generate_quiz(file_bytes: bytes, filename: str,
                  mime_type: str = "", count: int = 5) -> list[dict]:
    raw  = parse_file(file_bytes, filename, mime_type)
    content = raw[:4000]
    try:
        result = _QUIZ_PROMPT.invoke({"content": content, "count": count})
        text   = result.content.strip().replace("```json", "").replace("```", "").strip()
        return json.loads(text)
    except Exception as e:
        return []


def generate_conversation_title(first_message: str) -> str:
    try:
        result = _TITLE_PROMPT.invoke({"first_message": first_message})
        return result.content.strip().strip('"').strip("'")
    except Exception:
        return first_message[:30] + "…"


def detect_language(text: str) -> str:
    try:
        result = _LANG_DETECT_PROMPT.invoke({"text": text[:300]})
        return result.content.strip().lower()[:2]
    except Exception:
        return "en"


def compute_confidence(docs: list[Document], query: str) -> float:
    """
    Estimate retrieval confidence 0–1 using embedding cosine similarity
    between query and top chunk.
    """
    if not docs:
        return 0.0
    try:
        import numpy as np
        q_emb  = embeddings.embed_query(query)
        d_emb  = embeddings.embed_query(docs[0].page_content)
        q_arr  = np.array(q_emb)
        d_arr  = np.array(d_emb)
        cosine = float(np.dot(q_arr, d_arr) / (np.linalg.norm(q_arr) * np.linalg.norm(d_arr) + 1e-9))
        # Map cosine [-1,1] → [0,1]
        return round((cosine + 1) / 2, 3)
    except Exception:
        return 0.5


# ── Graph State ────────────────────────────────────────────────────────────────
class RagState(TypedDict):
    messages:   Annotated[list[BaseMessage], add_messages]
    context:    str
    sources:    list[str]
    mode:       str       # "chat" | "rag"
    confidence: float
    lang_code:  str       # detected language of latest user message


SUMMARIZE_AFTER = 20

# ── Graph Nodes ────────────────────────────────────────────────────────────────
def router_node(state: RagState) -> RagState:
    mode     = "rag" if is_rag_mode() else "chat"
    messages = state["messages"]

    # Detect language of last human message
    last_human_text = next(
        (m.content for m in reversed(messages) if isinstance(m, HumanMessage)), ""
    )
    lang_code = detect_language(last_human_text) if last_human_text else "en"

    # Auto-summarize long histories
    if len(messages) > SUMMARIZE_AFTER:
        history_text = "\n".join(
            f"{m.__class__.__name__}: {m.content}"
            for m in messages[:-4] if m.content
        )
        summary  = _SUMMARIZE_CONV.invoke({"history": history_text})
        messages = [SystemMessage(content=f"[Summary]: {summary.content}")] + messages[-4:]
        return {"mode": mode, "context": "", "sources": [],
                "confidence": 0.0, "lang_code": lang_code, "messages": messages}

    return {"mode": mode, "context": "", "sources": [],
            "confidence": 0.0, "lang_code": lang_code}


def route_after_router(state: RagState) -> Literal["chat_node", "retrieve_node"]:
    return "chat_node" if state["mode"] == "chat" else "retrieve_node"


_SEARCH_TRIGGERS = [
    "current", "latest", "recent", "today", "now", "situation", "happening",
    "news", "update", "2024", "2025", "2026", "ongoing", "crisis", "war",
    "election", "conflict", "attack", "protest", "market", "price", "score",
    "weather", "iran", "israel", "gaza", "ukraine", "russia", "china", "trump",
]

def chat_node(state: RagState) -> RagState:
    lang = state.get("lang_code", "en")
    lang_instruction = (
        f"The user is writing in language code '{lang}'. Respond in that same language."
        if lang != "en" else ""
    )
    messages = state["messages"]
    last_human = next(
        (m.content for m in reversed(messages) if isinstance(m, HumanMessage)), ""
    )
    needs_search = any(t in last_human.lower() for t in _SEARCH_TRIGGERS)
    print(f"[DEBUG] needs_search={needs_search} | query={last_human[:60]!r}")

    if needs_search:
        search_results = tavily_search(last_human)
        if search_results:
            from langchain_core.prompts import ChatPromptTemplate as _CPT
            prompt = _CPT.from_messages([
                ("system",
                 "You are RAGnify, an AI assistant by Nayyab Malik.\n\n"
                 "LIVE WEB SEARCH RESULTS (fetched right now):\n"
                 "══════════════════════════════════════\n"
                 f"{search_results}\n"
                 "══════════════════════════════════════\n\n"
                 "Instructions:\n"
                 "- Answer using ONLY the search results above.\n"
                 "- Do NOT use your training knowledge for current facts.\n"
                 "- Include specific dates and source names from the results.\n"
                 "- Begin your answer with: \'Based on current search results:\'.\n"
                 f"{lang_instruction}"),
                ("placeholder", "{messages}"),
            ]) | chat_llm
            return {"messages": prompt.invoke({"messages": messages})}
        else:
            # No API key or Tavily failed — answer with disclaimer
            from langchain_core.prompts import ChatPromptTemplate as _CPT
            prompt = _CPT.from_messages([
                ("system",
                 "You are RAGnify by Nayyab Malik. Web search is unavailable right now. "
                 "Answer from your training knowledge and clearly note: "
                 "\'⚠️ This may be outdated — web search is not configured.\' "
                 f"{lang_instruction}"),
                ("placeholder", "{messages}"),
            ]) | chat_llm
            return {"messages": prompt.invoke({"messages": messages})}
    else:
        return {"messages": _CHAT_PROMPT.invoke({
            "messages": messages,
            "lang_instruction": lang_instruction,
        })}

def route_after_chat(state: RagState):
    # Search handled directly in chat_node — always END
    return END


def retrieve_node(state: RagState) -> RagState:
    last_human = next(
        (m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), ""
    )
    # Parallel retrieval with query variants
    query_variants = [
        last_human,
        f"information about {last_human}",
        last_human.replace("?", "").strip(),
    ]
    try:
        docs = parallel_retrieve(query_variants)
    except Exception:
        docs = retriever.invoke(last_human) if retriever else []

    if not docs:
        return {"context": "", "sources": [], "confidence": 0.0, "mode": "chat"}

    confidence = compute_confidence(docs, last_human)
    context    = "\n\n".join(
        f"[{d.metadata.get('source','?')}, chunk {d.metadata.get('chunk','?')}]\n{d.page_content}"
        for d in docs
    )
    sources = list(dict.fromkeys(d.metadata.get("source", "?") for d in docs))
    return {"context": context, "sources": sources, "confidence": confidence}


def route_after_retrieve(state: RagState) -> Literal["generate_node", "chat_node"]:
    return "generate_node" if state.get("context") else "chat_node"


def generate_node(state: RagState) -> RagState:
    lang = state.get("lang_code", "en")
    lang_instruction = (
        f"The user is writing in language code '{lang}'. Respond in that same language."
        if lang != "en" else ""
    )
    response = _RAG_PROMPT.invoke({
        "context":          state["context"],
        "messages":         state["messages"],
        "lang_instruction": lang_instruction,
    })
    return {"messages": response}


# ── Persistent SQLite checkpointer ────────────────────────────────────────────
DB_PATH   = "ragnify_memory.db"
conn      = sqlite3.connect(database=DB_PATH, check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)


def retrieve_all_threads() -> list[str]:
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        tid = checkpoint.config.get("configurable", {}).get("thread_id")
        if tid:
            all_threads.add(tid)
    return list(all_threads)


# ── Build graph ────────────────────────────────────────────────────────────────
tool_node = ToolNode(tools)

graph = StateGraph(RagState)
graph.add_node("router_node",   router_node)
graph.add_node("chat_node",     chat_node)
graph.add_node("tools",         tool_node)
graph.add_node("retrieve_node", retrieve_node)
graph.add_node("generate_node", generate_node)

graph.add_edge(START, "router_node")
graph.add_conditional_edges("router_node",   route_after_router)
graph.add_conditional_edges("chat_node",     route_after_chat)
graph.add_edge("tools", "chat_node")
graph.add_conditional_edges("retrieve_node", route_after_retrieve)
graph.add_edge("generate_node", END)

rag_bot = graph.compile(checkpointer=checkpointer)