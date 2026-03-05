"""
RAGnify — Frontend (Final Clean Version)
By Nayyab Malik
"""

import datetime, json, re, uuid
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from chatbot_complete_backend import (
    SUPPORTED_EXTENSIONS,
    build_vectorstore,
    compute_confidence,
    generate_conversation_title,
    generate_quiz,
    get_doc_summary,
    get_followup_suggestions,
    get_indexed_docs,
    is_rag_mode,
    rag_bot,
    rebuild_vectorstore_without,
    retrieve_all_threads,
)

# ── PAGE CONFIG ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAGnify · by Nayyab Malik",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── DESIGN SYSTEM ──────────────────────────────────────────────────────────────
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&display=swap');
:root {
  --bg:#07090f; --surface:#0e1117; --surface2:#151b24; --border:#1f2d3d;
  --amber:#f5a623; --teal:#00c9a7; --muted:#5a6a7a;
  --text:#dce8f0; --text-dim:#7a8fa0; --danger:#e05c5c; --radius:10px;
}
html,body,[data-testid="stAppViewContainer"]{background:var(--bg)!important;color:var(--text)!important;font-family:'DM Mono',monospace!important;}
[data-testid="stSidebar"]{background:var(--surface)!important;border-right:1px solid var(--border)!important;}
h1,h2,h3,h4{font-family:'Syne',sans-serif!important;}
.ragnify-title{font-family:'Syne',sans-serif;font-weight:800;font-size:1.6rem;background:linear-gradient(135deg,var(--amber),var(--teal));-webkit-background-clip:text;-webkit-text-fill-color:transparent;letter-spacing:-.5px;margin-bottom:2px;}
.ragnify-sub{font-size:.72rem;color:var(--muted);letter-spacing:.08em;margin-bottom:12px;}
.pill{display:inline-block;padding:3px 12px;border-radius:20px;font-size:.72rem;font-weight:600;letter-spacing:.06em;font-family:'Syne',sans-serif;}
.pill-rag{background:#0d3326;color:var(--teal);border:1px solid var(--teal);}
.pill-chat{background:#1a2535;color:var(--amber);border:1px solid var(--amber);}
.doc-card{background:var(--surface2);border:1px solid var(--border);border-radius:var(--radius);padding:8px 10px;margin:5px 0;}
.doc-card .doc-name{font-family:'Syne',sans-serif;font-size:.82rem;font-weight:600;color:var(--text);word-break:break-all;}
.doc-card .doc-meta{font-size:.70rem;color:var(--text-dim);margin-top:2px;}
.stButton>button{background:var(--surface2)!important;color:var(--text)!important;border:1px solid var(--border)!important;border-radius:var(--radius)!important;font-family:'DM Mono',monospace!important;font-size:.76rem!important;transition:border-color .2s,color .2s!important;}
.stButton>button:hover{border-color:var(--teal)!important;color:var(--teal)!important;}
.main-header{font-family:'Syne',sans-serif;font-size:2.4rem;font-weight:800;background:linear-gradient(135deg,var(--amber) 0%,var(--teal) 100%);-webkit-background-clip:text;-webkit-text-fill-color:transparent;line-height:1.1;margin-bottom:4px;}
.main-byline{font-size:.75rem;color:var(--muted);letter-spacing:.1em;margin-bottom:16px;}
.banner{border-radius:var(--radius);padding:10px 16px;font-size:.82rem;font-family:'DM Mono',monospace;margin-bottom:10px;}
.banner-rag{background:#0d2b23;border:1px solid var(--teal);color:var(--teal);}
.banner-chat{background:#1a1f2b;border:1px solid var(--amber);color:var(--amber);}
.conf-wrap{margin:6px 0 10px 0;}
.conf-label{font-size:.70rem;color:var(--text-dim);margin-bottom:3px;}
.conf-bar-bg{background:var(--surface2);border-radius:6px;height:6px;overflow:hidden;border:1px solid var(--border);}
.conf-bar-fill{height:100%;border-radius:6px;transition:width .5s ease;}
.src-tag{display:inline-block;background:var(--surface2);border:1px solid var(--border);color:var(--text-dim);padding:2px 8px;border-radius:5px;font-size:.70rem;margin:2px;}
.summary-card{background:var(--surface2);border:1px solid var(--teal);border-radius:var(--radius);padding:14px 18px;margin:10px 0;animation:fadeIn .4s ease;}
.summary-card h4{font-family:'Syne',sans-serif;color:var(--teal);font-size:1rem;margin:0 0 8px 0;}
.topic-chip{display:inline-block;background:#0d3326;color:var(--teal);border:1px solid #1a5940;border-radius:5px;padding:2px 9px;font-size:.70rem;margin:2px;}
.quiz-card{background:var(--surface2);border:1px solid var(--border);border-radius:var(--radius);padding:14px 18px;margin:8px 0;}
.quiz-card .q-num{font-size:.70rem;color:var(--amber);font-family:'Syne',sans-serif;font-weight:700;}
.quiz-card .q-text{font-size:.88rem;margin:4px 0 10px 0;color:var(--text);}
[data-testid="stMetric"]{background:var(--surface2);border:1px solid var(--border);border-radius:var(--radius);padding:10px 14px;}
[data-testid="stMetricLabel"]{color:var(--text-dim)!important;font-size:.72rem!important;}
[data-testid="stMetricValue"]{color:var(--teal)!important;font-family:'Syne',sans-serif!important;font-weight:700!important;}
[data-testid="stChatMessage"]{background:var(--surface2)!important;border:1px solid var(--border)!important;border-radius:var(--radius)!important;margin:4px 0!important;}
[data-testid="stChatInput"] textarea{background:var(--surface2)!important;border:1px solid var(--border)!important;border-radius:var(--radius)!important;color:var(--text)!important;font-family:'DM Mono',monospace!important;}
[data-testid="stChatInput"] textarea:focus{border-color:var(--teal)!important;box-shadow:0 0 0 2px rgba(0,201,167,.15)!important;}
hr{border-color:var(--border)!important;}
::-webkit-scrollbar{width:5px;}
::-webkit-scrollbar-track{background:var(--bg);}
::-webkit-scrollbar-thumb{background:var(--border);border-radius:4px;}
/* File uploader */
[data-testid="stFileUploader"]{background:#0e1520!important;border:1.5px dashed #00c9a7!important;border-radius:12px!important;padding:10px 18px!important;}
[data-testid="stFileUploaderDropzone"]{background:transparent!important;border:none!important;display:flex!important;flex-direction:row!important;align-items:center!important;gap:14px!important;}
[data-testid="stFileUploaderDropzone"] svg{color:#5a6a7a!important;width:32px!important;height:32px!important;flex-shrink:0!important;}
[data-testid="stFileUploaderDropzone"] div:first-of-type{flex:1!important;}
[data-testid="stFileUploaderDropzone"] p,[data-testid="stFileUploaderDropzone"] span{color:#dce8f0!important;font-family:'DM Mono',monospace!important;font-size:.82rem!important;margin:0!important;}
[data-testid="stFileUploaderDropzone"] small{color:#5a6a7a!important;font-family:'DM Mono',monospace!important;font-size:.72rem!important;}
[data-testid="stFileUploaderDropzone"] button{background:transparent!important;border:1.5px solid #00c9a7!important;color:#00c9a7!important;border-radius:8px!important;font-family:'DM Mono',monospace!important;font-size:.80rem!important;font-weight:600!important;padding:6px 18px!important;white-space:nowrap!important;flex-shrink:0!important;transition:background .15s!important;}
[data-testid="stFileUploaderDropzone"] button:hover{background:#0d3326!important;}
/* Web search badge */
@keyframes fadeIn{from{opacity:0;transform:translateY(6px);}to{opacity:1;transform:translateY(0);}}
@keyframes wsPulse{0%,100%{opacity:1;}50%{opacity:.4;}}
@keyframes wsSpin{from{transform:rotate(0deg);}to{transform:rotate(360deg);}}
.websearch-badge{display:inline-flex;align-items:center;gap:8px;background:#0d1f2d;border:1px solid #00c9a7;border-radius:20px;padding:5px 14px 5px 10px;font-family:'DM Mono',monospace;font-size:.78rem;color:#00c9a7;animation:wsPulse 1.4s ease-in-out infinite;margin-bottom:8px;}
.websearch-badge .ws-spin{display:inline-block;animation:wsSpin 1s linear infinite;font-size:.9rem;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ── Clipboard helper (executes JS when copy button clicked) ───────────────────
if "_copy_text" in st.session_state and st.session_state["_copy_text"]:
    _txt = st.session_state.pop("_copy_text").replace("\\", "\\\\").replace("`", "\\`").replace("\n", "\\n")
    st.components.v1.html(
        f'''<script>
        navigator.clipboard.writeText(`{_txt}`).catch(()=>{{
            const el=document.createElement("textarea");
            el.value=`{_txt}`;
            document.body.appendChild(el);
            el.select();
            document.execCommand("copy");
            document.body.removeChild(el);
        }});
        </script>''',
        height=0,
    )


# ── HELPERS ────────────────────────────────────────────────────────────────────
def gen_tid():
    return f"sess-{uuid.uuid4().hex[:8]}"

def reset_chat():
    tid = gen_tid()
    st.session_state["thread_id"]          = tid
    st.session_state["message_history"]    = []
    st.session_state["thread_titles"][tid] = None
    st.session_state["followups"]          = []
    st.session_state["last_conf"]          = None
    st.session_state["last_sources"]       = []
    st.session_state["last_question"]      = ""
    st.session_state["last_answer"]        = ""
    if tid not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(tid)

def load_history(thread_id):
    state = rag_bot.get_state(config={"configurable": {"thread_id": thread_id}})
    history = []
    for msg in state.values.get("messages", []):
        if isinstance(msg, HumanMessage) and msg.content:
            history.append({"role": "user",      "content": msg.content})
        elif isinstance(msg, AIMessage) and msg.content:
            history.append({"role": "assistant", "content": msg.content})
    return history

def export_md() -> str:
    tid   = st.session_state["thread_id"]
    title = st.session_state["thread_titles"].get(tid) or tid
    lines = [f"# {title}\n_Exported from RAGnify by Nayyab Malik — "
             f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}_\n"]
    for m in st.session_state["message_history"]:
        role = "**You**" if m["role"] == "user" else "**RAGnify**"
        lines.append(f"{role}:\n{m['content']}\n")
    return "\n".join(lines)

def confidence_color(score: float) -> str:
    if score >= 0.75: return "#00c9a7"
    if score >= 0.50: return "#f5a623"
    return "#e05c5c"

def export_eval_csv() -> str:
    rows = ["turn_id,question_preview,rating,confidence,sources,timestamp"]
    for ev in st.session_state["eval_log"]:
        q   = ev.get("question", "")[:60].replace(",", " ")
        r   = ev.get("rating", "")
        c   = ev.get("confidence", "")
        src = "|".join(ev.get("sources", []))
        ts  = ev.get("timestamp", "")
        rows.append(f"{ev.get('turn_id','')},{q},{r},{c},{src},{ts}")
    return "\n".join(rows)

def process_uploads(files):
    if not files:
        return False
    new_files = [f for f in files if f.name not in st.session_state["file_store"]]
    if not new_files:
        return False
    bar = st.progress(0, text="Indexing…")
    for i, f in enumerate(new_files):
        raw    = f.read()
        chunks = build_vectorstore(raw, f.name, f.type)
        with st.spinner(f"Analysing {f.name}…"):
            summary = get_doc_summary(raw, f.name, f.type)
        st.session_state["file_store"][f.name] = {
            "bytes":   raw,
            "mime":    f.type,
            "chunks":  chunks,
            "size_kb": round(len(raw) / 1024, 1),
            "summary": summary,
        }
        bar.progress((i + 1) / len(new_files), text=f"✅ {f.name} ({chunks} chunks)")
    bar.empty()
    return True


# ── SESSION STATE ──────────────────────────────────────────────────────────────
defaults = {
    "message_history":  [],
    "thread_id":        gen_tid(),
    "chat_threads":     [],
    "thread_titles":    {},
    "file_store":       {},
    "last_conf":        None,
    "last_sources":     [],
    "followups":        [],
    "last_question":    "",
    "last_answer":      "",
    "quiz_data":        {},
    "quiz_answers":     {},
    "show_quiz":        None,
    "pending_input":    "",
    "show_uploader":    False,
    # Analytics
    "conf_history":     [],
    "source_counter":   {},
    "mode_counter":     {"rag": 0, "chat": 0},
    # Evaluation
    "eval_log":         [],
    "pending_eval_tid": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

tid = st.session_state["thread_id"]
if tid not in st.session_state["chat_threads"]:
    st.session_state["chat_threads"].append(tid)
if tid not in st.session_state["thread_titles"]:
    st.session_state["thread_titles"][tid] = None

if len(st.session_state["chat_threads"]) <= 1:
    for persisted_tid in retrieve_all_threads():
        if persisted_tid not in st.session_state["chat_threads"]:
            st.session_state["chat_threads"].append(persisted_tid)
            st.session_state["thread_titles"][persisted_tid] = persisted_tid[:16]

CONFIG = {"configurable": {"thread_id": st.session_state["thread_id"]}, "recursion_limit": 10}


# ── SIDEBAR ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="ragnify-title">⚡ RAGnify</div>', unsafe_allow_html=True)
    st.markdown('<div class="ragnify-sub">BY NAYYAB MALIK · AI ENGINEER</div>', unsafe_allow_html=True)

    pill_cls = "pill-rag" if is_rag_mode() else "pill-chat"
    pill_lbl = "📄 RAG MODE" if is_rag_mode() else "💬 CHAT MODE"
    st.markdown(f'<span class="pill {pill_cls}">{pill_lbl}</span>', unsafe_allow_html=True)
    st.divider()

    docs = get_indexed_docs()
    if docs:
        st.markdown("**📂 Indexed Documents**")
        for name in list(docs.keys()):
            meta         = st.session_state["file_store"].get(name, {})
            summary_data = meta.get("summary", {})
            doc_type     = summary_data.get("doc_type", "Document")
            col_doc, col_del = st.columns([5, 1])
            with col_doc:
                st.markdown(
                    f'<div class="doc-card">'
                    f'<div class="doc-name">📄 {name}</div>'
                    f'<div class="doc-meta">{doc_type} · {docs[name]} chunks · {meta.get("size_kb","?")} KB</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            with col_del:
                if st.button("✕", key=f"del_{name}", help=f"Remove {name}"):
                    rebuild_vectorstore_without(name, st.session_state["file_store"])
                    del st.session_state["file_store"][name]
                    if name in st.session_state["quiz_data"]:
                        del st.session_state["quiz_data"][name]
                    if st.session_state["show_quiz"] == name:
                        st.session_state["show_quiz"] = None
                    st.rerun()

        st.markdown("**🧩 Generate Quiz**")
        quiz_target = st.selectbox(
            "Select document", options=list(docs.keys()),
            label_visibility="collapsed", key="quiz_select",
        )
        q_count = st.slider("Questions", 3, 10, 5, key="quiz_count")
        if st.button("⚡ Generate Quiz", width='stretch'):
            with st.spinner("Generating quiz…"):
                meta = st.session_state["file_store"].get(quiz_target, {})
                quiz = generate_quiz(meta["bytes"], quiz_target, meta["mime"], q_count)
                st.session_state["quiz_data"][quiz_target] = quiz
                st.session_state["quiz_answers"] = {}
                st.session_state["show_quiz"]    = quiz_target
            st.rerun()

        st.divider()

    st.markdown("**🗂 Sessions**")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("➕ New", width='stretch'):
            reset_chat(); st.rerun()
    with c2:
        if st.session_state["message_history"]:
            st.download_button(
                "⬇ Export", data=export_md(),
                file_name=f"ragnify_{st.session_state['thread_id']}.md",
                mime="text/markdown", width='stretch',
            )

    for t in st.session_state["chat_threads"][::-1]:
        active = t == st.session_state["thread_id"]
        title  = st.session_state["thread_titles"].get(t) or t[:18]
        label  = f"{'▶ ' if active else ''}{title}"
        if st.button(label, key=f"tb_{t}", width='stretch',
                     type="primary" if active else "secondary"):
            st.session_state["message_history"] = load_history(t)
            st.session_state["thread_id"]       = t
            st.session_state["followups"]       = []
            st.session_state["last_conf"]       = None
            st.rerun()


# ── MAIN HEADER ────────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">⚡ RAGnify</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="main-byline">RETRIEVAL-AUGMENTED INTELLIGENCE · BY NAYYAB MALIK</div>',
    unsafe_allow_html=True,
)

# ── TABS ───────────────────────────────────────────────────────────────────────
tab_chat, tab_analytics, tab_eval = st.tabs(["💬 Chat", "📊 Analytics", "🧪 Evaluation"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — CHAT
# ══════════════════════════════════════════════════════════════════════════════
with tab_chat:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Mode",            "RAG" if is_rag_mode() else "Chat")
    m2.metric("Documents",       len(get_indexed_docs()))
    m3.metric("Messages",        len(st.session_state["message_history"]))
    conf_val = st.session_state.get("last_conf")
    m4.metric("Last Confidence", f"{int(conf_val*100)}%" if conf_val is not None else "—")

    if not is_rag_mode():
        st.markdown(
            '<div class="banner banner-chat">💬 <b>Chat Mode</b> — Web search enabled. '
            'Click 📂 Browse Files below to upload a document and activate RAG mode.</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="banner banner-rag">📄 <b>RAG Mode</b> — Grounded on '
            f'<b>{len(get_indexed_docs())}</b> document(s). '
            f'Low-confidence queries fall back to web search.</div>',
            unsafe_allow_html=True,
        )

    if st.session_state["last_sources"]:
        tags = "".join(
            f'<span class="src-tag">📄 {s}</span>'
            for s in st.session_state["last_sources"]
        )
        st.markdown(f"<div style='margin-bottom:6px'>Sources: {tags}</div>",
                    unsafe_allow_html=True)

    if conf_val is not None:
        pct   = int(conf_val * 100)
        color = confidence_color(conf_val)
        label = "High" if pct >= 75 else "Medium" if pct >= 50 else "Low"
        st.markdown(
            f'<div class="conf-wrap">'
            f'<div class="conf-label">Retrieval confidence: {pct}% ({label})</div>'
            f'<div class="conf-bar-bg">'
            f'<div class="conf-bar-fill" style="width:{pct}%;background:{color}"></div>'
            f'</div></div>',
            unsafe_allow_html=True,
        )

    st.divider()

    # ── Quiz panel ─────────────────────────────────────────────────────────────
    show_quiz = st.session_state.get("show_quiz")
    if show_quiz and show_quiz in st.session_state["quiz_data"]:
        quiz = st.session_state["quiz_data"][show_quiz]
        st.markdown(f"### 🧩 Quiz — *{show_quiz}*")
        score        = 0
        answered_all = True
        for i, q in enumerate(quiz):
            with st.container():
                st.markdown(
                    f'<div class="quiz-card"><div class="q-num">Q{i+1}</div>'
                    f'<div class="q-text">{q["question"]}</div></div>',
                    unsafe_allow_html=True,
                )
                chosen = st.radio(
                    f"q_{i}", options=q.get("options", []),
                    key=f"quiz_q_{i}", label_visibility="collapsed",
                )
                if chosen:
                    st.session_state["quiz_answers"][i] = chosen
                else:
                    answered_all = False

        if answered_all and st.button("✅ Submit Quiz"):
            for i, q in enumerate(quiz):
                chosen         = st.session_state["quiz_answers"].get(i, "")
                correct_letter = q.get("answer", "")
                correct_option = next(
                    (o for o in q.get("options", []) if o.startswith(correct_letter)), ""
                )
                if chosen == correct_option:
                    score += 1
            st.success(
                f"🎉 Score: {score}/{len(quiz)} — "
                f"{'Excellent!' if score == len(quiz) else 'Keep practicing!'}"
            )
            for i, q in enumerate(quiz):
                st.info(f"**Q{i+1} explanation:** {q.get('explanation','')}")

        if st.button("✕ Close Quiz"):
            st.session_state["show_quiz"] = None
            st.rerun()
        st.divider()

    # ── Chat history ───────────────────────────────────────────────────────────
    for idx, msg in enumerate(st.session_state["message_history"]):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                copy_key = f"copy_hist_{idx}"
                if st.button("📋 Copy", key=copy_key, help="Copy response"):
                    st.session_state["_copy_text"] = msg["content"]
                    st.toast("✅ Copied to clipboard!")
                # Rating — only on the last assistant message
                ptid = st.session_state.get("pending_eval_tid")
                _hist = st.session_state["message_history"]
                if ptid and idx == len(_hist) - 1:
                    st.markdown('<div style="font-size:.72rem;color:#5a6a7a;margin-top:4px;">Rate this response:</div>', unsafe_allow_html=True)
                    _rc1, _rc2, _ = st.columns([1, 1, 8])
                    with _rc1:
                        if st.button("👍", key=f"up_{ptid}"):
                            ev = next((e for e in st.session_state["eval_log"] if e["turn_id"] == ptid), None)
                            if ev: ev["rating"] = "positive"
                            st.session_state["pending_eval_tid"] = None
                            st.rerun()
                    with _rc2:
                        if st.button("👎", key=f"dn_{ptid}"):
                            ev = next((e for e in st.session_state["eval_log"] if e["turn_id"] == ptid), None)
                            if ev: ev["rating"] = "negative"
                            st.session_state["pending_eval_tid"] = None
                            st.rerun()

    # ── Follow-up suggestions ──────────────────────────────────────────────────
    if st.session_state["followups"]:
        st.markdown(
            '<div style="font-size:.72rem;color:#5a6a7a;margin:6px 0 4px 0;">'
            '💡 Suggested follow-ups:</div>',
            unsafe_allow_html=True,
        )
        cols = st.columns(len(st.session_state["followups"]))
        for i, suggestion in enumerate(st.session_state["followups"]):
            with cols[i]:
                if st.button(suggestion, key=f"followup_{i}", width='stretch'):
                    st.session_state["pending_input"] = suggestion
                    st.session_state["followups"]     = []
                    st.rerun()

    # Rating rendered inside chat history loop below

    # ── Browse Files toggle ────────────────────────────────────────────────────
    att_col, _ = st.columns([1, 9])
    with att_col:
        attach_label = "📂" if not st.session_state.get("show_uploader") else "✕ Close"
        if st.button(attach_label, key="toggle_attach", width='stretch'):
            st.session_state["show_uploader"] = not st.session_state.get("show_uploader", False)
            st.rerun()

    inline_upload = None
    if st.session_state.get("show_uploader"):
        inline_upload = st.file_uploader(
            "Drop files here or click Browse files",
            type=SUPPORTED_EXTENSIONS,
            accept_multiple_files=True,
            key="inline_uploader",
            label_visibility="collapsed",
        )
        if inline_upload:
            st.session_state["show_uploader"] = False

    if process_uploads(list(inline_upload or [])):
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# CHAT INPUT — top-level so st.chat_input stays fixed at viewport bottom
# ══════════════════════════════════════════════════════════════════════════════
hint = (
    "Ask about your documents… (falls back to web if not found)"
    if is_rag_mode() else
    "Ask anything…"
)
user_input = st.chat_input(hint) or st.session_state.pop("pending_input", "")

if user_input:
    tid = st.session_state["thread_id"]
    if not st.session_state["thread_titles"].get(tid):
        with st.spinner("Naming conversation…"):
            title = generate_conversation_title(user_input)
        st.session_state["thread_titles"][tid] = title

    st.session_state["message_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        websearch_placeholder = st.empty()
        response_placeholder  = st.empty()
        full_response = ""

        with st.spinner(""):
            for message_chunk, _ in rag_bot.stream(
                {
                    "messages":   [HumanMessage(content=user_input)],
                    "context":    "",
                    "sources":    [],
                    "mode":       "",
                    "confidence": 0.0,
                    "lang_code":  "en",
                },
                config=CONFIG,
                stream_mode="messages",
            ):
                if isinstance(message_chunk, AIMessage):
                    if hasattr(message_chunk, "tool_calls") and message_chunk.tool_calls:
                        # DuckDuckGo tool invoked — show animated web-search badge
                        websearch_placeholder.markdown(
                            '<div class="websearch-badge">'
                            '<span class="ws-spin">🔍</span>'
                            '&nbsp;Searching the web…'
                            '</div>',
                            unsafe_allow_html=True,
                        )
                    elif message_chunk.content:
                        # Answer streaming in — hide badge, stream text
                        websearch_placeholder.empty()
                        full_response += message_chunk.content
                        response_placeholder.markdown(full_response + "▌")

        response_placeholder.markdown(full_response)
        if st.button("📋 Copy", key=f"copy_live_{len(st.session_state['message_history'])}", help="Copy response"):
            st.session_state["_copy_text"] = full_response
            st.toast("✅ Copied!")

    st.session_state["message_history"].append({"role": "assistant", "content": full_response})
    st.session_state["last_question"] = user_input
    st.session_state["last_answer"]   = full_response

    found = re.findall(r'\[Source:\s*([^\],]+)', full_response)
    st.session_state["last_sources"] = list(dict.fromkeys(found))

    try:
        graph_state = rag_bot.get_state(CONFIG)
        conf = graph_state.values.get("confidence", None)
        st.session_state["last_conf"] = float(conf) if conf else None
    except Exception:
        st.session_state["last_conf"] = None

    current_conf = st.session_state["last_conf"]
    current_mode = "rag" if is_rag_mode() else "chat"

    st.session_state["conf_history"].append({
        "turn":      len(st.session_state["conf_history"]) + 1,
        "conf":      round(current_conf * 100, 1) if current_conf else 0,
        "mode":      current_mode,
        "timestamp": datetime.datetime.now().strftime("%H:%M:%S"),
    })
    st.session_state["mode_counter"][current_mode] += 1
    for src in st.session_state["last_sources"]:
        st.session_state["source_counter"][src] = \
            st.session_state["source_counter"].get(src, 0) + 1

    turn_id = f"turn-{uuid.uuid4().hex[:8]}"
    st.session_state["eval_log"].append({
        "turn_id":    turn_id,
        "question":   user_input,
        "answer":     full_response[:300],
        "rating":     "pending",
        "confidence": round(current_conf * 100, 1) if current_conf else None,
        "sources":    list(st.session_state["last_sources"]),
        "mode":       current_mode,
        "timestamp":  datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    })
    st.session_state["pending_eval_tid"] = turn_id

    with st.spinner("Generating suggestions…"):
        suggestions = get_followup_suggestions(user_input, full_response[:800])
    st.session_state["followups"] = suggestions if isinstance(suggestions, list) else []

    st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
with tab_analytics:
    st.markdown("### 📊 Session Analytics")

    conf_hist   = st.session_state["conf_history"]
    src_counter = st.session_state["source_counter"]
    mode_ctr    = st.session_state["mode_counter"]
    eval_log    = st.session_state["eval_log"]

    total_turns = len(conf_hist)
    rag_turns   = mode_ctr.get("rag", 0)
    chat_turns  = mode_ctr.get("chat", 0)
    avg_conf    = (sum(e["conf"] for e in conf_hist) / total_turns) if total_turns else 0
    rated       = [e for e in eval_log if e["rating"] in ("positive", "negative")]
    pos_rate    = (sum(1 for e in rated if e["rating"] == "positive") / len(rated) * 100) if rated else 0

    a1, a2, a3, a4, a5 = st.columns(5)
    a1.metric("Total Turns",    total_turns)
    a2.metric("RAG Turns",      rag_turns)
    a3.metric("Chat Turns",     chat_turns)
    a4.metric("Avg Confidence", f"{avg_conf:.1f}%")
    a5.metric("👍 Rate",        f"{pos_rate:.0f}%" if rated else "—")

    st.divider()

    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("#### Confidence Over Turns")
        if conf_hist:
            try:
                import plotly.graph_objects as go
                fig = go.Figure()
                turns  = [e["turn"] for e in conf_hist]
                confs  = [e["conf"] for e in conf_hist]
                colors = ["#00c9a7" if c >= 75 else "#f5a623" if c >= 50 else "#e05c5c" for c in confs]
                fig.add_trace(go.Scatter(
                    x=turns, y=confs, mode="lines+markers",
                    line=dict(color="#00c9a7", width=2),
                    marker=dict(color=colors, size=8),
                    hovertemplate="Turn %{x}<br>Confidence: %{y}%<extra></extra>",
                ))
                fig.add_hline(y=75, line_dash="dot", line_color="#00c9a7", opacity=0.4,
                              annotation_text="High threshold")
                fig.add_hline(y=50, line_dash="dot", line_color="#f5a623", opacity=0.4,
                              annotation_text="Medium threshold")
                fig.update_layout(
                    paper_bgcolor="#07090f", plot_bgcolor="#0e1117",
                    font=dict(color="#dce8f0", family="DM Mono"),
                    xaxis=dict(title="Turn", gridcolor="#1f2d3d", color="#7a8fa0"),
                    yaxis=dict(title="Confidence %", range=[0, 100], gridcolor="#1f2d3d", color="#7a8fa0"),
                    margin=dict(l=10, r=10, t=10, b=10), height=280,
                )
                st.plotly_chart(fig, width='stretch')
            except ImportError:
                st.line_chart({e["turn"]: e["conf"] for e in conf_hist})
        else:
            st.info("No turns yet — start chatting to see the confidence trend.")

    with col_r:
        st.markdown("#### Source Hit Frequency")
        if src_counter:
            try:
                import plotly.graph_objects as go
                fig2 = go.Figure(go.Bar(
                    x=list(src_counter.values()), y=list(src_counter.keys()),
                    orientation="h", marker_color="#f5a623",
                    hovertemplate="%{y}<br>Hits: %{x}<extra></extra>",
                ))
                fig2.update_layout(
                    paper_bgcolor="#07090f", plot_bgcolor="#0e1117",
                    font=dict(color="#dce8f0", family="DM Mono"),
                    xaxis=dict(title="Hits", gridcolor="#1f2d3d", color="#7a8fa0"),
                    yaxis=dict(gridcolor="#1f2d3d", color="#7a8fa0"),
                    margin=dict(l=10, r=10, t=10, b=10), height=280,
                )
                st.plotly_chart(fig2, width='stretch')
            except ImportError:
                st.bar_chart(src_counter)
        else:
            st.info("No document sources cited yet.")

    st.divider()

    col3, col4 = st.columns(2)

    with col3:
        st.markdown("#### Mode Distribution")
        if total_turns:
            try:
                import plotly.graph_objects as go
                fig3 = go.Figure(go.Pie(
                    labels=["RAG", "Chat"],
                    values=[rag_turns, chat_turns],
                    marker=dict(colors=["#00c9a7", "#f5a623"]),
                    hole=0.5, textinfo="label+percent",
                    hovertemplate="%{label}: %{value} turns<extra></extra>",
                ))
                fig3.update_layout(
                    paper_bgcolor="#07090f",
                    font=dict(color="#dce8f0", family="DM Mono"),
                    margin=dict(l=10, r=10, t=10, b=10), height=240,
                    showlegend=False,
                )
                st.plotly_chart(fig3, width='stretch')
            except ImportError:
                st.write(f"RAG: {rag_turns}  |  Chat: {chat_turns}")
        else:
            st.info("No turns yet.")

    with col4:
        st.markdown("#### Confidence Buckets")
        if conf_hist:
            try:
                import plotly.graph_objects as go
                high   = sum(1 for e in conf_hist if e["conf"] >= 75)
                medium = sum(1 for e in conf_hist if 50 <= e["conf"] < 75)
                low    = sum(1 for e in conf_hist if e["conf"] < 50)
                fig4 = go.Figure(go.Bar(
                    x=["High (≥75%)", "Medium (50-74%)", "Low (<50%)"],
                    y=[high, medium, low],
                    marker_color=["#00c9a7", "#f5a623", "#e05c5c"],
                    hovertemplate="%{x}: %{y} turns<extra></extra>",
                ))
                fig4.update_layout(
                    paper_bgcolor="#07090f", plot_bgcolor="#0e1117",
                    font=dict(color="#dce8f0", family="DM Mono"),
                    xaxis=dict(gridcolor="#1f2d3d", color="#7a8fa0"),
                    yaxis=dict(title="Turns", gridcolor="#1f2d3d", color="#7a8fa0"),
                    margin=dict(l=10, r=10, t=10, b=10), height=240,
                )
                st.plotly_chart(fig4, width='stretch')
            except ImportError:
                st.write(f"High: {high}  |  Medium: {medium}  |  Low: {low}")
        else:
            st.info("No turns yet.")

    if st.button("🗑 Reset Analytics", key="reset_analytics"):
        st.session_state["conf_history"]   = []
        st.session_state["source_counter"] = {}
        st.session_state["mode_counter"]   = {"rag": 0, "chat": 0}
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — EVALUATION
# ══════════════════════════════════════════════════════════════════════════════
with tab_eval:
    st.markdown("### 🧪 Response Evaluation")

    eval_log = st.session_state["eval_log"]
    rated    = [e for e in eval_log if e["rating"] in ("positive", "negative")]
    pending  = [e for e in eval_log if e["rating"] == "pending"]
    positive = sum(1 for e in rated if e["rating"] == "positive")
    negative = len(rated) - positive
    pos_rate = (positive / len(rated) * 100) if rated else 0

    e1, e2, e3, e4 = st.columns(4)
    e1.metric("Total Rated",  len(rated))
    e2.metric("👍 Positive",  positive)
    e3.metric("👎 Negative",  negative)
    e4.metric("Satisfaction", f"{pos_rate:.0f}%" if rated else "—")

    st.divider()

    if rated:
        st.markdown("#### Satisfaction Trend")
        try:
            import plotly.graph_objects as go
            pos_count = 0
            cum_pos, cum_all = [], []
            for i, e in enumerate(rated):
                if e["rating"] == "positive":
                    pos_count += 1
                cum_pos.append(pos_count)
                cum_all.append(i + 1)
            sat_pct = [p / a * 100 for p, a in zip(cum_pos, cum_all)]
            fig5 = go.Figure()
            fig5.add_trace(go.Scatter(
                x=cum_all, y=sat_pct, mode="lines+markers",
                line=dict(color="#00c9a7", width=2),
                marker=dict(color="#00c9a7", size=7),
                hovertemplate="After %{x} ratings<br>Satisfaction: %{y:.1f}%<extra></extra>",
            ))
            fig5.add_hline(y=80, line_dash="dot", line_color="#f5a623",
                           opacity=0.5, annotation_text="80% target")
            fig5.update_layout(
                paper_bgcolor="#07090f", plot_bgcolor="#0e1117",
                font=dict(color="#dce8f0", family="DM Mono"),
                xaxis=dict(title="Cumulative Ratings", gridcolor="#1f2d3d", color="#7a8fa0"),
                yaxis=dict(title="Satisfaction %", range=[0, 100], gridcolor="#1f2d3d", color="#7a8fa0"),
                margin=dict(l=10, r=10, t=10, b=10), height=260,
            )
            st.plotly_chart(fig5, width='stretch')
        except ImportError:
            pass
        st.divider()

    if pending:
        st.markdown(f"#### ⏳ Awaiting Rating ({len(pending)})")
        for ev in pending[-5:]:
            with st.expander(f"Turn {ev['turn_id']} — {ev['question'][:60]}…"):
                st.markdown(f"**Q:** {ev['question']}")
                st.markdown(f"**A:** {ev['answer'][:300]}{'…' if len(ev['answer']) > 300 else ''}")
                st.markdown(f"Confidence: `{ev['confidence']}%` | Mode: `{ev['mode']}`")
                rc1, rc2, _ = st.columns([1, 1, 8])
                with rc1:
                    if st.button("👍", key=f"ep_up_{ev['turn_id']}"):
                        ev["rating"] = "positive"; st.rerun()
                with rc2:
                    if st.button("👎", key=f"ep_dn_{ev['turn_id']}"):
                        ev["rating"] = "negative"; st.rerun()
        st.divider()

    st.markdown("#### 📋 Full Evaluation Log")
    if eval_log:
        log_cols = st.columns([3, 1, 1, 1, 2])
        for col, hdr in zip(log_cols, ["**Question**", "**Rating**", "**Conf %**", "**Mode**", "**Timestamp**"]):
            col.markdown(hdr)
        st.markdown("<hr style='margin:4px 0;border-color:#1f2d3d;'>", unsafe_allow_html=True)

        for ev in reversed(eval_log):
            icon = "👍" if ev["rating"] == "positive" else ("👎" if ev["rating"] == "negative" else "⏳")
            cols = st.columns([3, 1, 1, 1, 2])
            q    = ev["question"]
            cols[0].markdown(f"<span style='font-size:.78rem;color:#dce8f0'>{q[:55]}{'…' if len(q)>55 else ''}</span>", unsafe_allow_html=True)
            cols[1].markdown(f"<span style='font-size:.82rem'>{icon}</span>", unsafe_allow_html=True)
            cols[2].markdown(f"<span style='font-size:.78rem;color:#7a8fa0'>{ev['confidence'] or '—'}</span>", unsafe_allow_html=True)
            cols[3].markdown(f"<span style='font-size:.78rem;color:#7a8fa0'>{ev['mode']}</span>", unsafe_allow_html=True)
            cols[4].markdown(f"<span style='font-size:.72rem;color:#5a6a7a'>{ev['timestamp']}</span>", unsafe_allow_html=True)

        st.divider()
        dl1, dl2 = st.columns([2, 1])
        with dl1:
            st.download_button(
                "⬇ Export Evaluation CSV",
                data=export_eval_csv(),
                file_name=f"ragnify_eval_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv", width='stretch',
            )
        with dl2:
            if st.button("🗑 Clear Log", width='stretch', key="clear_eval"):
                st.session_state["eval_log"]         = []
                st.session_state["pending_eval_tid"] = None
                st.rerun()
    else:
        st.info("No evaluations yet. Start chatting and rate responses to build the log.")