import streamlit as st
import os
import json
from datetime import datetime
import asyncio

from agents import RetrieverAgent, DatabaseAgent
from memory_enhanced import EnhancedMemoryAgent
from tools import CacheAgent
from langgraph_workflow import run_security_assistant_streaming
from config import load_config

APP_CONFIG = load_config()

# =========================================================
# Page Configuration
# =========================================================
st.set_page_config(
    page_title="LangGraph Mobile Security Assistant",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =========================================================
# Professional CSS
# =========================================================
st.markdown("""
<style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }

    .hero {
        background: linear-gradient(135deg, #0F5132, #198754, #2E7D32);
        padding: 2rem;
        border-radius: 24px;
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 24px rgba(0,0,0,0.15);
    }

    .hero-title {
        font-size: 2.4rem;
        font-weight: 800;
        margin-bottom: 0.4rem;
    }

    .hero-subtitle {
        font-size: 1.05rem;
        opacity: 0.95;
    }

    .card {
        background: white;
        padding: 1.1rem;
        border-radius: 20px;
        box-shadow: 0 4px 18px rgba(0,0,0,0.08);
        border: 1px solid #eef0f2;
        margin-bottom: 1rem;
    }

    .status-card {
        background: white;
        padding: 1rem;
        border-radius: 18px;
        border-left: 6px solid #198754;
        box-shadow: 0 4px 18px rgba(0,0,0,0.07);
        margin-bottom: 1rem;
    }

    .danger-card {
        border-left: 6px solid #dc3545;
    }

    .warning-card {
        border-left: 6px solid #ffc107;
    }

    .metric-box {
        background: linear-gradient(135deg, #198754, #2E7D32);
        color: white;
        padding: 1rem;
        border-radius: 18px;
        text-align: center;
        box-shadow: 0 4px 14px rgba(0,0,0,0.12);
    }

    .metric-number {
        font-size: 1.8rem;
        font-weight: 800;
    }

    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }

    .agent-pill {
        display: inline-block;
        padding: 0.35rem 0.7rem;
        margin: 0.2rem;
        border-radius: 999px;
        background: #E8F5E9;
        color: #1B5E20;
        font-size: 0.85rem;
        font-weight: 600;
    }

    .small-muted {
        color: #6c757d;
        font-size: 0.9rem;
    }

    .stButton button {
        border-radius: 14px;
        padding: 0.55rem 1rem;
        font-weight: 700;
        border: none;
        transition: 0.2s ease;
    }

    .stButton button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }

    .stChatMessage {
        border-radius: 18px;
        padding: 0.5rem;
    }

    .stChatMessage, .stMarkdown, p, div {
        unicode-bidi: plaintext;
        text-align: start;
    }

    .streamlit-expanderHeader {
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

# =========================================================
# Session State
# =========================================================
defaults = {
    "messages": [],
    "memory_agent": None,
    "db_agent": None,
    "retriever_agent": None,
    "cache_agent": None,
    "is_ready": False,
    "response_model": APP_CONFIG.response_model,
    "enable_web_search": APP_CONFIG.enable_web_search,
    "enable_cache": APP_CONFIG.enable_cache,
    "last_used_agents": [],
    "last_trace": [],
    "total_questions": 0,
    "cache_hits": 0,
    "cache_misses": 0,
    "rag_calls": 0,
    "blocked_queries": 0,
    "last_response_time": None,
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# Guard LLM scan is disabled in agents.py.
# We keep a fixed value only because langgraph_workflow.py still expects guard_model_name.
FIXED_GUARD_MODEL = APP_CONFIG.guard_model

# =========================================================
# Helper Functions
# =========================================================
def render_status_card(title, value, icon="✅", card_type="status-card"):
    st.markdown(f"""
    <div class="{card_type}">
        <h4>{icon} {title}</h4>
        <p style="margin-bottom:0;">{value}</p>
    </div>
    """, unsafe_allow_html=True)


def render_metric(label, value):
    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-number">{value}</div>
        <div class="metric-label">{label}</div>
    </div>
    """, unsafe_allow_html=True)


def render_agent_pills(agents):
    if not agents:
        st.markdown("<span class='small-muted'>No agent trace yet.</span>", unsafe_allow_html=True)
        return

    html = ""
    for agent in agents:
        html += f"<span class='agent-pill'>{agent}</span>"
    st.markdown(html, unsafe_allow_html=True)


def export_chat_json():
    return json.dumps(
        [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages],
        indent=2,
        ensure_ascii=False
    )


def add_user_prompt(prompt):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.total_questions += 1


def get_last_answer():
    for msg in reversed(st.session_state.messages):
        if msg["role"] == "assistant":
            return msg["content"]
    return ""


# =========================================================
# Header
# =========================================================
st.markdown("""
<div class="hero">
    <div class="hero-title">🛡️ LangGraph Multi-Agent Mobile Security Assistant</div>
    <div class="hero-subtitle">
        Professional AI security dashboard powered by LangGraph, RAG, enhanced memory, cache, and web search.
    </div>
</div>
""", unsafe_allow_html=True)

# =========================================================
# Top Metrics
# =========================================================
metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

with metric_col1:
    render_metric("System Status", "Ready" if st.session_state.is_ready else "Offline")

with metric_col2:
    render_metric("Questions", st.session_state.total_questions)

with metric_col3:
    render_metric("Cache", "ON" if st.session_state.enable_cache else "OFF")

with metric_col4:
    render_metric("Web Search", "ON" if st.session_state.enable_web_search else "OFF")

# =========================================================
# Main Tabs
# =========================================================
tab_chat, tab_system, tab_memory, tab_logs, tab_about = st.tabs([
    "💬 Chat",
    "⚙️ System Settings",
    "🧠 Memory & RAG",
    "📊 Agent Logs",
    "ℹ️ About"
])

# =========================================================
# Chat Tab
# =========================================================
with tab_chat:
    left_col, right_col = st.columns([2.4, 1], gap="large")

    with left_col:
        st.markdown("### 💬 Security Chat")

        chat_container = st.container(height=540)

        with chat_container:
            if not st.session_state.messages:
                st.info("Start by initializing the system, then ask a mobile security question.")
            else:
                for msg in st.session_state.messages:
                    avatar = "👤" if msg["role"] == "user" else "🛡️"
                    with st.chat_message(msg["role"], avatar=avatar):
                        st.markdown(msg["content"])

        st.markdown("#### ⚡ Quick Actions")
        q1, q2, q3, q4 = st.columns(4)

        quick_prompt = None

        with q1:
            if st.button("🔐 Secure Storage", use_container_width=True):
                quick_prompt = "Explain secure storage in mobile apps and mention Android Keystore and iOS Keychain."

        with q2:
            if st.button("🌐 Network Security", use_container_width=True):
                quick_prompt = "Explain mobile network security, TLS, and certificate pinning."

        with q3:
            if st.button("🧪 Test Vulnerability", use_container_width=True):
                quick_prompt = "How can I safely test insecure data storage in a mobile app?"

        with q4:
            if st.button("📚 MASVS Help", use_container_width=True):
                quick_prompt = "What does MASVS say about authentication and secure storage?"

        typed_prompt = st.chat_input("Ask your mobile security question...")

        prompt = quick_prompt or typed_prompt

        if prompt:
            if not st.session_state.is_ready:
                st.warning("Please initialize the LangGraph system first from the System Settings tab.")
            else:
                add_user_prompt(prompt)

                with chat_container:
                    with st.chat_message("user", avatar="👤"):
                        st.markdown(prompt)

                last_answer = get_last_answer()
                start_time = datetime.now()

                with chat_container:
                    with st.chat_message("assistant", avatar="🛡️"):
                        placeholder = st.empty()
                        placeholder.markdown("🔄 **Thinking, routing, and gathering security context...**")

                        full_response = []
                        trace_agents = []

                        async def token_generator():
                            async for event in run_security_assistant_streaming(
                                user_query=prompt,
                                user_id="default",
                                last_answer=last_answer,
                                chat_history=st.session_state.messages,
                                memory_agent=st.session_state.memory_agent,
                                db_agent=st.session_state.db_agent,
                                retriever_agent=st.session_state.retriever_agent,
                                guard_model_name=FIXED_GUARD_MODEL,
                                response_model_name=st.session_state.response_model,
                                cache_agent=st.session_state.cache_agent if st.session_state.enable_cache else None,
                                enable_web_search=st.session_state.enable_web_search
                            ):
                                if event["type"] == "chunk":
                                    cleaned = event["content"].replace("SAFE", "").replace("MALICIOUS", "")
                                    placeholder.empty()
                                    full_response.append(cleaned)
                                    yield cleaned

                                elif event["type"] == "error":
                                    placeholder.empty()
                                    st.session_state.blocked_queries += 1
                                    yield f"🚫 {event['content']}"
                                    break

                                elif event["type"] == "done":
                                    agents = event.get("agents", [])
                                    trace_agents.extend(agents)

                        st.write_stream(token_generator())
                        final_response = "".join(full_response)

                end_time = datetime.now()
                st.session_state.last_response_time = round((end_time - start_time).total_seconds(), 2)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": final_response
                })

                st.session_state.last_used_agents = trace_agents
                st.session_state.last_trace = trace_agents

                # ============= FINAL METRICS COUNTERS START =============
                is_blocked = any("GuardAgent(blocked)" in a for a in trace_agents)
                is_cache_hit = any("CacheAgent(hit)" in a for a in trace_agents)
                is_cache_skipped = any("CacheAgent(skipped" in a for a in trace_agents)

                if is_blocked:
                    st.session_state.blocked_queries += 1

                elif st.session_state.enable_cache:
                    if is_cache_hit:
                        st.session_state.cache_hits += 1
                    elif not is_cache_skipped:
                        # لو مش hit ومش follow-up → يبقى miss
                        st.session_state.cache_misses += 1
                # ============= FINAL METRICS COUNTERS END =============

                if any(x in a for a in trace_agents for x in [
                    "RetrieverAgent",
                    "DatabaseAgent",
                    "ResponseAgent(RAG)"
                ]):
                    st.session_state.rag_calls += 1
                    

                st.rerun()

    with right_col:
        st.markdown("### 🧭 Live Panel")

        if st.session_state.is_ready:
            render_status_card("System", "LangGraph system is initialized and ready.", "✅")
        else:
            render_status_card("System", "System is not initialized yet.", "⚠️", "status-card warning-card")

        render_status_card(
            "Active Model",
            f"Response: `{st.session_state.response_model}`",
            "🤖"
        )

        render_status_card(
            "Security Filter",
            "Regex-based filtering before response generation.",
            "🛡️"
        )

        st.markdown("### 🔍 Last Agent Trace")
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        render_agent_pills(st.session_state.last_used_agents)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("### 📌 Session Actions")
        c1, c2 = st.columns(2)

        with c1:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.messages = []
                st.session_state.last_used_agents = []
                st.session_state.last_trace = []
                st.rerun()

        with c2:
            if st.button("🧹 Clear Cache", use_container_width=True, disabled=not st.session_state.enable_cache):
                if st.session_state.cache_agent:
                    st.session_state.cache_agent.clear()
                st.success("Cache cleared.")

        if st.session_state.messages:
            st.download_button(
                "📥 Export Chat JSON",
                data=export_chat_json(),
                file_name=f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )

# =========================================================
# System Settings Tab
# =========================================================
with tab_system:
    st.markdown("### ⚙️ System Configuration")

    settings_col, status_col = st.columns([1.4, 1], gap="large")

    with settings_col:
        st.markdown("#### 🤖 Model Settings")

        response_model = st.selectbox(
            "Response Model",
            options=["mistral:latest", "llama3:latest"],
            index=["mistral:latest", "llama3:latest"].index(st.session_state.response_model)
            if st.session_state.response_model in ["mistral:latest", "llama3:latest"] else 0
        )

        st.session_state.response_model = response_model

        st.markdown("#### 📚 RAG Settings")

        embeddings_model = APP_CONFIG.embeddings_model
        st.info(f"Embeddings model: `{embeddings_model}`")

        chunk_size = st.slider("Chunk Size", 300, 2000, APP_CONFIG.chunk_size, step=100)
        chunk_overlap = st.slider("Chunk Overlap", 0, 500, APP_CONFIG.chunk_overlap, step=50)

        data_folder = st.text_input("Data Folder", value=APP_CONFIG.data_folder)
        qdrant_path = st.text_input("Qdrant Path", value=APP_CONFIG.qdrant_path)
        collection_name = st.text_input("Collection Name", value=APP_CONFIG.collection_name)

        st.markdown("#### 🔧 Advanced Features")

        st.session_state.enable_web_search = st.toggle(
            "Enable Web Search",
            value=st.session_state.enable_web_search
        )

        if st.session_state.enable_web_search:
            tavily_key = st.text_input(
                "Tavily API Key",
                type="password",
                help="If empty, DuckDuckGo fallback will be used."
            )
            if tavily_key:
                os.environ["TAVILY_API_KEY"] = tavily_key

        st.session_state.enable_cache = st.toggle(
            "Enable Cache-Aware Generation",
            value=st.session_state.enable_cache
        )

        cache_ttl = APP_CONFIG.cache_ttl_minutes
        if st.session_state.enable_cache:
            cache_ttl = st.slider("Cache TTL in minutes", 5, 240, APP_CONFIG.cache_ttl_minutes, step=5)

        if st.button("🚀 Build / Load LangGraph System", type="primary", use_container_width=True):
            with st.spinner("Initializing agents, memory, database, RAG, and tools..."):
                try:
                    retriever = RetrieverAgent(
                        data_folder=data_folder,
                        qdrant_path=qdrant_path,
                        collection_name=collection_name,
                        embeddings_model=embeddings_model,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap
                    )

                    memory_agent = EnhancedMemoryAgent(
                        embeddings_model=APP_CONFIG.memory_embeddings_model,
                        memory_index_path=APP_CONFIG.memory_index_path,
                        compression_threshold=APP_CONFIG.compression_threshold,
                        summarization_model=response_model
                    )

                    db_agent = DatabaseAgent(masvs_file=APP_CONFIG.masvs_file)

                    cache_agent = None
                    if st.session_state.enable_cache:
                        cache_agent = CacheAgent(ttl_minutes=cache_ttl, max_size=APP_CONFIG.cache_max_size)

                    st.session_state.memory_agent = memory_agent
                    st.session_state.db_agent = db_agent
                    st.session_state.retriever_agent = retriever
                    st.session_state.cache_agent = cache_agent
                    st.session_state.is_ready = True

                    st.success("✅ System initialized successfully.")

                except Exception as e:
                    st.session_state.is_ready = False
                    st.error(f"Failed to initialize system: {e}")

    with status_col:
        st.markdown("#### 🟢 System Status")

        if st.session_state.is_ready:
            render_status_card("LangGraph", "Workflow compiled and ready.", "✅")
            render_status_card("Memory", "Enhanced FAISS memory is active.", "🧠")
            render_status_card("Retriever", "Qdrant RAG retriever is active.", "📚")
        else:
            render_status_card("LangGraph", "Not initialized.", "⚠️", "status-card warning-card")
            render_status_card("Memory", "Waiting for initialization.", "🧠", "status-card warning-card")
            render_status_card("Retriever", "Waiting for initialization.", "📚", "status-card warning-card")

        render_status_card(
            "Security",
            "Regex-based filtering is used before response generation.",
            "🛡️"
        )

        render_status_card(
            "Feature Flags",
            f"Cache: `{st.session_state.enable_cache}`<br>Web Search: `{st.session_state.enable_web_search}`",
            "🔧"
        )

        st.markdown("#### 🧩 Architecture")
        st.markdown("""
        <div class="card">
            <b>Pipeline:</b><br>
            Guard → Cache → Router → Optimizer → Memory → Database → Retriever → Web Search → Response
        </div>
        """, unsafe_allow_html=True)

# =========================================================
# Memory & RAG Tab
# =========================================================
with tab_memory:
    st.markdown("### 🧠 Memory & Retrieval Dashboard")

    mem_col1, mem_col2, mem_col3 = st.columns(3)

    with mem_col1:
        memory_count = 0
        if st.session_state.memory_agent and hasattr(st.session_state.memory_agent, "memory_items"):
            memory_count = len(st.session_state.memory_agent.memory_items)
        render_metric("Memory Items", memory_count)

    with mem_col2:
        render_metric("Knowledge Calls", st.session_state.rag_calls)

    with mem_col3:
        cache_size = 0
        if st.session_state.cache_agent and hasattr(st.session_state.cache_agent, "cache"):
            cache_size = len(st.session_state.cache_agent.cache)
        render_metric("Cache Items", cache_size)

    st.markdown("### 🧠 Memory Preview")

    if st.session_state.memory_agent and hasattr(st.session_state.memory_agent, "memory_items"):
        items = st.session_state.memory_agent.memory_items[-5:]

        if items:
            for item in reversed(items):
                with st.expander(f"Memory: {item.query[:80]}"):
                    st.markdown(f"**Query:** {item.query}")
                    st.markdown(f"**Response:** {item.response[:600]}...")
                    st.markdown(f"**Importance:** `{item.importance}`")
                    st.markdown(f"**Access Count:** `{item.access_count}`")
                    st.markdown(f"**Timestamp:** `{item.timestamp}`")
        else:
            st.info("No memory items yet.")
    else:
        st.info("Initialize the system to activate memory.")

    st.markdown("### 📚 RAG Controls")

    rag_col1, rag_col2 = st.columns(2)

    with rag_col1:
        if st.button("🔁 Rebuild RAG Index", use_container_width=True, disabled=not st.session_state.is_ready):
            try:
                count = st.session_state.retriever_agent.rebuild_index()
                st.success(f"RAG index rebuilt successfully. Indexed chunks: {count}")
            except Exception as e:
                st.error(f"Failed to rebuild index: {e}")

    with rag_col2:
        if st.button("🗑️ Clear Conversation Memory", use_container_width=True, disabled=not st.session_state.is_ready):
            try:
                if st.session_state.memory_agent:
                    st.session_state.memory_agent.memory_items = []
                    st.session_state.memory_agent._rebuild_index()
                    st.session_state.memory_agent._save()
                st.success("Memory cleared.")
            except Exception as e:
                st.error(f"Failed to clear memory: {e}")

# =========================================================
# Agent Logs Tab
# =========================================================
with tab_logs:
    st.markdown("### 📊 Agent Logs & Analytics")

    log_col1, log_col2, log_col3, log_col4 = st.columns(4)

    with log_col1:
        render_metric("Cache Hits", st.session_state.cache_hits)

    with log_col2:
        render_metric("Cache Misses", st.session_state.cache_misses)

    with log_col3:
        render_metric("Blocked Queries", st.session_state.blocked_queries)

    with log_col4:
        render_metric("Last Response", f"{st.session_state.last_response_time or 0}s")

    st.markdown("### 🔍 Last Execution Trace")
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    render_agent_pills(st.session_state.last_trace)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("### 🧾 Chat Transcript")

    if st.session_state.messages:
        for idx, msg in enumerate(st.session_state.messages, start=1):
            role = "User" if msg["role"] == "user" else "Assistant"
            with st.expander(f"{idx}. {role}"):
                st.markdown(msg["content"])
    else:
        st.info("No logs yet.")

# =========================================================
# About Tab
# =========================================================
with tab_about:
    st.markdown("### ℹ️ About This System")

    st.markdown("""
    <div class="card">
        <h4>🛡️ Project Domain</h4>
        <p>
        This application is a multi-agent mobile security assistant. It helps answer questions about
        mobile application security, MASVS concepts, secure storage, network protection, authentication,
        and safe testing practices.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
        <h4>🤖 Agents</h4>
        <p>
        The system uses specialized agents for regex-based guarding, routing, memory retrieval,
        database lookup, RAG retrieval, web search, caching, and response generation.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
        <h4>⭐ Professional Features</h4>
        <p>
        Interactive dashboard, quick actions, live agent trace, memory preview, RAG controls,
        analytics, exportable chat logs, and bilingual-friendly display.
        </p>
    </div>
    """, unsafe_allow_html=True)