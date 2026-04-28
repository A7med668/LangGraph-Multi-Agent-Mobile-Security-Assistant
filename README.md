<div align="center">

# 🛡️ LangGraph Multi-Agent Mobile Security Assistant

### Dark Professional AI Security Assistant powered by LangGraph, RAG, FAISS Memory, Cache, Web Search, and LangSmith Observability

<img src="https://img.shields.io/badge/AI-Agentic_System-0B1020?style=for-the-badge&logo=openai&logoColor=white"/>
<img src="https://img.shields.io/badge/LangGraph-Multi_Agent-22C55E?style=for-the-badge"/>
<img src="https://img.shields.io/badge/RAG-Qdrant-3B82F6?style=for-the-badge"/>
<img src="https://img.shields.io/badge/Memory-FAISS-8B5CF6?style=for-the-badge"/>
<img src="https://img.shields.io/badge/UI-Streamlit-FF4B4B?style=for-the-badge"/>
<img src="https://img.shields.io/badge/Tracing-LangSmith-FACC15?style=for-the-badge"/>

<br/>

> A production-style multi-agent AI assistant for mobile application security, MASVS reasoning, secure storage guidance, network security, authentication, and safe testing workflows.

</div>

---

## ✨ Overview

**LangGraph Multi-Agent Mobile Security Assistant** is an advanced agentic AI system designed for mobile security Q&A.

It combines:

- 🧠 **Advanced long-term memory**
- 📚 **Retrieval-Augmented Generation**
- ⚡ **Cache-Augmented Generation**
- 🛡️ **Prompt-injection and malicious query guard**
- 🌐 **Optional web search**
- 📊 **LangSmith tracing**
- 🎛️ **Professional Streamlit dashboard**
- 🌍 **Arabic + English support**

---

## 🧠 Core Idea

Instead of using a single chatbot pipeline, this project uses a real **multi-agent architecture**.

Each user query passes through specialized agents:

```mermaid
flowchart TD
    A["👤 User Query"] --> B["🛡️ Guard Agent"]

    B -->|"Safe"| C["⚡ Cache Agent"]
    B -->|"Blocked"| X["🚫 Blocked Response"]

    C -->|"Cache Hit"| Y["⚡ Cached Answer"]
    C -->|"Cache Miss"| D["🧭 Router Agent"]

    D -->|"Greeting / Simple Query"| E["💬 Direct Response Agent"]
    D -->|"Follow-up Request"| F["🔁 Clarify / Translate / Summarize Agent"]
    D -->|"Security Question"| G["🔎 Query Optimizer"]

    G --> H["🧠 Memory Retrieval"]
    H --> I["📘 MASVS Database Agent"]
    I --> J{"Enough Context?"}

    J -->|"Yes"| K["🌐 Optional Web Search"]
    J -->|"No"| L["📚 Qdrant RAG Retriever"]

    L --> K
    K --> M["🤖 Final RAG Response Agent"]
    M --> N["✅ Final Answer"]

    style A fill:#0f172a,stroke:#38bdf8,color:#ffffff
    style B fill:#1e293b,stroke:#facc15,color:#ffffff
    style C fill:#172554,stroke:#60a5fa,color:#ffffff
    style D fill:#312e81,stroke:#a78bfa,color:#ffffff
    style H fill:#064e3b,stroke:#34d399,color:#ffffff
    style I fill:#422006,stroke:#fbbf24,color:#ffffff
    style L fill:#1e3a8a,stroke:#93c5fd,color:#ffffff
    style M fill:#581c87,stroke:#c084fc,color:#ffffff
    style N fill:#14532d,stroke:#22c55e,color:#ffffff
    style X fill:#7f1d1d,stroke:#f87171,color:#ffffff
    style Y fill:#164e63,stroke:#67e8f9,color:#ffffff
```

---

## 🔄 End-to-End Execution Flow

```mermaid
sequenceDiagram
    autonumber

    participant U as 👤 User
    participant UI as 🎛️ Streamlit UI
    participant G as 🛡️ Guard
    participant C as ⚡ Cache
    participant R as 🧭 Router
    participant M as 🧠 Memory
    participant DB as 📘 MASVS DB
    participant V as 📚 Qdrant RAG
    participant W as 🌐 Web Search
    participant LLM as 🤖 Ollama LLM
    participant LS as 📊 LangSmith

    U->>UI: Ask mobile security question
    UI->>LS: Start trace
    UI->>G: Validate input

    alt Malicious query
        G-->>UI: Block request
        UI-->>U: Safe refusal
    else Safe query
        G->>C: Check cached response

        alt Cache hit
            C-->>UI: Return cached answer
            UI-->>U: Fast response
        else Cache miss
            C->>R: Continue routing
            R->>M: Retrieve memory context
            M->>DB: Query MASVS knowledge
            DB->>V: Retrieve relevant docs
            V->>W: Optional live search
            W->>LLM: Build grounded prompt
            LLM-->>UI: Stream final answer
            UI-->>U: Display answer
            UI->>LS: Save trace metadata
        end
    end
```

---

## 🧩 System Components

| Component | Role |
|---|---|
| 🛡️ **Guard Agent** | Detects malicious or unsafe queries before they reach the LLM |
| ⚡ **Cache Agent** | Returns repeated answers instantly without rerunning the full pipeline |
| 🧭 **Router Agent** | Routes each query to direct, follow-up, or RAG workflow |
| 🔎 **Query Optimizer** | Prepares search-ready queries for retrieval |
| 🧠 **Enhanced Memory Agent** | Stores and retrieves long-term semantic memory with FAISS |
| 📘 **Database Agent** | Looks up MASVS security requirements |
| 📚 **Retriever Agent** | Uses Qdrant for document-based RAG |
| 🌐 **Web Search Tool** | Uses Tavily or DuckDuckGo fallback for live information |
| 🤖 **Response Agent** | Generates final grounded answers |
| 📊 **LangSmith** | Provides tracing, debugging, and observability |

---

## 🧠 Advanced Memory Architecture

The memory module is not simple chat history.

It supports:

- Semantic embeddings
- FAISS vector indexing
- Similarity retrieval
- Importance scoring
- Recency-aware ranking
- Access-count tracking
- Long-term memory compression

```mermaid
flowchart LR
    A["User Interaction"] --> B["Encode Query + Answer"]
    B --> C["SentenceTransformer Embedding"]
    C --> D["FAISS Vector Index"]
    D --> E["Similarity Search"]
    E --> F["Score by Similarity + Importance + Recency"]
    F --> G["Inject Relevant Memory into Prompt"]

    style A fill:#111827,stroke:#38bdf8,color:#ffffff
    style B fill:#1f2937,stroke:#818cf8,color:#ffffff
    style C fill:#312e81,stroke:#a78bfa,color:#ffffff
    style D fill:#064e3b,stroke:#34d399,color:#ffffff
    style E fill:#713f12,stroke:#facc15,color:#ffffff
    style F fill:#7c2d12,stroke:#fb923c,color:#ffffff
    style G fill:#14532d,stroke:#22c55e,color:#ffffff
```

---

## ⚡ Cache-Aware Generation

Repeated queries are served instantly using a cache layer.

```mermaid
flowchart TD
    A["Incoming Query"] --> B["Hash Query"]
    B --> C{"Exists in Cache?"}

    C -->|"Yes"| D["Return Cached Answer"]
    C -->|"No"| E["Run Full Agent Pipeline"]
    E --> F["Generate Answer"]
    F --> G["Store in Cache"]
    G --> H["Return Answer"]

    style A fill:#0f172a,stroke:#38bdf8,color:#ffffff
    style C fill:#422006,stroke:#facc15,color:#ffffff
    style D fill:#164e63,stroke:#67e8f9,color:#ffffff
    style E fill:#312e81,stroke:#a78bfa,color:#ffffff
    style G fill:#064e3b,stroke:#34d399,color:#ffffff
```

---

## 🛡️ Security Guard Flow

```mermaid
flowchart TD
    A["User Input"] --> B["Regex Scan"]
    B --> C{"Dangerous Pattern?"}

    C -->|"Yes"| D["Block Query"]
    C -->|"No"| E["Allow Pipeline"]

    E --> F["Cache / Router / RAG"]
    D --> G["Safe Refusal"]

    style A fill:#111827,stroke:#38bdf8,color:#ffffff
    style B fill:#1e293b,stroke:#facc15,color:#ffffff
    style C fill:#7c2d12,stroke:#fb923c,color:#ffffff
    style D fill:#7f1d1d,stroke:#f87171,color:#ffffff
    style E fill:#14532d,stroke:#22c55e,color:#ffffff
    style G fill:#450a0a,stroke:#ef4444,color:#ffffff
```

---

## 📊 Observability with LangSmith

LangSmith tracing is supported for debugging every graph execution.

You can inspect:

- Agent execution order
- Routing decisions
- Cache hits
- LLM latency
- Prompt inputs
- Final outputs
- Runtime metadata

```mermaid
flowchart LR
    A["Streamlit App"] --> B["LangGraph Runtime"]
    B --> C["RunnableConfig"]
    C --> D["Tags + Metadata"]
    D --> E["LangSmith Trace"]
    E --> F["Debug Runs"]
    E --> G["Latency Analysis"]
    E --> H["Agent Graph View"]

    style A fill:#0f172a,stroke:#38bdf8,color:#ffffff
    style B fill:#312e81,stroke:#a78bfa,color:#ffffff
    style C fill:#1e293b,stroke:#facc15,color:#ffffff
    style E fill:#064e3b,stroke:#34d399,color:#ffffff
    style F fill:#14532d,stroke:#22c55e,color:#ffffff
    style G fill:#422006,stroke:#fbbf24,color:#ffffff
    style H fill:#581c87,stroke:#c084fc,color:#ffffff
```

---

## 🎛️ Streamlit Dashboard

The interface includes:

- 💬 Security chat
- ⚙️ System settings
- 🧠 Memory and RAG dashboard
- 📊 Agent logs
- 🔍 Last execution trace
- 📥 Export chat as JSON
- 🌐 Web search toggle
- ⚡ Cache toggle
- 📊 LangSmith toggle

---

## 🧪 Example Questions

```text
What does MASVS say about secure storage?
```

```text
Explain mobile network security, TLS, and certificate pinning.
```

```text
My app stores JWT tokens locally. Remember this context.
```

```text
What should I improve in my app?
```

```text
اشرح secure storage في تطبيقات الموبايل
```

```text
Translate the previous answer to English.
```

---

## 📁 Project Structure

```bash
.
├── app.py                         # Streamlit dashboard
├── agents.py                      # Guard, database, retriever, routing logic
├── langgraph_workflow.py          # LangGraph orchestration
├── memory_enhanced.py             # Advanced FAISS memory
├── tools.py                       # Web search and cache tools
├── config.py                      # Environment-based config
├── masvs.json                     # MASVS knowledge base
├── requirements.txt               # Dependencies
├── .env.example                   # Environment template
└── README.md
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/A7med668/LangGraph-Multi-Agent-Mobile-Security-Assistant.git
cd LangGraph-Multi-Agent-Mobile-Security-Assistant
```

### 2. Create virtual environment

```bash
python -m venv llm_env
```

### 3. Activate environment

#### Windows

```bash
llm_env\Scripts\activate
```

#### macOS / Linux

```bash
source llm_env/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🦙 Ollama Setup

Start Ollama:

```bash
ollama serve
```

Pull required models:

```bash
ollama pull mistral:latest
ollama pull llama3:latest
ollama pull nomic-embed-text:latest
```

---

## 🔐 Environment Setup

Copy the environment template:

```bash
copy .env.example .env
```

On macOS / Linux:

```bash
cp .env.example .env
```

Edit `.env`:

```env
RESPONSE_MODEL=mistral:latest
GUARD_MODEL=llama3:latest
EMBEDDINGS_MODEL=nomic-embed-text:latest
MEMORY_EMBEDDINGS_MODEL=paraphrase-multilingual-MiniLM-L12-v2

DATA_FOLDER=./owasp_rag_data
QDRANT_PATH=./qdrant_local
QDRANT_COLLECTION=security_assistant
MASVS_FILE=./masvs.json
MEMORY_INDEX_PATH=./faiss_memory_enhanced

ENABLE_CACHE=true
ENABLE_WEB_SEARCH=false

ENABLE_LANGSMITH=false
LANGSMITH_API_KEY=
LANGSMITH_PROJECT=mobile-security-assistant

TAVILY_API_KEY=
```

---

## ▶️ Run the App

```bash
streamlit run app.py
```

Then open:

```text
http://localhost:8501
```

---

## 🌐 Optional Web Search

To enable live web search:

```env
ENABLE_WEB_SEARCH=true
TAVILY_API_KEY=your_tavily_key
```

If no Tavily key is configured, the system can fall back to DuckDuckGo when available.

---

## 📊 Optional LangSmith Tracing

To enable LangSmith:

```env
ENABLE_LANGSMITH=true
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your_langsmith_key
LANGSMITH_PROJECT=mobile-security-assistant

LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_key
LANGCHAIN_PROJECT=mobile-security-assistant
```

Then restart Streamlit.

---

## 🧪 Testing Checklist

| Test | Expected Result |
|---|---|
| Ask `hello` | Direct response |
| Ask same query twice | Cache hit |
| Ask MASVS secure storage question | Database / RAG response |
| Ask Arabic question | Arabic response |
| Ask follow-up translation | Uses previous answer |
| Ask unsafe hacking query | Guard blocks it |
| Enable LangSmith | Trace appears in LangSmith dashboard |

---

## 🏆 Project Grading Alignment

| Requirement | Implementation |
|---|---|
| Multi-Agent System | LangGraph agents with conditional routing |
| Advanced Memory | FAISS semantic memory with scoring and compression |
| Tool Integration | RAG, CAG, MASVS DB, optional web search |
| User Interface | Streamlit dashboard |
| Bonus: Multilingual | Arabic + English |
| Bonus: Prompt Injection Detection | Guard Agent |
| Bonus: Observability | LangSmith tracing |

---

## 🚀 Tech Stack

| Layer | Technology |
|---|---|
| UI | Streamlit |
| Orchestration | LangGraph |
| LLM Runtime | Ollama |
| LLM Interface | LangChain |
| Vector Database | Qdrant |
| Memory Index | FAISS |
| Embeddings | SentenceTransformers + Ollama Embeddings |
| Web Search | Tavily / DuckDuckGo |
| Observability | LangSmith |
| Language | Python |

---

## 🔮 Future Improvements

- Docker deployment
- Authentication system
- Evaluation dashboard
- Per-agent latency analytics
- More MASVS datasets
- PDF upload from UI
- Multimodal mobile app screenshot analysis
- Agent decision explanations

---

## 👨‍💻 Author

**Ahmed Hussein**  
Faculty of Artificial Intelligence

---

## 📄 License

Academic Use Only

---

<div align="center">

### ⭐ If you like this project, give it a star.

**Built with LangGraph, Streamlit, FAISS, Qdrant, Ollama, and LangSmith.**

</div>