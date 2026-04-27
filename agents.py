# agents.py
import os
import re
import json
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.document_loaders import TextLoader, UnstructuredMarkdownLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from qdrant_client.http.exceptions import UnexpectedResponse

# =========================
# 1. Guard Agent (Regex + LLM)
# =========================
class GuardAgent:
    def __init__(self, llm: Optional[ChatOllama] = None, guard_model_name: str = "llama3:latest"):
        self.llm = llm if llm else ChatOllama(model=guard_model_name, temperature=0.0)
        self.blocked_patterns = [
            r"act\s+as\s+hacker", r"pretend\s+to\s+be\s+hacker", r"role\s+play\s+hacker",
            r"unethical\s+hacker", r"unethical\s+hacking", r"how\s+to\s+hack",
            r"bypass\s+security", r"exploit\s+vulnerability", r"inject\s+malicious",
            r"steal\s+credentials", r"phishing\s+attack", r"jailbreak\s+detection\s+bypass",
            r"modify\s+app\s+binary", r"repack\s+certificate", r"url\s+scheme\s+hijacking",
            r"session\s+hijacking", r"remote\s+code\s+execution", r"gain\s+control",
            r"steal\s+sensitive\s+data", r"distribute\s+malicious\s+app", r"inject\s+spyware",
            r"spyware", r"remote\s+access", r"device\s+takeover",
        ]
        self.dangerous_keywords = ["hack", "exploit", "bypass", "malicious", "steal", "phishing"]
        self.action_phrases = ["how to", "show me", "teach me", "explain how", "method to", "way to", "act as"]

    def _regex_scan(self, user_query: str) -> Tuple[bool, str]:
        query_lower = user_query.lower()
        for pattern in self.blocked_patterns:
            if re.search(pattern, query_lower):
                return True, f"Blocked by regex: '{pattern}'"
        for keyword in self.dangerous_keywords:
            if keyword in query_lower and any(phrase in query_lower for phrase in self.action_phrases):
                return True, f"Blocked by keyword+action: '{keyword}'"
        return False, ""

    async def _llm_scan(self, user_query: str) -> Tuple[bool, str]:
        prompt = f"""Classify the following user query as either 'SAFE' or 'MALICIOUS' based on whether it asks for harmful actions, hacking techniques, bypassing security, or unethical activities. Only output one word: SAFE or MALICIOUS.

Query: {user_query}

Classification:"""
        try:
            response = await self.llm.ainvoke(prompt)
            result = response.content.strip().upper()
            if "MALICIOUS" in result:
                return True, "Blocked by LLM analysis: malicious intent detected"
            return False, ""
        except Exception as e:
            print(f"LLM scan failed: {e}")
            return False, ""

    async def scan(self, user_query: str) -> Tuple[bool, str]:
        is_mal, reason = self._regex_scan(user_query)
        if is_mal:
            return True, reason
        
        # Disabled LLM scan to eliminate the ~2-3s delay before response generation.
        # If you need LLM-based security scanning, uncomment the lines below, 
        # but note it will introduce latency before the first token streams.
        # is_mal, reason = await self._llm_scan(user_query)
        # if is_mal:
        #     return True, reason
            
        return False, "Query is safe"

# =========================
# 2. Memory Agent (FAISS)
# =========================
class MemoryAgent:
    def __init__(self, embeddings_model: str = "nomic-embed-text:latest", memory_index_path: str = "faiss_memory"):
        self.embeddings_model = embeddings_model
        self.memory_index_path = memory_index_path
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.dimension = 384
        self.index = None
        self.memory_store = []
        self._load_or_create_index()

    def _load_or_create_index(self):
        if os.path.exists(self.memory_index_path + ".faiss"):
            self.index = faiss.read_index(self.memory_index_path + ".faiss")
            with open(self.memory_index_path + ".json", "r", encoding="utf-8") as f:
                self.memory_store = json.load(f)
        else:
            self.index = faiss.IndexFlatL2(self.dimension)
            self.memory_store = []

    def _save(self):
        faiss.write_index(self.index, self.memory_index_path + ".faiss")
        with open(self.memory_index_path + ".json", "w", encoding="utf-8") as f:
            json.dump(self.memory_store, f, ensure_ascii=False, indent=2)

    def _embed(self, text: str) -> np.ndarray:
        return self.encoder.encode([text])[0].astype(np.float32)

    def retrieve_similar(self, query: str, k: int = 2) -> str:
        if len(self.memory_store) == 0 or self.index.ntotal == 0:
            return ""
        query_vec = self._embed(query).reshape(1, -1)
        distances, indices = self.index.search(query_vec, k)
        contexts = []
        for idx in indices[0]:
            if idx != -1 and idx < len(self.memory_store):
                item = self.memory_store[idx]
                contexts.append(f"Previous Q: {item['query']}\nPrevious A: {item['response']}")
        return "\n\n".join(contexts)

    def add_interaction(self, user_query: str, assistant_response: str, metadata: dict):
        text = f"Q: {user_query}\nA: {assistant_response}"
        vec = self._embed(text).reshape(1, -1)
        self.index.add(vec)
        self.memory_store.append({
            "query": user_query,
            "response": assistant_response,
            "metadata": metadata
        })
        self._save()

# =========================
# 3. Database Agent (MASVS JSON)
# =========================
class DatabaseAgent:
    def __init__(self, masvs_file: str = "masvs.json"):
        self.masvs_file = masvs_file
        self.data = self._load_masvs()

    def _load_masvs(self) -> Dict:
        default_data = {
            "MASVS-STORAGE-1": "Ensure that sensitive data is not stored insecurely. Use Android Keystore or iOS Keychain.",
            "MASVS-STORAGE-2": "Ensure that sensitive data is not exposed via IPC mechanisms.",
            "MASVS-AUTH-1": "Implement proper authentication and authorization checks.",
            "MASVS-NETWORK-1": "Ensure that all network communication is encrypted (TLS).",
            "MASVS-PLATFORM-1": "Ensure that the app uses up-to-date platform security features."
        }
        if os.path.exists(self.masvs_file):
            with open(self.masvs_file, "r", encoding="utf-8") as f:
                return json.load(f)
        else:
            with open(self.masvs_file, "w", encoding="utf-8") as f:
                json.dump(default_data, f, indent=2)
            return default_data

    def query(self, user_query: str) -> str:
        query_lower = user_query.lower()
        for key, value in self.data.items():
            if key.lower() in query_lower:
                return f"{key}: {value}"
        keywords = ["storage", "authentication", "network", "platform", "cryptography"]
        for kw in keywords:
            if kw in query_lower:
                for k, v in self.data.items():
                    if kw in k.lower():
                        return f"{k}: {v}"
        return ""

# =========================
# 4. Retriever Agent (Qdrant مباشرة)
# =========================
class RetrieverAgent:
    def __init__(self, data_folder: str, qdrant_path: str, collection_name: str,
                 embeddings_model: str, chunk_size: int, chunk_overlap: int):
        self.data_folder = data_folder
        self.qdrant_path = qdrant_path
        self.collection_name = collection_name
        self.embeddings_model = embeddings_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.embeddings = OllamaEmbeddings(model=embeddings_model)
        os.makedirs(qdrant_path, exist_ok=True)
        self.client = QdrantClient(path=qdrant_path)

        self._setup_collection()
        self.indexed_count = 0
        self._load_and_index_documents()

    def _setup_collection(self):
        collections = self.client.get_collections().collections
        exists = any(c.name == self.collection_name for c in collections)
        if not exists:
            # نحتاج إلى معرفة أبعاد embedding
            test_embed = self.embeddings.embed_query("test")
            vector_size = len(test_embed)
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
            )
            print(f"[Qdrant] Created collection '{self.collection_name}' with dim={vector_size}")
        else:
            print(f"[Qdrant] Collection '{self.collection_name}' already exists.")

    def _load_and_index_documents(self):
        # التحقق إذا كانت المجموعة تحتوي على بيانات
        collection_info = self.client.get_collection(collection_name=self.collection_name)
        if collection_info.points_count > 0:
            print(f"[Qdrant] Collection already has {collection_info.points_count} vectors. Skipping indexing.")
            self.indexed_count = collection_info.points_count
            return

        if not os.path.exists(self.data_folder):
            os.makedirs(self.data_folder, exist_ok=True)
            print(f"[Qdrant] Created empty data folder: {self.data_folder}. Please add documents.")
            return

        # تحميل المستندات
        documents = []
        loaders = {
            "*.txt": TextLoader,
            "*.md": UnstructuredMarkdownLoader,
            "*.pdf": PyPDFLoader
        }
        for pattern, LoaderClass in loaders.items():
            for file_path in Path(self.data_folder).glob(pattern):
                try:
                    print(f"[Qdrant] Loading: {file_path}")
                    loader = LoaderClass(str(file_path))
                    docs = loader.load()
                    documents.extend(docs)
                    print(f"   -> Loaded {len(docs)} pages from {file_path.name}")
                except Exception as e:
                    print(f"   Error loading {file_path}: {e}")

        if not documents:
            print("[Qdrant] No documents found.")
            return

        # تقسيم النصوص
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = text_splitter.split_documents(documents)
        print(f"[Qdrant] Split into {len(chunks)} chunks.")

        # إضافة النقاط إلى Qdrant
        points = []
        for idx, doc in enumerate(chunks):
            vector = self.embeddings.embed_query(doc.page_content)
            point = PointStruct(
                id=idx,
                vector=vector,
                payload={"text": doc.page_content, "metadata": doc.metadata}
            )
            points.append(point)
            # نضيف على دفعات لتجنب الضغط
            if len(points) >= 100:
                self.client.upsert(collection_name=self.collection_name, points=points)
                points = []
        if points:
            self.client.upsert(collection_name=self.collection_name, points=points)

        self.indexed_count = len(chunks)
        print(f"[Qdrant] Indexed {len(chunks)} chunks successfully.")

    def retrieve(self, query: str, k: int = 5) -> str:
        query_vector = self.embeddings.embed_query(query)

        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=k,
            with_payload=True
             )

        if not results.points:
            return ""

        texts = []
        for hit in results.points:
            if hit.payload and "text" in hit.payload:
                texts.append(hit.payload["text"])

        return "\n\n".join(texts)

    def rebuild_index(self):
        """حذف المجموعة وإعادة بناء الفهرس."""
        try:
            self.client.delete_collection(collection_name=self.collection_name)
            print(f"[Qdrant] Deleted collection '{self.collection_name}'")
        except Exception as e:
            print(f"[Qdrant] Could not delete collection: {e}")
        self._setup_collection()
        self._load_and_index_documents()
        return self.indexed_count

# =========================
# 5. Helper functions
# =========================

# Clarification / follow-up markers in English and Arabic.
# These help the router understand that the user is referring to the previous answer,
# for example: translate it, explain it again, summarize it, or simplify it.
clarify_markers_en = [
    # direct confusion
    "i don't understand", "i dont understand", "i didn't understand", "i did not understand",
    "i'm confused", "im confused", "this is confusing", "that is confusing",

    # asking for explanation
    "can you explain", "could you explain", "please explain",
    "explain this", "explain that", "explain it", "explain more",

    # asking for more detail
    "tell me more", "more details", "more detail",
    "in detail", "in details", "explain in detail", "explain in details",

    # asking to repeat
    "explain again", "say that again", "repeat that", "can you repeat",

    # asking to simplify
    "simplify", "make it simpler", "explain simply", "simply explain",
    "explain in simple terms", "break it down", "break this down",

    # clarification phrases
    "what do you mean", "what does that mean", "what is that supposed to mean",
    "i don't get it", "i dont get it", "i still don't understand", "i still dont understand",

    # previous-answer operations
    "translate", "translate it", "translate this", "translate the previous answer",
    "summarize", "summarise", "summarize it", "summarize this", "summary",

    # short/implicit signals
    "more", "again", "huh"
]

clarify_markers_ar = [
    # عدم الفهم المباشر
    "مش فاهم", "مش فاهمة", "مش فاهمك", "مش فاهمة قصدك",
    "لم أفهم", "لم افهم", "ما فهمتش", "مفهمتش", "مش واضح",
    "غير واضح", "مش مفهوم", "مش مفهومة", "الكلام مش واضح",

    # طلب شرح
    "اشرح", "اشرح لي", "اشرحلي", "ممكن تشرح", "هل يمكنك الشرح",
    "وضح", "وضح لي", "وضحلي", "ممكن توضح",
    "اشرح ده", "اشرح هذا", "اشرح الكلام ده", "وضح الكلام ده",

    # طلب تفاصيل
    "تفاصيل", "تفصيل", "عايز تفاصيل", "عايز تفصيل", "اريد تفاصيل",
    "بالتفصيل", "اشرح بالتفصيل", "ممكن تفاصيل اكتر", "ممكن تفاصيل أكثر",
    "قوللي اكتر", "قول لي اكتر", "عايز اعرف اكتر", "عايز أعرف أكتر",

    # طلب إعادة
    "عيد", "أعد", "اعد", "كرر", "قول تاني", "قولها تاني",
    "ممكن تعيد", "ممكن تكرر", "اشرح تاني", "اشرح مرة تانية",

    # تبسيط
    "ببساطة", "بشكل بسيط", "اشرح ببساطة", "بسطها", "بسّطها",
    "بسط الموضوع", "بسّط الموضوع", "خليها بسيطة", "شرح مبسط",
    "قسمها", "جزئها", "جزّأها",

    # طلب توضيح المعنى
    "يعني ايه", "يعني إيه", "ايه المقصود", "ما المقصود",
    "ماذا تقصد", "مش فاهم قصدك", "مش فاهم النقطة",
    "ايه ده", "ده معناه ايه", "ما معنى ذلك",

    # استمرار عدم الفهم
    "لسه مش فاهم", "لسه مش فاهمة", "برضه مش فاهم", "ما زلت لا أفهم",

    # عمليات على الإجابة السابقة
    "ترجم", "ترجملي", "ترجم لي", "ترجم الاجابة", "ترجم الإجابة",
    "ترجم الاجابة الماضية", "ترجم الإجابة الماضية", "ترجم الرد", "ترجم الرد السابق",
    "لخص", "لخصلي", "لخص لي", "لخص الاجابة", "لخص الإجابة",
    "اختصر", "اختصرها", "ملخص", "الخلاصة",

    # إشارات قصيرة
    "؟", "ليه", "لماذا", "ازاي", "إزاي", "كيف"
]

clarify_markers = clarify_markers_en + clarify_markers_ar


def is_clarify(query: str) -> bool:
    """Return True when the user is likely referring to the previous answer."""
    q = query.lower().strip()

    # Very short follow-up signals. Keep these strict to avoid treating new questions
    # like "how to secure storage" as clarification requests.
    short_signals = {"?", "؟؟", "why", "ليه", "لماذا", "ازاي", "إزاي", "كيف", "more", "again", "تاني"}
    if q in short_signals or (len(q.split()) <= 2 and any(sig in q for sig in short_signals)):
        return True

    return any(marker.lower() in q for marker in clarify_markers)


def heuristic_route(user_query: str) -> str:
    query_lower = user_query.lower().strip()
    greetings = ["hello", "hi", "hey", "good morning", "good afternoon", "مرحبا", "أهلا", "اهلا", "السلام عليكم"]
    if any(g in query_lower for g in greetings):
        return "direct"

    if is_clarify(user_query):
        return "clarify_last"

    return "retrieve"

def optimize_search_query(user_query: str, chat_history: List[Dict], llm) -> str:
    return user_query


def detect_followup_intent(query: str) -> str:
    """
    Deterministic intent classifier for follow-up operations.
    Returns one of: TRANSLATE, SUMMARIZE, SIMPLIFY, EXPLAIN, REPEAT, CLARIFY, NONE.
    """
    q = query.lower().strip()

    translate_words = ["translate", "translate it", "translate this", "ترجم", "ترجملي", "ترجم لي"]
    summarize_words = ["summarize", "summarise", "summary", "لخص", "لخصلي", "لخص لي", "اختصر", "اختصرها", "ملخص", "الخلاصة"]
    simplify_words = ["simplify", "make it simpler", "simple terms", "بسط", "بسطها", "بسّط", "بسّطها", "ببساطة", "خليها بسيطة", "شرح مبسط"]
    explain_words = ["explain more", "explain again", "explain it", "اشرح", "اشرحلي", "اشرح لي", "وضح", "وضحلي", "وضح لي"]
    repeat_words = ["repeat", "say again", "again", "كرر", "عيد", "اعد", "أعد", "قول تاني", "قولها تاني"]

    if any(w in q for w in translate_words):
        return "TRANSLATE"
    if any(w in q for w in summarize_words):
        return "SUMMARIZE"
    if any(w in q for w in simplify_words):
        return "SIMPLIFY"
    if any(w in q for w in explain_words):
        return "EXPLAIN"
    if any(w in q for w in repeat_words):
        return "REPEAT"
    if is_clarify(query):
        return "CLARIFY"
    return "NONE"

