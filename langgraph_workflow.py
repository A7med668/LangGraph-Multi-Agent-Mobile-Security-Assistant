# langgraph_workflow.py
import operator
import re
import os
from typing import TypedDict, Annotated, List, Dict, Any, Optional, AsyncIterator, Tuple
from datetime import datetime

from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnableConfig
from langgraph.config import get_stream_writer

from agents import (
    GuardAgent, DatabaseAgent, RetrieverAgent,
    heuristic_route, optimize_search_query, detect_followup_intent
)
from memory_enhanced import EnhancedMemoryAgent
from tools import WebSearchTool, CacheAgent

# ------------------------------
# Helper: Remove SAFE/MALICIOUS prefix
# ------------------------------
def remove_safe_prefix(text: str) -> str:
    """إزالة كلمة SAFE أو MALICIOUS من بداية النص إذا وجدت."""
    cleaned = re.sub(r'^(SAFE|MALICIOUS)\s*[:;,\-]?\s*', '', text.strip(), flags=re.IGNORECASE)
    return cleaned if cleaned else text

# ------------------------------
# 1. Define the State (موسع)
# ------------------------------
class AgentState(TypedDict):
    user_query: str
    user_id: str
    last_answer: str
    chat_history: List[Dict[str, Any]]
    is_malicious: bool
    malicious_reason: Optional[str]
    route: str
    followup_intent: str
    search_query: str
    memory_context: str
    db_result: str
    retrieved_docs: str
    web_results: str
    final_answer: str
    used_agents: Annotated[List[str], operator.add]
    guard_model_name: str
    response_model_name: str
    memory_agent: Optional[EnhancedMemoryAgent]
    db_agent: Optional[DatabaseAgent]
    retriever_agent: Optional[RetrieverAgent]
    cache_agent: Optional[CacheAgent]
    use_web_search: bool

# ------------------------------
# 2. LLM Initialization
# ------------------------------
def get_llm(model_name: str, temperature: float = 0.2):
    return ChatOllama(model=model_name, temperature=temperature)


def has_arabic(text: str) -> bool:
    """Detect Arabic characters."""
    return bool(re.search(r"[\u0600-\u06FF]", text or ""))


def normalize_arabic_output(text: str) -> str:
    """
    Lightweight post-processing for common broken Arabic terms produced by small local LLMs.
    This is not a translator; it only fixes recurring bad terminology.
    """
    replacements = {
        "تحفيز": "تشفير",
        "تحفيز البيانات": "تشفير البيانات",
        "الغذاء الأمني": "الخوارزمية الأمنية",
        "الغذاء": "الخوارزمية",
        "الوصول الغير مصرحي": "الوصول غير المصرح به",
        "الوصول الغير مصرح": "الوصول غير المصرح",
        "المعلق والمنقول": "البيانات المخزنة وأثناء النقل",
        "البيانات السخيفة": "البيانات الحساسة",
        "تحقق من الصحة المعتمدة": "المصادقة متعددة العوامل",
        "أحكام جوال": "آليات آمنة",
    }
    cleaned = text
    for bad, good in replacements.items():
        cleaned = cleaned.replace(bad, good)
    return cleaned.strip()


def build_arabic_term_rules() -> str:
    return """
ARABIC QUALITY RULES - CRITICAL:
- Use clear, simple Modern Standard Arabic.
- Do NOT translate word-by-word.
- Explain the meaning, not the literal words.
- Avoid broken Arabic and unnatural phrases.
- Never use incorrect translations such as:
  - "تحفيز" for encryption
  - "الغذاء الأمني" for algorithm/security
  - "المعلق والمنقول" for at rest/in transit
  - "الوصول الغير مصرحي" for unauthorized access
- Use these preferred terms:
  - encryption = تشفير
  - ciphertext = نص مشفّر / بيانات مشفّرة
  - plaintext = نص عادي / بيانات عادية
  - security = أمان / حماية
  - secure = آمن
  - authentication = مصادقة
  - authorization = تفويض
  - key = مفتاح
  - key management = إدارة المفاتيح
  - algorithm = خوارزمية
  - unauthorized access = وصول غير مصرح به
  - data at rest = البيانات المخزنة
  - data in transit = البيانات أثناء النقل
  - data breach = تسريب بيانات
  - user privacy = خصوصية المستخدم
"""


def deterministic_followup_intent(user_request: str) -> str:
    """
    Code-level intent classification so clarify behavior is not left to the LLM.
    """
    try:
        intent = detect_followup_intent(user_request)
        if intent and intent != "NONE":
            return intent
    except Exception:
        pass

    q = (user_request or "").lower().strip()
    if any(w in q for w in ["translate", "ترجم"]):
        return "TRANSLATE"
    if any(w in q for w in ["summarize", "summarise", "summary", "لخص", "اختصر", "ملخص"]):
        return "SUMMARIZE"
    if any(w in q for w in ["simplify", "simple", "بسط", "بسّط", "ببساطة", "بسيط"]):
        return "SIMPLIFY"
    if any(w in q for w in ["explain", "اشرح", "وضح"]):
        return "EXPLAIN"
    if any(w in q for w in ["repeat", "again", "كرر", "عيد", "تاني"]):
        return "REPEAT"
    return "CLARIFY"


def build_followup_prompt(intent: str, user_request: str, previous_answer: str, memory_context: str) -> str:
    """
    One prompt per task. This gives more stable behavior than asking the LLM to both classify and execute.
    """
    arabic = has_arabic(user_request)
    arabic_rules = build_arabic_term_rules()

    if intent == "TRANSLATE":
        target = "Arabic" if arabic else "the same language requested by the user"
        return f"""
You are a professional mobile-security translator.

TASK: TRANSLATE

Translate ONLY the previous answer to {target}.
Do not add greetings, notes, explanations, examples, or new facts.
Preserve meaning exactly.

{arabic_rules if arabic else ""}

PREVIOUS ANSWER:
{previous_answer}

TRANSLATION:
"""

    if intent == "SUMMARIZE":
        if arabic:
            return f"""
أنت مساعد متخصص في أمن تطبيقات الموبايل.

المهمة: لخص النص السابق باللغة العربية فقط.

القواعد:
- اكتب 3 إلى 5 سطور قصيرة فقط.
- استخدم عربية طبيعية وواضحة.
- لا تكرر النص بالكامل.
- لا تضف معلومات جديدة.
- لا تستخدم ترجمة حرفية.

{arabic_rules}

النص السابق:
{previous_answer}

الملخص:
"""
        return f"""
You are a mobile-security assistant.

TASK: SUMMARIZE

Summarize the previous answer in English only.
Rules:
- 3-5 short lines maximum.
- Do not copy the full text.
- Do not add new facts.

PREVIOUS ANSWER:
{previous_answer}

SUMMARY:
"""

    if intent == "SIMPLIFY":
        if arabic:
            return f"""
أنت مساعد يشرح أمن تطبيقات الموبايل للمبتدئين.

المهمة: بسّط النص السابق باللغة العربية فقط.

القواعد:
- استخدم كلمات بسيطة جدًا.
- اجعل الجمل قصيرة.
- اشرح كأنك تتكلم مع شخص غير متخصص.
- لا تضف معلومات جديدة.
- لا تستخدم ترجمة حرفية.
- لا تستخدم مصطلحات غريبة أو عربية مكسّرة.

{arabic_rules}

النص السابق:
{previous_answer}

الشرح المبسط:
"""
        return f"""
You are explaining mobile security to a beginner.

TASK: SIMPLIFY

Simplify the previous answer in English.
Rules:
- Use very simple words.
- Short sentences.
- Avoid jargon where possible.
- Do not add new facts.

PREVIOUS ANSWER:
{previous_answer}

SIMPLIFIED ANSWER:
"""

    if intent == "EXPLAIN":
        if arabic:
            return f"""
أنت مهندس أمن تطبيقات موبايل.

المهمة: اشرح النص السابق بشكل أوضح باللغة العربية.

القواعد:
- اشرح المعنى بدون تغيير المحتوى.
- استخدم نقاط منظمة.
- لا تضف معلومات غير موجودة إلا إذا كانت توضيحًا مباشرًا.
- استخدم عربية طبيعية.

{arabic_rules}

سياق اختياري من الذاكرة:
{memory_context}

النص السابق:
{previous_answer}

الشرح:
"""
        return f"""
You are a senior mobile-security engineer.

TASK: EXPLAIN

Explain the previous answer more clearly in English.
Use structure and examples only when they clarify the same meaning.
Do not change the topic.

OPTIONAL MEMORY CONTEXT:
{memory_context}

PREVIOUS ANSWER:
{previous_answer}

EXPLANATION:
"""

    if intent == "REPEAT":
        if arabic:
            return f"""
أعد صياغة النص السابق بالعربية بوضوح، بدون إضافة معلومات جديدة.

{arabic_rules}

النص السابق:
{previous_answer}

الإجابة:
"""
        return f"""
Restate the previous answer clearly in English.
Do not add new facts.

PREVIOUS ANSWER:
{previous_answer}

RESTATED ANSWER:
"""

    # CLARIFY fallback
    if arabic:
        return f"""
أنت مساعد متخصص في أمن تطبيقات الموبايل.

المهمة: وضّح النص السابق باللغة العربية، مع إزالة أي غموض.

القواعد:
- استخدم عربية واضحة وطبيعية.
- لا تضف معلومات غير مرتبطة.
- لا تكرر النص بالكامل.

{arabic_rules}

سياق اختياري:
{memory_context}

النص السابق:
{previous_answer}

التوضيح:
"""
    return f"""
You are a mobile-security assistant.

TASK: CLARIFY

Clarify the previous answer.
Do not change the topic and do not add unrelated facts.

OPTIONAL MEMORY CONTEXT:
{memory_context}

PREVIOUS ANSWER:
{previous_answer}

CLARIFICATION:
"""


# ------------------------------
# 3. Node Functions
# ------------------------------
async def guard_node(state: AgentState) -> dict:
    guard_llm = get_llm(state["guard_model_name"], temperature=0.0)
    guard = GuardAgent(llm=guard_llm)
    is_malicious, reason = await guard.scan(state["user_query"])
    if is_malicious:
        return {
            "is_malicious": True,
            "malicious_reason": reason,
            "used_agents": ["GuardAgent(blocked)"],
            "final_answer": f"🚫 {reason}",
            "route": "blocked"
        }
    return {
        "is_malicious": False,
        "malicious_reason": None,
        "used_agents": ["GuardAgent(passed)"]
    }

async def cache_check_node(state: AgentState) -> dict:
    """
    Check cache before expensive processing.

    Important production fix:
    Do NOT cache-hit follow-up requests like:
    - "لخص الكلام ده"
    - "ترجم الإجابة السابقة"
    - "بسّطها"
    because their correct answer depends on the latest last_answer.
    """
    cache = state.get("cache_agent")
    user_query = state.get("user_query", "")

    # Follow-up requests depend on last_answer, so a plain query cache is unsafe here.
    try:
        if heuristic_route(user_query) == "clarify_last":
            return {"used_agents": ["CacheAgent(skipped:followup)"]}
    except Exception:
        pass

    if cache:
        cached_response = cache.get(user_query)
        if cached_response:
            writer = get_stream_writer()
            for ch in cached_response:
                writer(ch)
            return {
                "final_answer": cached_response,
                "used_agents": ["CacheAgent(hit)"],
                "route": "cached"
            }

    return {"used_agents": ["CacheAgent(miss)"]}

async def router_node(state: AgentState) -> dict:
    route = heuristic_route(state["user_query"])
    return {"route": route, "used_agents": [f"Router({route})"]}

async def optimizer_node(state: AgentState) -> dict:
    llm = get_llm(state["response_model_name"], temperature=0.2)
    search_query = optimize_search_query(state["user_query"], state["chat_history"], llm)
    return {"search_query": search_query, "used_agents": ["QueryOptimizer"]}

async def memory_retrieval_node(state: AgentState) -> dict:
    memory_agent = state.get("memory_agent")
    if not memory_agent:
        return {"memory_context": "", "used_agents": []}
    context, retrieved_items = memory_agent.retrieve_similar(state["user_query"], k=3)
    return {"memory_context": context, "used_agents": ["MemoryAgent(retrieve)"] if context else []}

async def database_node(state: AgentState) -> dict:
    db_agent = state.get("db_agent")
    if not db_agent:
        return {"db_result": "", "used_agents": []}
    result = db_agent.query(state["user_query"])
    return {"db_result": result, "used_agents": ["DatabaseAgent"] if result else []}

async def retriever_node(state: AgentState) -> dict:
    retriever_agent = state.get("retriever_agent")
    if not retriever_agent:
        return {"retrieved_docs": "", "used_agents": []}
    docs = retriever_agent.retrieve(state["search_query"], k=5)
    return {"retrieved_docs": docs, "used_agents": ["RetrieverAgent"] if docs else []}

async def web_search_node(state: AgentState) -> dict:
    """البحث على الويب إذا كان مطلوبًا"""
    if not state.get("use_web_search", False):
        return {"web_results": "", "used_agents": []}
    
    api_key = os.getenv("TAVILY_API_KEY", "")
    try:
        web_tool = WebSearchTool(api_key=api_key, use_tavily=bool(api_key))
        results = await web_tool.search(state["search_query"], max_results=3)
        return {"web_results": results, "used_agents": ["WebSearch"] if results else []}
    except Exception as e:
        print(f"[WebSearch] Failed: {e}")
        return {"web_results": "", "used_agents": []}

async def direct_response_node(state: AgentState) -> dict:
    writer = get_stream_writer()
    llm = get_llm(state["response_model_name"], temperature=0.2)
    prompt = f"""You are a senior mobile security engineer. Greet the user and offer help. Do not include any word like "SAFE" or "MALICIOUS" in your response. Just answer naturally.

User: {state['user_query']}

CRITICAL INSTRUCTION: You MUST reply in the EXACT SAME LANGUAGE as the user's question. If the user's question is in Arabic, your entire answer MUST be in Arabic. If it is in English, answer in English.

Answer:"""
    full_response = ""
    async for chunk in llm.astream(prompt):
        if chunk.content:
            full_response += chunk.content
    cleaned = remove_safe_prefix(full_response)
    for ch in cleaned:
        writer(ch)
    memory_agent = state.get("memory_agent")
    if memory_agent:
        memory_agent.add_interaction(state["user_query"], cleaned, {"user_id": state["user_id"], "route": "direct"})
    cache_agent = state.get("cache_agent")
    if cache_agent:
        cache_agent.set(state["user_query"], cleaned)
    return {"final_answer": cleaned, "used_agents": ["ResponseAgent"]}

async def clarify_response_node(state: AgentState) -> dict:
    """
    Handles follow-up requests that operate on the previous answer:
    translate, summarize, explain, simplify, repeat, or clarify.

    Full-marks production fixes:
    - Intent is classified deterministically in code, not only by prompt.
    - Each intent uses a separate prompt.
    - Arabic output gets stricter terminology rules and light post-processing.
    - Follow-up responses are not cached by plain query text.
    - Saves useful follow-up responses to memory.
    """
    writer = get_stream_writer()
    llm = get_llm(state["response_model_name"], temperature=0.1)

    user_request = state.get("user_query", "")
    previous_answer = state.get("last_answer", "")
    intent = deterministic_followup_intent(user_request)

    if not previous_answer.strip():
        fallback = (
            "لا توجد إجابة سابقة أعمل عليها. اسألني سؤالًا أولًا، ثم اطلب مني الترجمة أو التلخيص أو التبسيط."
            if has_arabic(user_request)
            else "There is no previous answer to work on. Ask me a question first, then ask me to translate, summarize, or simplify it."
        )
        for ch in fallback:
            writer(ch)
        return {
            "final_answer": fallback,
            "followup_intent": intent,
            "used_agents": [f"IntentClassifier({intent})", "ResponseAgent(clarify:no_previous_answer)"]
        }

    memory_context = state.get("memory_context", "")
    memory_agent = state.get("memory_agent")
    if not memory_context and memory_agent:
        try:
            memory_context, _ = memory_agent.retrieve_similar(user_request, k=3)
        except Exception as e:
            print(f"[Memory] Clarify retrieval failed: {e}")
            memory_context = ""

    prompt = build_followup_prompt(
        intent=intent,
        user_request=user_request,
        previous_answer=previous_answer,
        memory_context=memory_context
    )

    full_response = ""
    async for chunk in llm.astream(prompt):
        if chunk.content:
            full_response += chunk.content

    cleaned = remove_safe_prefix(full_response)
    if has_arabic(user_request):
        cleaned = normalize_arabic_output(cleaned)

    for ch in cleaned:
        writer(ch)

    if memory_agent:
        memory_agent.add_interaction(
            user_request,
            cleaned,
            {"user_id": state["user_id"], "route": "clarify", "intent": intent}
        )

    # Do NOT cache follow-up responses by plain user_request.
    # "لخص الكلام ده" and "بسّطها" depend on the latest previous_answer.

    return {
        "final_answer": cleaned,
        "memory_context": memory_context,
        "followup_intent": intent,
        "used_agents": [f"IntentClassifier({intent})", "ResponseAgent(clarify)"]
    }

async def combined_rag_response_node(state: AgentState) -> dict:
    """
    استجابة RAG محسنة تدمج:
    - الذاكرة (المحسنة)
    - قاعدة البيانات (MASVS)
    - المستندات المسترجعة (Qdrant)
    - نتائج البحث على الويب (اختياري)
    """
    writer = get_stream_writer()
    llm = get_llm(state["response_model_name"], temperature=0.2)
    
    # تجميع السياق من جميع المصادر
    contexts = []
    
    if state.get("db_result"):
        contexts.append(f"[MASVS Requirement]\n{state['db_result']}")
    
    if state.get("retrieved_docs"):
        contexts.append(f"[Documentation]\n{state['retrieved_docs']}")
    
    if state.get("web_results"):
        contexts.append(f"[Web Search Results]\n{state['web_results']}")
    
    if state.get("memory_context"):
        contexts.append(f"[Conversation Context]\n{state['memory_context']}")

    if state.get("last_answer"):
        contexts.append(f"[Last Answer]\n{state['last_answer']}")
    
    context_text = "\n\n".join(contexts) if contexts else "No specific documentation found. Use your general knowledge as a security expert."
    
    prompt_template = f"""
You are a senior mobile security engineer.

LANGUAGE RULES (VERY IMPORTANT):
- Detect the language of the user's question automatically.
- ALWAYS respond in the SAME language as the user.
- If the user asks to translate, summarize, or explain, you MUST follow that instruction exactly.

TASK HANDLING:
- If the user says "summarize" or "لخص": give a concise summary.
- If the user says "translate" or "ترجم": translate correctly.
- If the user says "explain" or "اشرح": give detailed explanation.
- If unclear, answer normally.

STYLE:
- Be clear and professional.
- Use simple and correct Arabic when responding in Arabic.
- Avoid broken or mixed language.
- When responding in Arabic, do NOT translate technical concepts word-by-word.
- Use these terms consistently: encryption = تشفير, authentication = مصادقة, key = مفتاح, algorithm = خوارزمية, unauthorized access = وصول غير مصرح به.
- Never use broken terms such as "تحفيز" for encryption or "الغذاء الأمني" for algorithm/security.

IMPORTANT:
- Do NOT add closing phrases like "feel free to ask".
- Do NOT add unnecessary greetings or repetition.

Context:
{context_text}

Question: {state['user_query']}

Answer:
"""
    
    full_response = ""
    async for chunk in llm.astream(prompt_template):
        if chunk.content:
            full_response += chunk.content
    
    cleaned = remove_safe_prefix(full_response)
    
    # دفق الإجابة
    for ch in cleaned:
        writer(ch)
    
    # تخزين في الذاكرة المحسنة
    memory_agent = state.get("memory_agent")
    if memory_agent:
        memory_agent.add_interaction(state["user_query"], cleaned, {"user_id": state["user_id"], "route": state["route"]})
    
    # تخزين في الكاش
    cache_agent = state.get("cache_agent")
    if cache_agent:
        cache_agent.set(state["user_query"], cleaned)
    
    return {"final_answer": cleaned, "used_agents": ["ResponseAgent(RAG)"]}

# ------------------------------
# 4. Conditional Edges
# ------------------------------
def after_guard_condition(state: AgentState) -> str:
    if state["is_malicious"]:
        return "blocked"
    return "safe"

def after_cache_condition(state: AgentState) -> str:
    if state.get("route") == "cached":
        return "cached"
    return "continue"

def after_router_condition(state: AgentState) -> str:
    route = state["route"]
    if route == "direct":
        return "direct"
    elif route == "clarify_last":
        return "clarify"
    else:
        return "retrieve"

def after_db_condition(state: AgentState) -> str:
    if state["db_result"]:
        return "use_db"
    return "do_retrieval"

# ------------------------------
# 5. Build the Graph (محدث)
# ------------------------------
def build_security_graph(
    memory_agent: EnhancedMemoryAgent,
    db_agent: DatabaseAgent,
    retriever_agent: RetrieverAgent,
    guard_model_name: str,
    response_model_name: str,
    cache_agent: Optional[CacheAgent] = None,
    enable_web_search: bool = False
) -> StateGraph:
    
    workflow = StateGraph(AgentState)
    
    # إضافة العقد
    workflow.add_node("guard", guard_node)
    workflow.add_node("cache_check", cache_check_node)
    workflow.add_node("router", router_node)
    workflow.add_node("optimizer", optimizer_node)
    workflow.add_node("memory_retrieval", memory_retrieval_node)
    workflow.add_node("database", database_node)
    workflow.add_node("retriever", retriever_node)
    workflow.add_node("web_search", web_search_node)
    workflow.add_node("direct_response", direct_response_node)
    workflow.add_node("clarify_response", clarify_response_node)
    workflow.add_node("combined_rag_response", combined_rag_response_node)
    
    workflow.set_entry_point("guard")
    
    # Guard -> Cache Check
    workflow.add_conditional_edges(
        "guard",
        after_guard_condition,
        {"blocked": END, "safe": "cache_check"}
    )
    
    # Cache Check -> Router or END
    workflow.add_conditional_edges(
        "cache_check",
        after_cache_condition,
        {"cached": END, "continue": "router"}
    )
    
    # Router -> Direct / Clarify / Optimizer
    workflow.add_conditional_edges(
        "router",
        after_router_condition,
        {"direct": "direct_response", "clarify": "clarify_response", "retrieve": "optimizer"}
    )
    
    # سلسلة RAG
    workflow.add_edge("optimizer", "memory_retrieval")
    workflow.add_edge("memory_retrieval", "database")
    workflow.add_conditional_edges(
        "database",
        after_db_condition,
        {"use_db": "web_search", "do_retrieval": "retriever"}
    )
    workflow.add_edge("retriever", "web_search")
    workflow.add_edge("web_search", "combined_rag_response")
    
    # نهاية المسارات المباشرة
    workflow.add_edge("direct_response", END)
    workflow.add_edge("clarify_response", END)
    workflow.add_edge("combined_rag_response", END)
    
    graph = workflow.compile()
    return graph

# ------------------------------
# 6. Streaming Wrapper (محدث)
# ------------------------------
async def run_security_assistant_streaming(
    user_query: str,
    user_id: str,
    last_answer: str,
    chat_history: List[Dict],
    memory_agent: EnhancedMemoryAgent,
    db_agent: DatabaseAgent,
    retriever_agent: RetrieverAgent,
    guard_model_name: str,
    response_model_name: str,
    cache_agent: Optional[CacheAgent] = None,
    enable_web_search: bool = False
) -> AsyncIterator[Dict[str, Any]]:
    
    
    initial_state: AgentState = {
        "user_query": user_query,
        "user_id": user_id,
        "last_answer": last_answer,
        "chat_history": chat_history,
        "is_malicious": False,
        "malicious_reason": None,
        "route": "",
        "followup_intent": "",
        "search_query": user_query,
        "memory_context": "",
        "db_result": "",
        "retrieved_docs": "",
        "web_results": "",
        "final_answer": "",
        "used_agents": [],
        "guard_model_name": guard_model_name,
        "response_model_name": response_model_name,
        "memory_agent": memory_agent,
        "db_agent": db_agent,
        "retriever_agent": retriever_agent,
        "cache_agent": cache_agent,
        "use_web_search": enable_web_search
    }
    
    graph = build_security_graph(
        memory_agent=memory_agent,
        db_agent=db_agent,
        retriever_agent=retriever_agent,
        guard_model_name=guard_model_name,
        response_model_name=response_model_name,
        cache_agent=cache_agent,
        enable_web_search=enable_web_search
    )
    
    config = RunnableConfig(configurable={"thread_id": user_id})

    streamed_anything = False

    async for event in graph.astream_events(initial_state, config=config, version="v2"):
        if event["event"] == "on_chat_model_stream":
            chunk = event["data"]["chunk"]
            if chunk.content:
                streamed_anything = True
                yield {"type": "chunk", "content": chunk.content}

        elif event["event"] == "on_chain_end":
            name = event.get("name", "")
            output = event.get("data", {}).get("output", {})

            # Handle blocked Guard response or Cache hit response
            if name in ["guard", "cache_check"]:
                final_answer = output.get("final_answer", "")
                used_agents = output.get("used_agents", [])

                if final_answer:
                    yield {"type": "chunk", "content": final_answer}
                    yield {"type": "done", "agents": used_agents}
                    break

            # Handle normal response nodes
            if name in ["direct_response", "clarify_response", "combined_rag_response"]:
                final_answer = output.get("final_answer", "")
                used_agents = output.get("used_agents", [])

                # مهم: ما ترجعش الرد كامل لو already streamed
                if final_answer and not streamed_anything:
                    yield {"type": "chunk", "content": final_answer}

                yield {"type": "done", "agents": used_agents}
                break