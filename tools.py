# tools.py
import os
import hashlib
import json
from datetime import datetime, timedelta
from typing import Optional, Dict, Tuple
from dataclasses import dataclass

# محاولة استيراد Tavily (اختياري)
try:
    from tavily import TavilyClient
    TAVILY_AVAILABLE = True
except ImportError:
    TAVILY_AVAILABLE = False
    print("[Tools] Tavily not installed. Web search will use DuckDuckGo fallback.")

# DuckDuckGo fallback (بدون API key)
try:
    from duckduckgo_search import DDGS
    DDGS_AVAILABLE = True
except ImportError:
    DDGS_AVAILABLE = False
    print("[Tools] duckduckgo-search not installed. Install with: pip install duckduckgo-search")


class WebSearchTool:
    """أداة البحث على الويب (Tavily API أو DuckDuckGo)"""
    
    def __init__(self, api_key: Optional[str] = None, use_tavily: bool = True):
        self.use_tavily = use_tavily and TAVILY_AVAILABLE and api_key
        if self.use_tavily:
            self.client = TavilyClient(api_key=api_key)
        elif DDGS_AVAILABLE:
            self.ddgs = DDGS()
        else:
            raise ImportError("No web search backend available. Install tavily or duckduckgo-search")
    
    async def search(self, query: str, max_results: int = 3) -> str:
        """البحث وإرجاع النتائج كنص منسق"""
        if self.use_tavily:
            return await self._search_tavily(query, max_results)
        else:
            return await self._search_duckduckgo(query, max_results)
    
    async def _search_tavily(self, query: str, max_results: int) -> str:
        try:
            results = self.client.search(query, max_results=max_results)
            contents = []
            for result in results.get('results', []):
                content = result.get('content', '')
                url = result.get('url', '')
                if content:
                    contents.append(f"[Source: {url}]\n{content[:800]}")
            return "\n\n".join(contents) if contents else ""
        except Exception as e:
            print(f"[WebSearch] Tavily error: {e}")
            return ""
    
    async def _search_duckduckgo(self, query: str, max_results: int) -> str:
        try:
            # DuckDuckGo search returns generator
            results = list(self.ddgs.text(query, max_results=max_results))
            contents = []
            for r in results:
                body = r.get('body', '')
                href = r.get('href', '')
                if body:
                    contents.append(f"[Source: {href}]\n{body[:800]}")
            return "\n\n".join(contents) if contents else ""
        except Exception as e:
            print(f"[WebSearch] DuckDuckGo error: {e}")
            return ""


class CacheAgent:
    """Cache Aware Generation (CAG) – تخزين مؤقت للاستعلامات المتكررة"""
    
    def __init__(self, ttl_minutes: int = 60, max_size: int = 100):
        self.ttl = timedelta(minutes=ttl_minutes)
        self.max_size = max_size
        self.cache: Dict[str, Tuple[str, datetime]] = {}  # hash -> (response, timestamp)
    
    def _hash_query(self, query: str) -> str:
        """إنشاء hash موحد للاستعلام (تجاهل حالة الأحرف والمسافات الزائدة)"""
        normalized = " ".join(query.lower().strip().split())
        return hashlib.md5(normalized.encode()).hexdigest()
    
    def get(self, query: str) -> Optional[str]:
        """استرجاع استجابة من الكاش إذا كانت لا تزال صالحة"""
        key = self._hash_query(query)
        if key in self.cache:
            response, timestamp = self.cache[key]
            if datetime.now() - timestamp < self.ttl:
                return response
            else:
                # منتهي الصلاحية
                del self.cache[key]
        return None
    
    def set(self, query: str, response: str):
        """تخزين استجابة في الكاش"""
        key = self._hash_query(query)
        # إدارة الحجم
        if len(self.cache) >= self.max_size:
            # حذف الأقدم
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k][1])
            del self.cache[oldest_key]
        self.cache[key] = (response, datetime.now())
    
    def clear(self):
        """مسح الكاش بالكامل"""
        self.cache.clear()
    
    def stats(self) -> dict:
        return {"size": len(self.cache), "max_size": self.max_size}
