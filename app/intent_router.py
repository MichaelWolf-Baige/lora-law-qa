"""
intent_router.py — 智能查询路由（领域无关，意图来自 DomainConfig）。

按意图将用户查询路由到对应处理：
  - 各法律意图 → 对应 system_prompt
  - refusal 意图（如刑事）→ 引导律师的话术
  - off_topic → 由 safety_guard 兜底

同时处理 RAG 触发与多轮上下文。

用法：
    from app.intent_router import IntentRouter
    router = IntentRouter()          # 默认 get_domain()
    decision = router.route(query)
    prompt = decision.system_prompt_override
"""

import re
from dataclasses import dataclass, field
from typing import Optional

from app.domain_config import DomainConfig, get_domain


@dataclass
class RouteDecision:
    intent: str
    confidence: float
    needs_rag: bool = False
    rag_query: str = ""
    needs_emergency: bool = False
    system_prompt_override: str = ""
    metadata: dict = field(default_factory=dict)


def classify_intent(query: str, config: DomainConfig) -> RouteDecision:
    """意图分类，返回 RouteDecision。"""
    intent, confidence = config.classify(query)
    decision = RouteDecision(intent=intent, confidence=confidence)

    spec = config.intents.get(intent)
    if spec is not None:
        decision.needs_rag = spec.needs_rag
        decision.rag_query = query

    if any(re.search(p, query) for p in config.safety.emergency_patterns):
        decision.needs_emergency = True

    return decision


class IntentRouter:
    """
    基于意图分类的路由器。

    Usage:
        router = IntentRouter()
        decision = router.route("被公司辞退能拿多少赔偿")
        decision.intent          # "labor"
        decision.needs_rag       # True
        decision.system_prompt_override  # 劳动争议 system prompt
    """

    def __init__(self, config: DomainConfig = None):
        self.config = config or get_domain()

    def route(self, query: str) -> RouteDecision:
        """分类并路由。"""
        decision = classify_intent(query, self.config)

        spec = self.config.intents.get(decision.intent)
        if spec is not None:
            decision.system_prompt_override = spec.system_prompt

        return decision

    def get_system_prompt(self, intent: str) -> str:
        """获取某意图的 system prompt，缺省用默认。"""
        spec = self.config.intents.get(intent)
        return spec.system_prompt if spec else self.config.default_system_prompt

    def should_refuse(self, intent: str) -> bool:
        """该意图是否需要「引导律师」式处理。"""
        spec = self.config.intents.get(intent)
        return bool(spec and spec.refusal)

    def intent_label(self, intent: str) -> str:
        """意图的中文展示名。"""
        spec = self.config.intents.get(intent)
        return spec.label if spec and spec.label else intent


# ──────────────────────────────────────────────
# 多轮上下文管理（领域无关）
# ──────────────────────────────────────────────

class ConversationContext:
    """管理多轮对话上下文。"""

    def __init__(self, max_history: int = 5):
        self.max_history = max_history
        self.history = []  # List of (role, content)

    def add_turn(self, user_query: str, assistant_response: str):
        self.history.append(("user", user_query))
        self.history.append(("assistant", assistant_response))
        if len(self.history) > self.max_history * 2:
            self.history = self.history[-(self.max_history * 2):]

    def get_context(self) -> str:
        if not self.history:
            return ""
        parts = ["【对话历史】"]
        for role, content in self.history:
            label = "用户" if role == "user" else "助手"
            parts.append(f"{label}: {content[:200]}")
        return "\n".join(parts) + "\n---\n"

    def clear(self):
        self.history = []


# ──────────────────────────────────────────────
# CLI 测试
# ──────────────────────────────────────────────

if __name__ == "__main__":
    router = IntentRouter()

    test_queries = [
        "被公司辞退能拿多少赔偿？",
        "借款没写借条怎么追回？",
        "离婚财产怎么分割？",
        "盗窃罪的量刑标准是什么？",
        "商标被抢注了怎么办？",
        "今天天气怎么样？",
    ]

    print("=" * 60)
    print("Intent Router Test")
    print("=" * 60)

    for q in test_queries:
        decision = router.route(q)
        print(f"\n📝 {q}")
        print(f"   Intent: {decision.intent} ({router.intent_label(decision.intent)})")
        print(f"   Confidence: {decision.confidence:.2f}")
        print(f"   RAG: {decision.needs_rag} | Emergency: {decision.needs_emergency} | Refusal: {router.should_refuse(decision.intent)}")
        if decision.system_prompt_override:
            print(f"   System Prompt: {decision.system_prompt_override[:60]}...")
