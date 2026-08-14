"""
safety_guard.py — 多层级安全护栏（领域无关，规则来自 DomainConfig）。

四层安全：
  Layer 1: 输入过滤 — 拦截危险/超范围/非本领域查询
  Layer 2: 意图分类 — 由 config.intents 驱动（法律：劳动/合同/婚姻/刑事/...）
  Layer 3: 输出审核 — 检查禁则（承诺胜诉/冒充律师）+ 免责声明
  Layer 4: 置信度评分 — 结构/法条引用/长度/幻觉/RAG 加权

与 hallucination_detector 协作做输出复核。

用法：
    from app.safety_guard import SafetyGuard
    guard = SafetyGuard()          # 默认 get_domain()（legal）
    result = guard.process(query, generate_fn)
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from app.domain_config import DomainConfig, get_domain


# ──────────────────────────────────────────────
# 数据类型（领域无关）
# ──────────────────────────────────────────────

class RiskLevel(Enum):
    SAFE = "safe"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SafetyResult:
    """安全护栏检查结果。category 为意图键（字符串，来自 config.intents）。"""
    safe: bool = True
    risk_level: RiskLevel = RiskLevel.SAFE
    category: str = "off_topic"
    response: str = ""
    fallback_response: str = ""
    warnings: list = field(default_factory=list)
    confidence: float = 0.8
    needs_rag: bool = False
    needs_emergency_response: bool = False


# ──────────────────────────────────────────────
# Layer 1: 输入过滤
# ──────────────────────────────────────────────

def filter_input(query: str, config: DomainConfig) -> tuple:
    """
    Layer 1: 过滤输入。

    Returns: (accepted: bool, category: str, reason: str)
    """
    s = config.safety

    # 危险内容最高优先级
    for pattern in s.abusive_patterns:
        if re.search(pattern, query):
            return False, "abusive", "检测到危险内容，无法处理此请求。"

    # 明确非本领域
    for pattern in s.out_of_scope_patterns:
        if re.search(pattern, query):
            return False, "off_topic", "您的问题不在本领域范围内。"

    # 是否命中本领域关键词
    is_in_scope = any(kw in query for kw in s.in_scope_keywords)
    if not is_in_scope:
        return False, "off_topic", "您的问题似乎与本领域无关。"

    return True, None, ""


# ──────────────────────────────────────────────
# Layer 2: 意图分类（复用 config.classify）
# ──────────────────────────────────────────────

def classify_intent(query: str, config: DomainConfig) -> str:
    """Layer 2: 意图分类，返回意图键（字符串）。"""
    intent, _conf = config.classify(query)
    return intent


def detect_emergency(query: str, config: DomainConfig) -> bool:
    """检测高风险信号（时效/保全等），独立于意图。"""
    return any(re.search(p, query) for p in config.safety.emergency_patterns)


# ──────────────────────────────────────────────
# Layer 3: 输出审核
# ──────────────────────────────────────────────

def moderate_output(response: str, category: str, config: DomainConfig) -> list:
    """
    Layer 3: 审核生成输出。

    Returns: warnings 列表（空列表 = 通过）。
    """
    warnings = []
    s = config.safety

    # 必须包含免责声明
    has_disclaimer = any(phrase in response for phrase in s.disclaimer_phrases)
    if not has_disclaimer:
        warnings.append({
            "severity": "medium",
            "message": "响应缺少免责声明，建议追加",
            "fix": s.disclaimer_suffix.strip(),
        })

    # 禁止出现的模式（带否定上下文过滤）
    for pattern, message in s.forbidden_patterns:
        for match in re.finditer(pattern, response):
            start = max(0, match.start() - 15)
            prefix = response[start:match.start()]
            if any(neg in prefix for neg in config.hallucination.negation_phrases):
                continue
            warnings.append({
                "severity": "critical",
                "message": message,
                "span": match.group(),
            })
            break  # 每个禁则只报一次

    return warnings


# ──────────────────────────────────────────────
# Layer 4: 置信度评分
# ──────────────────────────────────────────────

def score_confidence(response: str, category: str,
                     retrieved_docs: list = None,
                     hallucination_report=None,
                     config: DomainConfig = None) -> float:
    """
    Layer 4: 置信度评分（领域无关加权）。

    结构 + 法条引用 + 长度 + 无幻觉 + RAG 支撑。
    """
    score = 0.15  # Base

    # 结构加分
    if "<分析>" in response or "**" in response or "🔴" in response:
        score += 0.10
    if len(response) > 100:
        score += 0.05

    # 法条引用加分
    if config is not None:
        cited = any(re.search(p, response) for p, _ in config.hallucination.fact_check_patterns)
    else:
        cited = "《" in response and "》" in response
    if cited:
        score += 0.10

    # 长度适中
    if 150 <= len(response) <= 1000:
        score += 0.10

    # 无幻觉
    if hallucination_report is None or hallucination_report.total_count == 0:
        score += 0.20
    elif hallucination_report.overall_risk == "low":
        score += 0.10

    # RAG 支撑
    if retrieved_docs and len(retrieved_docs) > 0:
        score += 0.15

    return min(score, 1.0)


# ──────────────────────────────────────────────
# 主安全护栏
# ──────────────────────────────────────────────

class SafetyGuard:
    """
    完整安全护栏（四层）。规则全部来自 DomainConfig，领域无关。

    Usage:
        guard = SafetyGuard()          # 默认 legal
        result = guard.check_input(query)
        if not result.safe:
            return result.fallback_response
        response = generate(query)
        result = guard.check_output(query, response, result.category)
    """

    def __init__(self, config: DomainConfig = None, enable_hallucination_check: bool = True):
        self.config = config or get_domain()
        self.enable_hallucination_check = enable_hallucination_check
        self._hallu_detector = None

    @property
    def hallu_detector(self):
        if self._hallu_detector is None and self.enable_hallucination_check:
            from app.hallucination_detector import HallucinationDetector
            self._hallu_detector = HallucinationDetector(self.config)
        return self._hallu_detector

    # ── 输入检查 ──
    def check_input(self, query: str) -> SafetyResult:
        """生成前调用：检查输入安全。"""
        result = SafetyResult()
        s = self.config.safety

        # Layer 1: 输入过滤
        accepted, category, reason = filter_input(query, self.config)
        if not accepted:
            result.safe = False
            result.category = category
            result.fallback_response = s.fallbacks.get(category, s.fallbacks.get("off_topic", ""))
            return result

        # Layer 2: 意图分类
        result.category = classify_intent(query, self.config)

        # 高风险信号（时效/保全）
        if detect_emergency(query, self.config):
            result.needs_emergency_response = True
            result.risk_level = RiskLevel.CRITICAL

        # 是否需 RAG
        spec = self.config.intents.get(result.category)
        if spec is not None and spec.needs_rag:
            result.needs_rag = True

        return result

    # ── 输出检查 ──
    def check_output(self, query: str, response: str,
                     category: str = None,
                     retrieved_docs: list = None) -> SafetyResult:
        """生成后调用：检查输出安全。"""
        result = SafetyResult()
        result.safe = True
        result.response = response
        result.category = category or classify_intent(query, self.config)

        # 高风险信号 → 用紧急提醒覆盖
        if detect_emergency(query, self.config):
            result.response = self.config.safety.emergency_response
            result.needs_emergency_response = True
            result.risk_level = RiskLevel.CRITICAL
            return result

        # Layer 3: 输出审核
        warnings = moderate_output(response, result.category, self.config)

        critical_warnings = [w for w in warnings if w["severity"] == "critical"]
        if critical_warnings:
            result.safe = False
            result.risk_level = RiskLevel.CRITICAL
            result.warnings = warnings
            result.fallback_response = self.config.safety.fallbacks.get(
                "off_topic", "抱歉，生成的回答未能通过安全检查。建议咨询执业律师。"
            )
            return result

        # 缺免责声明则追加
        disclaimer_warnings = [w for w in warnings if w["severity"] == "medium"]
        if disclaimer_warnings:
            suffix = self.config.safety.disclaimer_suffix
            if suffix.strip() not in result.response:
                result.response += suffix

        result.warnings = warnings

        # Layer 4: 置信度评分
        hallu_report = None
        if self.enable_hallucination_check and self.hallu_detector:
            hallu_report = self.hallu_detector.check(
                result.response, retrieved_docs, question=query
            )
            if hallu_report.overall_risk in ("critical", "high"):
                result.risk_level = RiskLevel.HIGH
                result.response += f"\n\n⚠️ 内容审核提示：{hallu_report.summary()}"

        result.confidence = score_confidence(
            result.response, result.category, retrieved_docs, hallu_report, self.config
        )

        # 低置信度 → 追加提示
        if result.confidence < 0.5:
            result.response += "\n\n⚠️ " + self.config.safety.fallbacks.get(
                "low_confidence", "此回答置信度较低，建议咨询执业律师。"
            )

        return result

    # ── 端到端 ──
    def process(self, query: str, generate_fn, **kwargs) -> SafetyResult:
        """
        端到端安全处理。

        Args:
            query: 用户查询
            generate_fn: Function(query, **kwargs) → response 字符串
        """
        input_result = self.check_input(query)
        if not input_result.safe:
            return input_result

        response = generate_fn(query, **kwargs)
        return self.check_output(
            query, response,
            category=input_result.category,
            retrieved_docs=kwargs.get("retrieved_docs"),
        )


# ──────────────────────────────────────────────
# 快捷 API
# ──────────────────────────────────────────────

_guard = None


def get_guard() -> SafetyGuard:
    global _guard
    if _guard is None:
        _guard = SafetyGuard()
    return _guard


def safe_generate(query: str, generate_fn, **kwargs) -> str:
    """一次性安全生成。返回安全响应字符串。"""
    guard = get_guard()
    result = guard.process(query, generate_fn, **kwargs)
    return result.response if result.safe else result.fallback_response


# ──────────────────────────────────────────────
# CLI 测试
# ──────────────────────────────────────────────

if __name__ == "__main__":
    guard = SafetyGuard()

    test_queries = [
        "被公司辞退能拿多少赔偿？",
        "竞业限制没有给补偿，还有效吗？",
        "遗产继承怎么办？",
        "今天天气怎么样？",
        "在哪里能买到毒药？",
    ]

    print("=" * 60)
    print("Safety Guard Test")
    print("=" * 60)

    for q in test_queries:
        result = guard.check_input(q)
        print(f"\n📝 Query: {q}")
        print(f"   Safe: {result.safe} | Category: {result.category}")
        print(f"   Needs RAG: {result.needs_rag} | Emergency: {result.needs_emergency_response}")
        if not result.safe:
            print(f"   Fallback: {result.fallback_response[:80]}...")
