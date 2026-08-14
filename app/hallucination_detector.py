"""
hallucination_detector.py — 法律幻觉检测（领域无关，规则来自 DomainConfig）。

针对法律领域的三类幻觉：
  Type 1: 事实/引用错误 — 编造法条、伪造胜诉率
  Type 2: 逻辑错误 — 承诺胜诉、过度自信、冒充律师
  Type 3: 编造 — 无出处的司法解释/判例引用、虚假统计

检测层次：
  Layer 1: 规则匹配（快、高精度）
  Layer 2: 事实格式校验（法条号格式、高风险词）
  Layer 3: RAG 交叉验证（声称是否出现在检索文档中）

注：法律正确性无法仅靠正则判定真伪，正则只负责「标记需核实」，
真实法条校验需在 RAG 语料库阶段完成。

用法：
    from app.hallucination_detector import HallucinationDetector
    detector = HallucinationDetector()
    result = detector.check(text, retrieved_docs=None)
"""

import re
from dataclasses import dataclass, field
from typing import Optional

from app.domain_config import DomainConfig, get_domain


# ──────────────────────────────────────────────
# 数据结构
# ──────────────────────────────────────────────

@dataclass
class HallucinationFinding:
    category: str          # factual_error / logical_error / fabrication / omission
    subtype: str
    severity: str          # critical / high / medium / low
    span: str
    explanation: str
    correction: str = ""


@dataclass
class HallucinationReport:
    findings: list = field(default_factory=list)
    text_length: int = 0
    overall_risk: str = "low"

    @property
    def critical_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == "critical")

    @property
    def high_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == "high")

    @property
    def total_count(self) -> int:
        return len(self.findings)

    @property
    def hallucination_rate(self) -> float:
        if self.text_length == 0:
            return 0.0
        return self.total_count / (self.text_length / 1000)

    def summary(self) -> str:
        if not self.findings:
            return "✅ No hallucinations detected."
        lines = [f"🔍 Hallucination Report: {self.total_count} finding(s), Risk: {self.overall_risk.upper()}"]
        for f in self.findings:
            icon = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}.get(f.severity, "⚪")
            lines.append(f"  {icon} [{f.category}/{f.subtype}] {f.explanation}")
            if f.span:
                lines.append(f"     Text: \"{f.span[:100]}...\"" if len(f.span) > 100 else f"     Text: \"{f.span}\"")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "total_findings": self.total_count,
            "critical": self.critical_count,
            "high": self.high_count,
            "overall_risk": self.overall_risk,
            "hallucination_rate_per_1k_chars": round(self.hallucination_rate, 3),
            "findings": [
                {
                    "category": f.category,
                    "subtype": f.subtype,
                    "severity": f.severity,
                    "span": f.span,
                    "explanation": f.explanation,
                }
                for f in self.findings
            ],
        }


class HallucinationDetector:
    """
    多层幻觉检测器，规则来自 DomainConfig。
    """

    def __init__(self, config: DomainConfig = None):
        self.config = config or get_domain()

    def check(self, text: str, retrieved_docs: list = None,
              question: str = None) -> HallucinationReport:
        report = HallucinationReport()
        report.text_length = len(text)

        # Layer 1: 规则匹配
        self._check_critical_patterns(text, report)
        self._check_overconfident_language(text, report)
        self._check_fabrications(text, report)

        # Layer 2: 事实格式校验
        self._check_fact_format(text, report)

        # Layer 3: RAG 交叉验证
        if retrieved_docs:
            self._cross_check_with_rag(text, retrieved_docs, report)
        else:
            self._flag_unverified_claims(text, report)

        report.overall_risk = self._assess_risk(report)
        return report

    # ── Layer 1 ──

    def _check_critical_patterns(self, text: str, report: HallucinationReport):
        for pattern, subtype, explanation in self.config.hallucination.critical_patterns:
            for match in re.finditer(pattern, text):
                start = max(0, match.start() - 15)
                prefix = text[start:match.start()]
                if any(neg in prefix for neg in self.config.hallucination.negation_phrases):
                    continue
                report.findings.append(HallucinationFinding(
                    category="logical_error", subtype=subtype, severity="critical",
                    span=match.group(), explanation=explanation,
                ))

    def _check_overconfident_language(self, text: str, report: HallucinationReport):
        for pattern, subtype, explanation in self.config.hallucination.overconfident_patterns:
            for match in re.finditer(pattern, text):
                report.findings.append(HallucinationFinding(
                    category="logical_error", subtype=subtype, severity="high",
                    span=match.group(), explanation=explanation,
                ))

    def _check_fabrications(self, text: str, report: HallucinationReport):
        for pattern, subtype, explanation in self.config.hallucination.fabrication_patterns:
            for match in re.finditer(pattern, text):
                report.findings.append(HallucinationFinding(
                    category="fabrication", subtype=subtype, severity="medium",
                    span=match.group(), explanation=explanation,
                ))

    # ── Layer 2: 事实格式校验（法条号格式等） ──

    def _check_fact_format(self, text: str, report: HallucinationReport):
        for pattern, explanation in self.config.hallucination.fact_check_patterns:
            for match in re.finditer(pattern, text):
                report.findings.append(HallucinationFinding(
                    category="factual_error", subtype="unverified_citation", severity="low",
                    span=match.group(),
                    explanation=f"{explanation}（正则仅标记需核实，真伪需法条库校验）",
                ))

    def _flag_unverified_claims(self, text: str, report: HallucinationReport):
        for term in self.config.hallucination.high_risk_terms:
            if term in text:
                for match in re.finditer(rf'[^。；\n]*{re.escape(term)}[^。；\n]*', text):
                    report.findings.append(HallucinationFinding(
                        category="omission", subtype="unverified_high_risk", severity="low",
                        span=match.group().strip(),
                        explanation=f"含高风险词 '{term}' —— 应结合法条库核实",
                    ))
                break

    # ── Layer 3: RAG 交叉验证 ──

    def _cross_check_with_rag(self, text: str, retrieved_docs: list, report: HallucinationReport):
        claims = self._extract_legal_claims(text)
        for claim in claims:
            supported = any(self._claim_in_doc(claim, doc) for doc in retrieved_docs)
            if not supported:
                report.findings.append(HallucinationFinding(
                    category="fabrication", subtype="unsupported_claim", severity="high",
                    span=claim,
                    explanation="声称未在检索到的法条/司法解释中找到依据——潜在幻觉",
                ))

    def _extract_legal_claims(self, text: str) -> list:
        claims = []
        sentences = re.split(r'[。；\n]', text)
        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            if re.search(r'(\d+\.?\d*|第.{0,4}条|司法解释|规定|赔偿|时效|应当|不得)', sent):
                if len(sent) > 10:
                    claims.append(sent)
        return claims

    def _claim_in_doc(self, claim: str, doc: str) -> bool:
        key_terms = re.findall(r'[一-鿿]{2,}', claim)
        key_terms = [t for t in key_terms if len(t) >= 3][:5]
        if not key_terms:
            return True
        matches = sum(1 for term in key_terms if term in doc)
        return matches >= len(key_terms) * 0.6

    # ── 风险评估 ──

    def _assess_risk(self, report: HallucinationReport) -> str:
        if report.critical_count > 0:
            return "critical"
        if report.high_count >= 2 or report.total_count >= 5:
            return "high"
        if report.high_count >= 1 or report.total_count >= 3:
            return "medium"
        return "low"


# ──────────────────────────────────────────────
# 快捷 API
# ──────────────────────────────────────────────

_detector = None


def get_detector() -> HallucinationDetector:
    global _detector
    if _detector is None:
        _detector = HallucinationDetector()
    return _detector


def quick_hallucination_check(text: str) -> str:
    detector = get_detector()
    return detector.check(text).summary()


def detailed_hallucination_check(text: str, retrieved_docs: list = None,
                                 question: str = None) -> HallucinationReport:
    detector = get_detector()
    return detector.check(text, retrieved_docs=retrieved_docs, question=question)


# ──────────────────────────────────────────────
# CLI 测试
# ──────────────────────────────────────────────

if __name__ == "__main__":
    test_outputs = [
        # 安全：引用真实法条 + 免责
        "根据《中华人民共和国劳动合同法》第四十七条，经济补偿按劳动者在本单位工作的年限计算。以上内容仅供参考，不构成法律意见。",
        # 危险：承诺胜诉
        "这个案子肯定能赢，胜诉率 99%，放心交给我。",
        # 冒充律师
        "我是律师，你的案件我可以代理，保证帮你拿回赔偿。",
        # 编造引用
        "根据某司法解释显示，87.3% 的劳动争议案件劳动者都能获赔。",
    ]

    detector = HallucinationDetector()

    for i, text in enumerate(test_outputs):
        print(f"\n{'='*60}")
        print(f"Test {i+1}: {text[:60]}...")
        report = detector.check(text)
        print(report.summary())
