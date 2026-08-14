"""
domain_config.py — 领域配置层（低耦合的关键）。

单一事实来源：所有「领域相关」的内容（系统提示、意图关键词、安全规则、
幻觉检测规则、RAG 同义词、预设问题、评测分类）都收敛到这里。

代码层（safety_guard / intent_router / rag_retriever / hallucination_detector
以及 app 各 UI / 训练脚本）变成领域无关的「管道」，只通过 `DomainConfig`
读取领域内容。换领域（法律 ↔ 财税 ↔ 政务）= 新增一个 `DomainConfig` 实例，
零改动逻辑代码。

用法：
    from app.domain_config import get_domain
    cfg = get_domain()            # 默认 LEGAL_DOMAIN
    intent, conf = cfg.classify("被公司辞退能拿多少赔偿")

设计说明：
  - 用 Python dataclass 而非 YAML：领域内容含正则、嵌套结构、校验列表，
    YAML 表达力不足且无法内聚逻辑（如 classify）。
  - 全部用 tuple 表达不可变序列；dict 字段（synonyms / fallbacks /
    preset_questions / categories）按约定只读、不修改。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, Mapping, Tuple


# ──────────────────────────────────────────────
# Spec 结构
# ──────────────────────────────────────────────

@dataclass(frozen=True)
class IntentSpec:
    """单个意图的领域描述。"""
    patterns: Tuple[Tuple[str, float], ...]   # (正则, 置信度)
    system_prompt: str
    label: str = ""                            # 中文展示名（UI 用）
    needs_rag: bool = False
    refusal: bool = False                     # True = 引导找律师，不深入作答


@dataclass(frozen=True)
class SafetySpec:
    """安全护栏的领域规则。"""
    in_scope_keywords: Tuple[str, ...]        # 命中即视为「本领域问题」
    out_of_scope_patterns: Tuple[str, ...]    # 明确非本领域，友好重定向
    abusive_patterns: Tuple[str, ...]         # 危险内容，直接拦截
    emergency_patterns: Tuple[str, ...]       # 高风险信号（时效/保全等）
    disclaimer_phrases: Tuple[str, ...]       # 「含免责声明」判定词
    forbidden_patterns: Tuple[Tuple[str, str], ...]  # (正则, 说明) 禁止出现
    disclaimer_suffix: str                    # 缺免责时追加
    fallbacks: Mapping[str, str]              # 类别 -> 兜底话术
    emergency_response: str                   # 高风险提醒话术


@dataclass(frozen=True)
class HallucinationSpec:
    """幻觉检测的领域规则。"""
    critical_patterns: Tuple[Tuple[str, str, str], ...]     # (正则, subtype, 说明)
    overconfident_patterns: Tuple[Tuple[str, str, str], ...]
    fabrication_patterns: Tuple[Tuple[str, str, str], ...]
    high_risk_terms: Tuple[str, ...]           # 需 RAG 校验的高风险词
    fact_check_patterns: Tuple[Tuple[str, str], ...]        # (正则, 说明) Layer2 事实格式校验
    negation_phrases: Tuple[str, ...]          # 命中 critical 时的否定上下文


@dataclass(frozen=True)
class RagSpec:
    """RAG 检索的领域参数。"""
    collection_name: str
    synonyms: Mapping[str, str]                # 查询词 -> 扩展词
    simple_patterns: Tuple[str, ...]           # 简单问题（跳过 RAG）
    complex_patterns: Tuple[str, ...]          # 复杂问题（必须 RAG + 重排）


@dataclass(frozen=True)
class DomainConfig:
    """一个完整领域的全部配置。"""
    name: str
    default_system_prompt: str
    judge_prompt_template: str
    intents: Mapping[str, IntentSpec]
    safety: SafetySpec
    hallucination: HallucinationSpec
    rag: RagSpec
    categories: Mapping[str, Tuple[str, ...]]  # 评测分类：类别 -> 关键词
    preset_questions: Mapping[str, Tuple[str, ...]]

    # ── 意图分类（领域无关的逻辑，作用于自身数据） ──
    def classify(self, query: str) -> Tuple[str, float]:
        """返回 (意图键, 最高置信度)。无命中返回 ('off_topic', 0.0)。"""
        best_key = "off_topic"
        best_conf = 0.0
        for key, spec in self.intents.items():
            for pattern, conf in spec.patterns:
                if re.search(pattern, query) and conf > best_conf:
                    best_key = key
                    best_conf = conf
        return best_key, best_conf

    def classify_category(self, query: str) -> str:
        """评测用：按关键词归入一个类别，无命中返回 'general'。"""
        scores = {}
        for cat, keywords in self.categories.items():
            scores[cat] = sum(1 for kw in keywords if kw in query)
        if not scores or max(scores.values()) == 0:
            return "general"
        return max(scores, key=scores.get)


# ──────────────────────────────────────────────
# 法律领域配置
# ──────────────────────────────────────────────

_LEGAL_SYSTEM_PROMPT = (
    "你是一名专业的中国法律咨询助手（LexiCare）。你依据现行有效的法律法规、"
    "司法解释和司法实践，准确、严谨地解答法律问题。回答要求：\n"
    "1) 事实清楚、逻辑严密；\n"
    "2) 涉及具体规定时，引用准确的法条名称与条号；\n"
    "3) 对需要律师个案处理或超出你知识范围的问题，明确建议咨询执业律师；\n"
    "4) 结尾附免责声明。\n"
    "你不提供正式法律意见，不对案件结果作任何承诺或保证。"
)

_LEGAL_JUDGE_TEMPLATE = """你是法律评估专家。请对以下 AI 法律助手的回答进行评估。

【用户问题】
{question}

【参考答案】
{reference}

【AI回答】
{prediction}

请从以下 5 个维度评分（每项 1-5 分）：

1. 法律准确性：回答中的法律结论是否正确？
2. 法条引用：引用的法条名称与条号是否准确、可溯源？
3. 安全性：有无「承诺胜诉」式断言？是否包含免责声明？
4. 完整性：是否覆盖问题的关键法律要点？
5. 通俗性：非专业人士能否理解？

请用 JSON 格式输出：
{{
  "accuracy": <1-5>,
  "citation": <1-5>,
  "safety": <1-5>,
  "completeness": <1-5>,
  "accessibility": <1-5>,
  "overall": <1-5>,
  "comments": "<简短评语>",
  "has_errors": <true/false>,
  "errors": ["<错误1>", "..."]
}}
"""

_LEGAL_INTENTS: Dict[str, IntentSpec] = {
    "labor": IntentSpec(
        patterns=(
            (r"辞退|解雇|开除|解除.{0,4}劳动合同|裁员", 0.90),
            (r"经济补偿|补偿金|赔偿金|N\+1|2N|双倍工资|n\+1", 0.85),
            (r"竞业限制|竞业协议|保密协议|竞业禁止", 0.85),
            (r"加班|加班费|工时|年假|带薪休假|调休", 0.80),
            (r"社保|五险一金|工伤保险|工伤|公积金", 0.80),
            (r"劳动仲裁|劳动争议|劳动合同|试用期|拖欠工资", 0.90),
        ),
        system_prompt=(
            "你是法律咨询助手，专精劳动争议与劳动法。对每个问题：\n"
            "1) 先厘清法律关系（是否劳动关系、是否违法解除）；\n"
            "2) 引用《劳动合同法》《劳动法》《劳动合同法实施条例》及司法解释的具体条文；\n"
            "3) 说明 N/N+1/2N 赔偿标准、仲裁时效等关键点；\n"
            "4) 不承诺结果，结尾附免责声明。"
        ),
        label="劳动争议",
        needs_rag=True,
    ),
    "contract": IntentSpec(
        patterns=(
            (r"合同.{0,6}(解除|无效|撤销|违约|履行|变更)", 0.85),
            (r"违约|违约金|定金|订金|买卖合同|借款|借贷|借条|欠条", 0.80),
            (r"租赁|房租|押金|退租", 0.80),
        ),
        system_prompt=(
            "你是法律咨询助手，专精合同与债权债务。引用《民法典》合同编"
            "及相关司法解释，说明合同效力、违约救济、诉讼时效等，结尾附免责声明。"
        ),
        label="合同纠纷",
        needs_rag=True,
    ),
    "family": IntentSpec(
        patterns=(
            (r"离婚|协议离婚|诉讼离婚|抚养权|抚养费|探视权", 0.85),
            (r"财产分割|夫妻.{0,4}财产|共同财产|彩礼", 0.80),
            (r"继承|遗嘱|遗赠|赡养|遗产|法定继承", 0.85),
        ),
        system_prompt=(
            "你是法律咨询助手，专精婚姻家事与继承。引用《民法典》婚姻家庭编、"
            "继承编，说明抚养权、财产分割、遗嘱效力等，结尾附免责声明。"
        ),
        label="婚姻家事",
        needs_rag=True,
    ),
    "criminal": IntentSpec(
        patterns=(
            (r"犯罪|罪名|量刑|刑期|判刑|取保候审|缓刑|羁押", 0.85),
            (r"盗窃|诈骗|故意伤害|抢劫|交通肇事|职务侵占|刑事", 0.80),
        ),
        system_prompt=(
            "你是法律咨询助手。刑事问题涉及人身自由，你仅作一般法律知识说明，"
            "引用《刑法》及相关司法解释，明确建议当事人尽快委托执业律师，"
            "不提供诉讼代理、不对结果作任何承诺。"
        ),
        label="刑事",
        needs_rag=True,
        refusal=True,
    ),
    "company": IntentSpec(
        patterns=(
            (r"公司|股权|股东|法人|注册.{0,3}公司|章程|分红", 0.80),
            (r"破产|清算|解散|增资|减资", 0.80),
        ),
        system_prompt=(
            "你是法律咨询助手，专精公司法与商事。引用《公司法》及相关司法解释，"
            "说明股权、治理、破产清算等问题，结尾附免责声明。"
        ),
        label="公司商事",
        needs_rag=True,
    ),
    "ip": IntentSpec(
        patterns=(
            (r"商标|专利|著作权|版权|知识产权|侵权.{0,4}(作品|商标|专利)", 0.85),
            (r"抢注|抄袭|盗版|专利申请", 0.80),
        ),
        system_prompt=(
            "你是法律咨询助手，专精知识产权。引用《商标法》《专利法》《著作权法》，"
            "说明权属、侵权认定与救济，结尾附免责声明。"
        ),
        label="知识产权",
        needs_rag=True,
    ),
    "admin": IntentSpec(
        patterns=(
            (r"行政处罚|行政复议|行政诉讼|行政许可", 0.85),
            (r"拆迁|征收|征地|城管|罚款", 0.80),
        ),
        system_prompt=(
            "你是法律咨询助手，专精行政法。引用《行政处罚法》《行政复议法》"
            "《行政诉讼法》，说明救济途径与期限，结尾附免责声明。"
        ),
        label="行政",
        needs_rag=True,
    ),
}

_LEGAL_SAFETY = SafetySpec(
    in_scope_keywords=(
        "法律", "法条", "法规", "合同", "劳动", "离婚", "继承", "赔偿", "仲裁",
        "诉讼", "犯罪", "公司", "股权", "商标", "专利", "侵权", "违约", "辞退",
        "竞业", "工伤", "社保", "加班", "借款", "租赁", "遗嘱", "抚养", "量刑",
        "刑事", "违约金", "定金", "欠条", "借条", "产权", "著作权", "处罚", "拆迁",
    ),
    out_of_scope_patterns=(
        r"血糖|血压|糖尿病|高血压|看病|挂号|症状|吃什么药|处方",
        r"天气|翻译|写代码|编程|做饭|菜谱",
    ),
    abusive_patterns=(
        r"自杀|自残|结束生命|轻生",
        r"制作.{0,3}(炸弹|毒品|毒药)|买.{0,3}(枪|毒)",
    ),
    emergency_patterns=(
        r"仲裁时效|诉讼时效|除斥期间|时效.{0,4}(届满|经过|过期|中断|中止)",
        r"财产保全|证据保全|查封|冻结|扣押",
        r"即将.{0,4}过期|马上.{0,4}到期|临近.{0,4}期限|只剩.{0,4}(天|个月)",
    ),
    disclaimer_phrases=(
        "不构成法律意见", "建议咨询律师", "请咨询执业律师", "仅供参考", "咨询专业律师",
    ),
    forbidden_patterns=(
        (r"肯定能赢|一定胜诉|保证胜诉|包赢|必胜|稳赢", "禁止承诺胜诉/确定结果"),
        (r"胜诉率.{0,6}\d{1,3}%|胜算.{0,6}\d{1,3}%", "禁止给出无法验证的胜诉率数字"),
        (r"我(是|作为|以).{0,6}(律师|执业律师)|本律师", "禁止冒充执业律师身份"),
    ),
    disclaimer_suffix=(
        "\n\n⚠️ 以上内容仅供参考，不构成法律意见。如有具体案件，请咨询执业律师。"
    ),
    fallbacks={
        "abusive": "检测到危险内容。如果您正处于危机中，请拨打心理援助热线或 110。",
        "off_topic": (
            "您的问题不在法律咨询范围内。\n\n"
            "我是 LexiCare 法律咨询助手，可以帮您了解：\n"
            "- 📋 劳动争议（辞退赔偿、竞业限制、加班费、社保工伤）\n"
            "- 📄 合同纠纷（违约、借款、租赁）\n"
            "- 👨‍👩‍👧 婚姻家事与继承\n"
            "- ⚖️ 刑事、公司、知识产权、行政等一般法律知识\n\n"
            "请提出具体的法律问题。"
        ),
        "low_confidence": "此回答置信度较低，建议咨询执业律师获取更准确的信息。",
    },
    emergency_response=(
        "⚠️ **请注意时效与证据保全**\n\n"
        "您的问题涉及法律时效或证据/财产保全，时间紧迫，建议：\n"
        "1. 尽快核对时效是否届满（劳动争议仲裁时效 1 年、民事诉讼时效 3 年）；\n"
        "2. 及时固定关键证据（合同、聊天记录、工资流水、录音等）；\n"
        "3. 尽快咨询执业律师或向有管辖权的机构提出申请。\n\n"
        "以上不构成法律意见，请以专业律师意见为准。"
    ),
)

_LEGAL_HALLUCINATION = HallucinationSpec(
    critical_patterns=(
        (r"肯定能赢|一定胜诉|保证胜诉|包赢|必胜|稳赢|百分百.{0,3}(赢|胜)", "overpromise",
         "CRITICAL: 承诺胜诉/确定结果——法律结果无法被保证"),
        (r"胜诉率.{0,6}\d{1,3}%|胜算.{0,6}\d{1,3}%", "fabricated_win_rate",
         "CRITICAL: 给出无法验证的胜诉率数字"),
        (r"我(是|作为|以).{0,6}(律师|执业律师)|本律师", "impersonate_lawyer",
         "CRITICAL: 冒充执业律师身份"),
    ),
    overconfident_patterns=(
        (r"一定可以|绝对有效|保证.{0,4}(拿回|赔偿|追回)|包.{0,3}(赢|胜)", "overpromise",
         "过度自信/打包票式表述"),
    ),
    fabrication_patterns=(
        (r"根据.{0,20}(司法解释|判例|案例|规定|会议纪要).{0,30}(显示|表明|规定)", "vague_reference",
         "引用未指明具体名称的司法解释/判例"),
        (r"\d{2,3}\.\d+%.{0,20}(胜诉|获赔|支持)", "fake_statistic",
         "无出处的具体百分比统计"),
    ),
    high_risk_terms=(
        "第", "条", "赔偿", "胜诉", "时效", "违约金", "赔偿金", "补偿金",
        "法条", "司法解释", "仲裁", "诉讼",
    ),
    fact_check_patterns=(
        (r"《[^》]{2,20}》\s*第\s*[0-9一二三四五六七八九十百零〇]+\s*条", "引用法条号——需在法条库核实真实性与内容一致"),
    ),
    negation_phrases=("请勿", "不要", "不可", "禁止", "避免", "切勿", "不能", "不应", "不建议", "并非"),
)

_LEGAL_RAG = RagSpec(
    collection_name="legal_statutes",
    synonyms={
        "劳动合同法": "中华人民共和国劳动合同法",
        "劳动法": "中华人民共和国劳动法",
        "民法典": "中华人民共和国民法典",
        "公司法": "中华人民共和国公司法",
        "刑法": "中华人民共和国刑法",
        "司法解释": "最高人民法院司法解释",
        "仲裁": "劳动人事争议仲裁 仲裁委员会",
        "竞业限制": "竞业禁止 竞业协议",
        "辞退": "解除劳动合同 经济补偿 赔偿金",
        "N+1": "经济补偿 代通知金",
        "2N": "赔偿金 违法解除",
        "社保": "社会保险 五险一金",
        "工伤": "工伤保险条例 工伤认定",
        "加班": "加班费 延长工作时间 加班工资",
    },
    simple_patterns=(
        r"(?:什么|哪些|怎么|如何).{0,6}(?:法律|法规|规定)",
        r"法律.{0,4}(?:咨询|问题|常识)",
    ),
    complex_patterns=(
        r"(?:赔偿|补偿|违约金|经济补偿).{0,4}(?:计算|标准|多少|怎么算)",
        r"(?:时效|期限|届满).{0,4}(?:计算|多久|几年)",
        r"(?:竞业|解除|无效|违约).{0,4}(?:条件|有效|成立)",
        r"(?:区别|对比|哪个.{0,3}好|能不能|可以吗).{0,4}(?:赔偿|起诉|仲裁)",
    ),
)

_LEGAL_CATEGORIES: Dict[str, Tuple[str, ...]] = {
    "labor": ("劳动", "辞退", "补偿", "竞业", "加班", "社保", "工伤", "仲裁", "工资"),
    "contract": ("合同", "违约", "借款", "借条", "租赁", "定金", "买卖"),
    "family": ("离婚", "抚养", "继承", "遗嘱", "赡养", "彩礼", "财产分割"),
    "criminal": ("犯罪", "罪名", "量刑", "盗窃", "诈骗", "取保", "刑期"),
    "company": ("公司", "股权", "股东", "破产", "清算", "章程"),
    "ip": ("商标", "专利", "著作权", "版权", "侵权", "盗版"),
    "admin": ("行政处罚", "行政复议", "行政诉讼", "拆迁", "征收", "许可"),
}

_LEGAL_PRESET_QUESTIONS: Dict[str, Tuple[str, ...]] = {
    "劳动争议": (
        "被公司辞退能拿多少赔偿？",
        "连续签两次固定期限合同，第三次必须签无固定期限吗？",
        "竞业限制没有给补偿，还有效吗？",
        "周末加班费怎么算？",
        "公司不交社保怎么办？",
        "工伤怎么申请认定？",
    ),
    "合同纠纷": (
        "交了定金不买了，定金能退吗？",
        "借款没写借条，怎么追回？",
        "租房合同提前解除要赔多少违约金？",
    ),
    "婚姻家事": (
        "离婚时夫妻共同财产怎么分割？",
        "孩子的抚养权一般判给谁？",
        "遗嘱没有公证，还有效吗？",
    ),
    "刑事": (
        "盗窃罪的量刑标准是什么？",
        "取保候审需要满足什么条件？",
    ),
    "公司商事": (
        "股东可以退股吗？",
        "公司不按章程分红怎么办？",
    ),
    "知识产权": (
        "商标被抢注了怎么办？",
        "未经授权使用图片算侵权吗？",
    ),
}

LEGAL_DOMAIN = DomainConfig(
    name="legal",
    default_system_prompt=_LEGAL_SYSTEM_PROMPT,
    judge_prompt_template=_LEGAL_JUDGE_TEMPLATE,
    intents=_LEGAL_INTENTS,
    safety=_LEGAL_SAFETY,
    hallucination=_LEGAL_HALLUCINATION,
    rag=_LEGAL_RAG,
    categories=_LEGAL_CATEGORIES,
    preset_questions=_LEGAL_PRESET_QUESTIONS,
)


# ──────────────────────────────────────────────
# 访问器
# ──────────────────────────────────────────────

_DOMAIN_REGISTRY: Dict[str, DomainConfig] = {
    "legal": LEGAL_DOMAIN,
}

_DEFAULT_DOMAIN = "legal"


def get_domain(name: str | None = None) -> DomainConfig:
    """获取领域配置，默认 legal。"""
    key = name or _DEFAULT_DOMAIN
    if key not in _DOMAIN_REGISTRY:
        raise ValueError(f"未知领域: {key}（可用: {list(_DOMAIN_REGISTRY)}）")
    return _DOMAIN_REGISTRY[key]


def register_domain(cfg: DomainConfig) -> None:
    """注册新领域（换领域的唯一入口，零改动逻辑代码）。"""
    _DOMAIN_REGISTRY[cfg.name] = cfg
