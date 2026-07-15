"""纪要质量诊断（Tech Design M4 §12/Phase 4）：范文污染 + 事实一致性。

设计取舍：
- 默认 `rules` 模式纯确定性、零额外 LLM 开销——契合 Phase 4 省 token 初衷；
- `hybrid` 模式在规则命中可疑条目时，用轻量 LLM 裁判（flash 优先、不碰 Air）
  过滤误报，绝不对每场会议全文执行；裁判失败/额度熔断一律 fail-open；
- 检查是**诊断而非判定**：结果写入 `_meta.quality`，只有明确高风险
  （无效引用 seq / 范文独有事实泄漏 / owner·deadline 无原文依据）才触发
  一次重跑，普通疑似项（未溯源数字）仅告警，避免误报导致无意义重跑。
"""
import logging
import re
from dataclasses import dataclass, field

from pydantic import BaseModel

from app.core.config import settings
from app.schemas.summary import SummaryContent

logger = logging.getLogger(__name__)

# 纯阿拉伯数字/百分比（跳过"八成"等中文数词以降误报）
_NUM_RE = re.compile(r"\d+(?:\.\d+)?%?")
# 拉丁词/数字 token：项目代号、版本号、KPI 值等"事实型"记号
_ALNUM_RE = re.compile(r"[A-Za-z][A-Za-z0-9]+|\d+(?:\.\d+)?%?")
# 责任人多值分隔符
_OWNER_SPLIT = re.compile(r"[、,，/]+")

# 中文专名候选提取（范文污染检测的实体白名单式方案）——只抽"专名形状"的串，
# 不做全量中文切词，避免把"同步推进""严格执行"等风格词误判为事实泄漏。
# 引号内项目/专名
_QUOTE_RE = re.compile(r"[「『“”‘’《〈]([^」』“”‘’》〉]{2,12})[」』“”‘’》〉]")
# 带机构/项目/地点后缀的实体（后缀前需 ≥2 个汉字）。只用多字、低歧义后缀：
# 单字后缀（部/局/处/科/组）会误吞"剩余部分""全部"等常用词，故一律不用
_ENTITY_RE = re.compile(
    r"[一-鿿]{2,8}(?:分公司|公司|集团|事业部|研究院|研究所|部门|项目组|项目|"
    r"中心|园区|大厦|酒店|银行|大学|学院|医院|工厂|办公室)"
)
# 职务锚定的中文人名：捕获职务前的 2~3 字姓名
_NAME_RE = re.compile(
    r"([一-鿿]{2,3})(?:董事长|副总经理|总经理|副总|总监|部长|处长|"
    r"科长|组长|主管|经理|主任|工程师|老师|先生|女士|同志)"
)


def _proper_noun_candidates(text: str) -> set[str]:
    """从范文中抽取"专名形状"候选（引号内容 / 带机构后缀实体 / 职务锚定人名）。"""
    cands: set[str] = set()
    for m in _QUOTE_RE.finditer(text):
        cands.add(m.group(1))
    for m in _ENTITY_RE.finditer(text):
        cands.add(m.group(0))
    for m in _NAME_RE.finditer(text):
        cands.add(m.group(1))
    return cands

# 判定失败时给裁判的转录节选上限（避免"全文裁判"）
_JUDGE_EXCERPT_CHARS = 6000


class JudgeVerdict(BaseModel):
    confirmed: list[str] = []


@dataclass
class QualityReport:
    # 高风险：触发一次重跑
    invalid_todo_seqs: list[int] = field(default_factory=list)
    example_leaks: list[str] = field(default_factory=list)
    ungrounded_owner_or_deadline: list[str] = field(default_factory=list)
    # 疑似：仅告警
    ungrounded_numbers: list[str] = field(default_factory=list)
    regenerated: bool = False

    @property
    def has_high_risk(self) -> bool:
        return bool(
            self.invalid_todo_seqs
            or self.example_leaks
            or self.ungrounded_owner_or_deadline
        )

    @property
    def is_empty(self) -> bool:
        return not (
            self.invalid_todo_seqs
            or self.example_leaks
            or self.ungrounded_owner_or_deadline
            or self.ungrounded_numbers
        )

    def fuzzy_suspects(self) -> list[str]:
        """交给 LLM 裁判复核的模糊项（不含确定性事实 invalid_todo_seqs）。"""
        return [
            *self.example_leaks,
            *self.ungrounded_owner_or_deadline,
            *self.ungrounded_numbers,
        ]

    def to_meta(self) -> dict:
        out: dict = {}
        if self.invalid_todo_seqs:
            out["invalid_todo_seqs"] = self.invalid_todo_seqs
        if self.example_leaks:
            out["example_leaks"] = self.example_leaks
        if self.ungrounded_owner_or_deadline:
            out["ungrounded_owner_or_deadline"] = self.ungrounded_owner_or_deadline
        if self.ungrounded_numbers:
            out["ungrounded_numbers"] = self.ungrounded_numbers
        if self.regenerated:
            out["regenerated"] = True
        return out


def _summary_body(content: SummaryContent) -> str:
    """人读正文（不含 source_segment_seq 整数——那是引用而非事实数字）。"""
    parts = [content.title, content.summary, *content.discussions, *content.decisions]
    for t in content.topics:
        parts.append(t.title)
        parts.extend(t.items)
    for todo in content.todos:
        parts.extend(filter(None, [todo.task, todo.owner, todo.deadline]))
    return "\n".join(parts)


def _grounded(text: str, transcript: str) -> bool:
    """有依据判定：整体精确命中走快速路径；否则拆分后**每个**非空分量都须
    在逐字稿出现（多责任人"张三、李四"里只要有一人是虚构的就判为无依据）。"""
    if text in transcript:
        return True
    parts = [p for p in _OWNER_SPLIT.split(text) if p]
    return bool(parts) and all(p in transcript for p in parts)


def check_rules(
    content: SummaryContent,
    transcript_text: str,
    segment_seqs: set[int],
    examples: list[str] | None,
) -> QualityReport:
    """确定性启发式检查；返回各类疑似项（空报告 = 干净）。"""
    report = QualityReport()
    body = _summary_body(content)

    # 1) 无效引用：todo 溯源到不存在的 segment（确定性事实）
    report.invalid_todo_seqs = [
        t.source_segment_seq
        for t in content.todos
        if t.source_segment_seq is not None and t.source_segment_seq not in segment_seqs
    ]

    # 2) 范文污染：范例独有（逐字稿没有）的事实型记号泄漏进纪要。
    # 候选 = 拉丁词/数字（项目代号、KPI 值）+ 中文专名（引号名/机构实体/职务人名）；
    # 不做全量中文切词，避免风格词（"同步推进"等）误报
    example_tokens: set[str] = set()
    for ex in examples or []:
        for tok in set(_ALNUM_RE.findall(ex)) | _proper_noun_candidates(ex):
            if tok not in transcript_text:
                example_tokens.add(tok)
    report.example_leaks = sorted(t for t in example_tokens if t in body)

    # 3) owner/deadline 无原文依据（高风险：责任归属/时限是纪要关键信息）
    for todo in content.todos:
        if todo.owner and not _grounded(todo.owner, transcript_text):
            report.ungrounded_owner_or_deadline.append(f"owner:{todo.owner}")
        if todo.deadline and not _grounded(todo.deadline, transcript_text):
            report.ungrounded_owner_or_deadline.append(f"deadline:{todo.deadline}")

    # 4) 未溯源数字（疑似，仅告警）：正文阿拉伯数字逐字稿未见、且非范文泄漏
    leaked = set(report.example_leaks)
    report.ungrounded_numbers = sorted(
        {
            n
            for n in _NUM_RE.findall(body)
            if n not in transcript_text and n not in leaked
        }
    )
    return report


def _evidence_excerpt(suspects: list[str], transcript_text: str) -> str:
    """为裁判挑选与可疑项相关的证据窗口（含该关键词的转录行），而非一律取开头。

    可疑项形如 "owner:张三"/"deadline:下周三"/"45%"/"Alpha"——取冒号后的关键词
    匹配转录行；无命中则回退到开头节选，整体不超过上限（避免"全文裁判"）。
    """
    terms = [s.split(":", 1)[-1] for s in suspects]
    hits = [
        ln
        for ln in transcript_text.split("\n")
        if any(t and t in ln for t in terms)
    ]
    excerpt = "\n".join(hits) if hits else transcript_text
    return excerpt[:_JUDGE_EXCERPT_CHARS]


async def _judge(suspects: list[str], transcript_text: str) -> set[str] | None:
    """LLM 裁判复核可疑项，返回确认为真问题的子集；失败返回 None（fail-open）。"""
    # 延迟导入：避免与 llm 包的加载顺序耦合
    from app.services.llm import LLMTaskType, build_router
    from app.services.llm.prompts import SYSTEM_QUALITY_JUDGE, quality_judge_prompt

    try:
        verdict, _ = await build_router(LLMTaskType.QUALITY_CHECK).generate_json(
            SYSTEM_QUALITY_JUDGE,
            quality_judge_prompt(suspects, _evidence_excerpt(suspects, transcript_text)),
            JudgeVerdict,
        )
        return set(verdict.confirmed)
    except Exception:
        logger.warning("quality judge unavailable, fail-open to rules result", exc_info=True)
        return None


def _is_high_value(meeting) -> bool:
    imp = (getattr(meeting, "importance", None) or "").strip().lower()
    return imp in {"high", "高", "重要", "critical"}


async def run_quality_check(
    content: SummaryContent,
    transcript_text: str,
    segment_seqs: set[int],
    examples: list[str] | None,
    mode: str | None = None,
    meeting=None,
) -> QualityReport:
    """按 SUMMARY_QUALITY_CHECK_MODE 分派 off/rules/hybrid。"""
    mode = mode or settings.summary_quality_check_mode
    if mode == "off":
        return QualityReport()

    report = check_rules(content, transcript_text, segment_seqs, examples)

    # hybrid：仅在规则命中可疑条目（或高价值纪要）时用裁判过滤误报，非全文
    if mode == "hybrid" and (report.fuzzy_suspects() or _is_high_value(meeting)):
        suspects = report.fuzzy_suspects()
        if suspects:
            confirmed = await _judge(suspects, transcript_text)
            if confirmed is not None:  # None = 裁判不可用，保留规则结果
                report.example_leaks = [
                    x for x in report.example_leaks if x in confirmed
                ]
                report.ungrounded_owner_or_deadline = [
                    x for x in report.ungrounded_owner_or_deadline if x in confirmed
                ]
                report.ungrounded_numbers = [
                    x for x in report.ungrounded_numbers if x in confirmed
                ]
    return report
