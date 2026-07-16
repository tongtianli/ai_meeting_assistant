"""聊天意图识别 + 多轮问题改写（RAG 设计 §4.1.1-A/B）。

- 意图分类与 standalone query 改写合并为**一次** LLM 调用，不新增独立改写热路径；
- 无历史消息：不做改写（standalone = 原问题），仍需一次意图分类；
- 指代门控：仅当当前问题含指代表达时，才附带上一轮合法引用的少量原文
  （数量与字符双预算），作为解析指代的事实语境；
- 失败降级：intent="query"、standalone=原问题——查询无副作用，绝不阻塞问答。
"""
import logging
import re
from dataclasses import dataclass

from app.core.config import settings
from app.schemas.chat import ChatIntent
from app.services.llm import LLMExhaustedError, LLMTaskType, build_router
from app.services.llm.prompts import (
    SYSTEM_INTENT,
    intent_prompt,
    intent_rewrite_prompt,
)

logger = logging.getLogger(__name__)

# 指代表达门控（§4.1.1-B）：命中才附带上一轮引用原文。多字词在前防止被单字抢先匹配
_ANAPHORA_RE = re.compile(
    r"他们|她们|它们|这件事|那件事|此事|该方案|这个方案|那个方案|上面说的|"
    r"刚才|之前说|后来|这个|那个|他|她|它"
)


@dataclass
class IntentResult:
    intent: str  # "query" | "edit"
    standalone_query: str  # 恒非空；未改写时 = 原问题
    rewritten: bool  # standalone 是否不同于原问题
    used_cited_context: bool  # 是否附带了上一轮引用原文


def has_anaphora(question: str) -> bool:
    return bool(_ANAPHORA_RE.search(question))


def _budget_cited(cited_texts: list[str] | None) -> list[str]:
    """按条数 + 字符双预算截断上一轮引用原文，防 prompt 膨胀。"""
    if not cited_texts:
        return []
    out: list[str] = []
    budget = settings.qa_rewrite_cited_max_chars
    for text in cited_texts[: settings.qa_rewrite_cited_max_segments]:
        if len(text) > budget:
            break
        out.append(text)
        budget -= len(text)
    return out


async def classify_and_rewrite(
    question: str,
    recent_user_messages: list[str],
    speaker_names: list[str],
    cited_texts: list[str] | None = None,
) -> IntentResult:
    """一次调用产出意图与可独立检索的问题；失败降级为 query + 原问题。"""
    used_cited = False
    if recent_user_messages:
        cited = _budget_cited(cited_texts) if has_anaphora(question) else []
        used_cited = bool(cited)
        prompt = intent_rewrite_prompt(
            question,
            recent_user_messages[-settings.qa_rewrite_recent_messages :],
            speaker_names,
            cited or None,
        )
    else:
        # 首轮：无需改写，也不为"问题是否完整"多打一次模型
        prompt = intent_prompt(question)
    try:
        parsed, _ = await build_router(LLMTaskType.INTENT_CLASSIFY).generate_json(
            SYSTEM_INTENT, prompt, ChatIntent
        )
    except LLMExhaustedError as exc:
        logger.warning("intent/rewrite failed, fallback to query + original: %s", exc)
        return IntentResult("query", question, False, False)
    standalone = (parsed.standalone_query or "").strip() or question
    if not recent_user_messages:
        standalone = question  # 无历史时改写无意义，防模型自作主张
    return IntentResult(
        intent=parsed.intent,
        standalone_query=standalone,
        rewritten=standalone != question,
        used_cited_context=used_cited,
    )
