"""prompt 统一管理（PRD §4：LLM service 职责）。

溯源硬约束（PRD Feature 3）：LLM 引用 segment 编号（seq）而非复述原文，
程序反查真实文本与时间戳，杜绝幻觉引用。
"""

SYSTEM_SUMMARIZER = (
    "你是一名专业的会议纪要助手。你只输出合法的 JSON 对象，"
    "不输出 markdown、解释或任何其他内容。"
    "引用会议内容时必须使用行首方括号内的 segment 编号（seq），不要复述原文位置。"
)

_SCHEMA_DESC = """输出 JSON 对象，字段如下：
{
  "title": "会议主题（简短，格式如'每日例会：工作安排及事务沟通会'）",
  "participants": ["参会人（用转录中出现的名字或 speaker 标签）"],
  "summary": "会议总结（一段话）",
  "topics": [
    {
      "title": "议题名称（如'酒店运营考核制度优化'）",
      "owner": "该议题责任人（多人用顿号分隔；不明确则为 null）",
      "items": ["条目 1", "条目 2"]
    }
  ],
  "decisions": ["决策事项（无则空数组）"],
  "todos": [
    {
      "task": "待办事项",
      "owner": "负责人（转录中的名字或 speaker 标签，不明确则为 null）",
      "deadline": "截止时间原文（如 '周五'，不明确则为 null）",
      "source_segment_seq": 该 TODO 来源的 segment 编号（整数，务必给出）
    }
  ]
}"""

# 公司纪要文风第一层：从既有纪要范本蒸馏的硬规则
_STYLE_RULES = """正文写作规范（严格遵守）：
1. 按议题分组（topics），每个议题标注责任人；同类事项归入同一议题，一场会议通常 3~6 个议题
2. 条目用行动式短句：事项 + 要求 + 时限/标准，一到两句一条，每个议题 2~4 条；
   示例：'资产盘点工作加快推进，外勤人员优先打车往返，提升盘点效率。'
3. 惯用语汇参照：'同步推进''及时上报''纳入考核''尽快落地执行''严格执行''择期安排'
4. 无法归组的零散要求收进末尾议题'其他工作要求'（无则不加）
5. 只写结论与要求，不写讨论过程、口语原话和车轱辘话"""


def _examples_block(examples: list[str] | None) -> str:
    """公司纪要文风第二层：范例库 few-shot。

    只注入最终产出阶段（single_pass / reduce），map 阶段不注入——
    文风由最后一步定型，中间纪要多带范例徒耗 token。
    """
    if not examples:
        return ""
    parts = "\n\n".join(
        f"--- 范例 {i + 1} ---\n{e}" for i, e in enumerate(examples)
    )
    return (
        "\n\n以下是本公司既往会议纪要范例。请模仿其文风：议题命名方式、"
        "条目的句式与详略、惯用语汇。注意：范例只用于学习文风，"
        "其中的事项、人名、数字严禁出现在本次纪要中；输出仍是上述 JSON 格式，"
        "范例正文对应 topics 字段的内容。\n\n" + parts
    )


def single_pass_prompt(transcript: str, examples: list[str] | None = None) -> str:
    return (
        f"以下是一场会议的完整转录，每行格式为 [seq] [时间] 说话人: 内容。\n\n"
        f"{transcript}\n\n请生成结构化会议纪要。{_SCHEMA_DESC}\n\n{_STYLE_RULES}"
        f"{_examples_block(examples)}"
    )


# 长会议 map 阶段的事实抽取系统提示。"事实抽取器" 是稳定标记，供 mock 识别。
SYSTEM_MAP_EXTRACT = (
    "你是一名会议事实抽取器。你只输出合法的 JSON 对象，不输出 markdown 或其他内容。"
    "你的任务是从会议片段中抽取结构化事实，不做文风润色、不写会议总结叙述。"
    "引用会议内容时必须使用行首方括号内的 segment 编号（seq），不要复述原文位置。"
)

# map 精简事实 schema（对应 MapFacts）：不含 title/participants/summary 叙述，省 token
_MAP_SCHEMA_DESC = """输出 JSON 对象，字段如下（只抽取本部分出现的事实，不做润色）：
{
  "topics": [
    {"title": "议题名称", "owner": "责任人（不明确则 null）", "items": ["要点1", "要点2"]}
  ],
  "decisions": ["本部分的决策事项（无则空数组）"],
  "todos": [
    {
      "task": "待办事项",
      "owner": "负责人（转录中的名字或 speaker 标签，不明确则 null）",
      "deadline": "截止时间原文（如 '周五'，不明确则 null）",
      "source_segment_seq": 该 TODO 来源的 segment 编号（整数，务必给出）
    }
  ],
  "risks": ["提到的风险或隐患（无则空数组）"],
  "open_questions": ["尚未有结论的待明确问题（无则空数组）"],
  "source_segment_seqs": [本部分涉及的关键 segment 编号列表]
}"""


def map_prompt(transcript_chunk: str, part: int, total: int) -> str:
    return (
        f"以下是一场长会议转录的第 {part}/{total} 部分，"
        f"每行格式为 [seq] [时间] 说话人: 内容。\n\n{transcript_chunk}\n\n"
        f"请抽取本部分的结构化事实。{_MAP_SCHEMA_DESC}\n\n"
        "注意：只抽取本部分内容，不要润色文风、不要写会议总结叙述；"
        "TODO 与决策可能跨部分延续，宁可多保留候选。"
    )


def reduce_prompt(
    partial_summaries: list[str], examples: list[str] | None = None
) -> str:
    parts = "\n\n".join(
        f"--- 第 {i + 1} 部分事实 ---\n{p}" for i, p in enumerate(partial_summaries)
    )
    return (
        f"以下是同一场长会议按顺序分段抽取的结构化事实（JSON，每段为 MapFacts）：\n\n"
        f"{parts}\n\n"
        f"请据此合成一份完整的最终会议纪要。要求：\n"
        f"- 去重合并：相邻部分因切块重叠会重复同一议题/TODO/决策，务必合并为一条，"
        f"不得让同一事实在最终纪要里重复出现；\n"
        f"- 同一责任人的相关事项归入同一议题；\n"
        f"- 跨段延续的 TODO 与决策不得丢失，保留原有的 source_segment_seq 引用；\n"
        f"- risks 与 open_questions 折入相关议题或会议总结，不要单独成字段、也不得丢弃。\n"
        f"{_SCHEMA_DESC}\n\n{_STYLE_RULES}{_examples_block(examples)}"
    )


SYSTEM_PLAIN = (
    "你是一名专业的会议纪要助手。请用简洁的中文输出一段纯文本会议纪要，"
    "包含主要讨论内容、决策与待办事项。"
)


def plain_text_prompt(transcript: str) -> str:
    return f"以下是一场会议的完整转录：\n\n{transcript}\n\n请生成纯文本会议纪要。"


# AI 会议问答（RAG，PRD Feature 5）。"会议问答助手" 是稳定标记，供 mock 识别。
# 最小证据不足拒答（RAG 设计 §4.1.1-F）：宁可拒答，不得凭常识补全。
SYSTEM_QA = (
    "你是一名会议问答助手。你只输出合法的 JSON 对象，不输出 markdown 或其他内容。"
    "只依据本轮给定的会议转录片段回答问题，不得臆造未出现的信息；"
    "禁止根据常识、会议标题、历史回答或任何片段之外的内容补全事实。"
    "日期、金额、负责人、版本号和最终决策必须有直接的片段引用依据。"
    "引用依据时使用片段行首方括号内的 segment 编号（seq）。"
    "若给定片段不足以回答，回答'会议原文中没有找到明确结论'，"
    "并置 insufficient_evidence 为 true、引用留空。"
)

_QA_SCHEMA_DESC = """输出 JSON 对象，字段如下：
{
  "answer": "对问题的回答（简洁中文；证据不足则回答'会议原文中没有找到明确结论'）",
  "cited_segment_seqs": [引用到的 segment 编号（整数）列表，无则空数组],
  "insufficient_evidence": 布尔值（片段不足以回答时为 true，此时引用必须为空数组）,
  "confidence": "high|medium|low（证据充分且直接为 high；有依据但需少量推断为 medium；证据不足为 low）"
}"""


def qa_prompt(question: str, segment_lines: str) -> str:
    return (
        f"以下是与问题相关的会议转录片段，每行格式为 [seq] [时间] 说话人: 内容。\n\n"
        f"{segment_lines}\n\n问题：{question}\n\n"
        f"请依据上述片段回答。{_QA_SCHEMA_DESC}"
    )


# 聊天意图路由 + 多轮问题改写（PRD §7.1 / RAG 设计 §4.1.1-A：一次调用同时产出）。
# "意图分类器" 是稳定标记，供 mock 识别。
SYSTEM_INTENT = (
    "你是一个意图分类器与问题改写器。你只输出合法的 JSON 对象。"
    "任务一：判断用户在会议助手聊天框里发的这条消息意图——"
    "查询会议内容（query）还是要求修改会议纪要（edit）。"
    "只有明确要求改动纪要内容（增删改措辞、合并议题、调整 TODO 等）才算 edit；"
    "提问、追溯、总结类一律 query。"
    "任务二（仅当给出对话历史时）：把当前问题改写为一个不依赖上下文、"
    "可独立检索的完整问题（standalone_query），解析其中的代词与省略指代。"
    "改写只能使用历史用户消息、引用原文和说话人名单中出现过的人名与事实，"
    "不得虚构；助手历史回答只用于理解指代，不作为人名、日期、金额或结论的事实来源。"
    "无需改写或没有历史时，standalone_query 返回原问题。"
    '输出 {"intent": "query"|"edit", "standalone_query": "..."}'
)


def intent_prompt(message: str) -> str:
    return f"当前问题：{message}"


def intent_rewrite_prompt(
    message: str,
    recent_user_messages: list[str],
    speaker_names: list[str],
    cited_texts: list[str] | None = None,
) -> str:
    """多轮场景的合并意图+改写 prompt（§4.1.1-B）。

    cited_texts 仅在当前问题含指代表达时传入（上一轮合法引用的原文），
    作为解析指代的事实语境。
    """
    parts = ["以下是本次会议聊天的上下文，用于解析当前问题中的指代："]
    if recent_user_messages:
        history = "\n".join(f"- {m}" for m in recent_user_messages)
        parts.append(f"最近的用户提问（从旧到新）：\n{history}")
    if speaker_names:
        parts.append("本会议的说话人名单：" + "、".join(speaker_names))
    if cited_texts:
        quotes = "\n".join(f"> {t}" for t in cited_texts)
        parts.append(f"上一轮回答引用的会议原文（事实语境）：\n{quotes}")
    parts.append(f"当前问题：{message}")
    return "\n\n".join(parts)


# 聊天改纪要（PRD §7.1 步骤 2）。"纪要编辑器" 是稳定标记，供 mock 识别。
SYSTEM_SUMMARY_EDIT = (
    "你是一名会议纪要编辑器。你只输出合法的 JSON 对象，不输出 markdown 或其他内容。"
    "根据用户指令在现有纪要基础上修改：只改动指令涉及的部分，"
    "未提及的内容必须原样保留（含 source_segment_seq 引用）；"
    "不得虚构会议中未出现的信息。"
)


def summary_edit_prompt(current_json: str, instruction: str) -> str:
    return (
        f"现有会议纪要（JSON）：\n\n{current_json}\n\n"
        f"用户修改指令：{instruction}\n\n"
        f"请输出修改后的完整纪要。{_SCHEMA_DESC}\n\n{_STYLE_RULES}"
    )


# 纪要质量裁判（Phase 4 hybrid 模式）。"质量审查员" 是稳定标记，供 mock 识别。
# 只对规则已命中的可疑条目做二次判定（非全文），降低确定性规则的误报。
SYSTEM_QUALITY_JUDGE = (
    "你是一名会议纪要质量审查员。你只输出合法的 JSON 对象，不输出其他内容。"
    "给你若干条从纪要中挑出的可疑条目，以及相关的会议转录片段。"
    "判断每条可疑项是否为真实问题（纪要中出现了转录里没有依据的内容，"
    "即幻觉或范文文风污染），还是其实有原文依据（误报）。"
)

_JUDGE_SCHEMA_DESC = """输出 JSON 对象，字段如下：
{
  "confirmed": [被确认为真实问题的可疑项原文列表（无则空数组）]
}"""


def quality_judge_prompt(suspect_items: list[str], transcript_excerpt: str) -> str:
    items = "\n".join(f"- {s}" for s in suspect_items)
    return (
        f"可疑条目：\n{items}\n\n"
        f"相关会议转录片段（每行 [seq] [时间] 说话人: 内容）：\n\n{transcript_excerpt}\n\n"
        f"请判断哪些可疑条目确属无原文依据的问题。{_JUDGE_SCHEMA_DESC}"
    )
