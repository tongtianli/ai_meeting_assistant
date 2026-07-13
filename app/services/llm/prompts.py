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


def map_prompt(transcript_chunk: str, part: int, total: int) -> str:
    return (
        f"以下是一场长会议转录的第 {part}/{total} 部分，"
        f"每行格式为 [seq] [时间] 说话人: 内容。\n\n{transcript_chunk}\n\n"
        f"请提取本部分的阶段性纪要。{_SCHEMA_DESC}\n\n{_STYLE_RULES}\n"
        "注意：只总结本部分内容；TODO 与决策可能跨部分延续，宁可多保留候选。"
    )


def reduce_prompt(
    partial_summaries: list[str], examples: list[str] | None = None
) -> str:
    parts = "\n\n".join(
        f"--- 第 {i + 1} 部分纪要 ---\n{p}" for i, p in enumerate(partial_summaries)
    )
    return (
        f"以下是同一场长会议按顺序分段生成的阶段性纪要（JSON）：\n\n{parts}\n\n"
        f"请合并为一份完整的最终会议纪要：去重、合并同类议题（同一责任人的"
        f"相关事项归入同一议题），跨段延续的 TODO 与决策不得丢失，"
        f"保留原有的 source_segment_seq 引用。"
        f"{_SCHEMA_DESC}\n\n{_STYLE_RULES}{_examples_block(examples)}"
    )


SYSTEM_PLAIN = (
    "你是一名专业的会议纪要助手。请用简洁的中文输出一段纯文本会议纪要，"
    "包含主要讨论内容、决策与待办事项。"
)


def plain_text_prompt(transcript: str) -> str:
    return f"以下是一场会议的完整转录：\n\n{transcript}\n\n请生成纯文本会议纪要。"
