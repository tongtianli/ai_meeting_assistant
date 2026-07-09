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
  "title": "会议主题（简短）",
  "participants": ["参会人（用转录中出现的名字或 speaker 标签）"],
  "summary": "会议总结（一段话）",
  "discussions": ["讨论事项 1", "讨论事项 2"],
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


def single_pass_prompt(transcript: str) -> str:
    return (
        f"以下是一场会议的完整转录，每行格式为 [seq] [时间] 说话人: 内容。\n\n"
        f"{transcript}\n\n请生成结构化会议纪要。{_SCHEMA_DESC}"
    )


def map_prompt(transcript_chunk: str, part: int, total: int) -> str:
    return (
        f"以下是一场长会议转录的第 {part}/{total} 部分，"
        f"每行格式为 [seq] [时间] 说话人: 内容。\n\n{transcript_chunk}\n\n"
        f"请提取本部分的阶段性纪要。{_SCHEMA_DESC}\n"
        "注意：只总结本部分内容；TODO 与决策可能跨部分延续，宁可多保留候选。"
    )


def reduce_prompt(partial_summaries: list[str]) -> str:
    parts = "\n\n".join(
        f"--- 第 {i + 1} 部分纪要 ---\n{p}" for i, p in enumerate(partial_summaries)
    )
    return (
        f"以下是同一场长会议按顺序分段生成的阶段性纪要（JSON）：\n\n{parts}\n\n"
        f"请合并为一份完整的最终会议纪要：去重、合并同类项，"
        f"跨段延续的 TODO 与决策不得丢失，保留原有的 source_segment_seq 引用。"
        f"{_SCHEMA_DESC}"
    )


SYSTEM_PLAIN = (
    "你是一名专业的会议纪要助手。请用简洁的中文输出一段纯文本会议纪要，"
    "包含主要讨论内容、决策与待办事项。"
)


def plain_text_prompt(transcript: str) -> str:
    return f"以下是一场会议的完整转录：\n\n{transcript}\n\n请生成纯文本会议纪要。"
