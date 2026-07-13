"""Summary content_json → 公司格式纪要纯文本。

用途：「采纳为范例」时把结构化纪要渲染成范例库存储的文本形态，
与 Word 导出正文保持同一版式（议题分组 + 中文序号 + 编号条目），
few-shot 注入的范例即模型被要求模仿的最终形态。
"""
from typing import Any

from app.services.word_export import _body_topics, _cn_num


def render_summary_text(content: dict[str, Any]) -> str:
    """结构化纪要 → 公司格式正文文本；降级纪要（纯文本）原样返回。"""
    if content.get("text") and not content.get("topics") and not content.get(
        "discussions"
    ):
        return str(content["text"])

    lines: list[str] = []
    if content.get("title"):
        lines.append(f"会议主题：{content['title']}")
    if content.get("participants"):
        lines.append(f"参会人员：{'、'.join(content['participants'])}")
    if content.get("summary"):
        lines.append(f"会议总结：{content['summary']}")
    lines.append("会议主要内容：")
    for i, group in enumerate(_body_topics(content)):
        owner_suffix = f"（责任人：{group['owner']}）" if group.get("owner") else ""
        lines.append(f"{_cn_num(i)}、{group['title']}{owner_suffix}")
        for j, item in enumerate(group["items"], start=1):
            lines.append(f"{j}．{item}")
    return "\n".join(lines)
