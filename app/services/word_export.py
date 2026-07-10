"""Word 导出：Summary 的 content_json 填 docxtpl 模板（PRD Feature 6）。

纯程序步骤，与 LLM 完全解耦；用户点击导出时实时渲染，不占管道。
负责人名字在导出时通过来源 segment 的 person_id 现算——
Speaker 重命名后无需重新摘要，导出立即用真名。
"""
import io
from datetime import datetime
from pathlib import Path
from typing import Any

from docxtpl import DocxTemplate

TEMPLATE_PATH = Path(__file__).resolve().parent.parent / "templates" / "summary_template.docx"


def _fmt_ts(seconds: float) -> str:
    total = int(seconds)
    return f"{total // 3600:02d}:{total % 3600 // 60:02d}:{total % 60:02d}"


def build_context(
    content: dict[str, Any],
    todo_rows: list[tuple[Any, float | None, str | None]],
) -> dict[str, Any]:
    """content_json + (ActionItem, 来源时间戳, 负责人真名) → 模板上下文。"""
    meta = content.get("_meta", {})
    degraded = bool(meta.get("degraded"))
    meeting_time = ""
    if meta.get("meeting_time"):
        meeting_time = datetime.fromisoformat(meta["meeting_time"]).strftime(
            "%Y-%m-%d %H:%M"
        )
    return {
        "degraded": degraded,
        "title": content.get("title", "会议纪要"),
        "meeting_time": meeting_time,
        "participants": "、".join(content.get("participants", [])) or "—",
        "plain_text": content.get("text", ""),
        "summary": content.get("summary", ""),
        "discussions": content.get("discussions", []),
        "decisions": content.get("decisions", []),
        "todos": [
            {
                "task": item.task,
                "owner": person_name or item.owner_text or "待定",
                "deadline": item.deadline or "—",
                "source": _fmt_ts(start_time) if start_time is not None else "—",
            }
            for item, start_time, person_name in todo_rows
        ],
    }


def render_summary_docx(context: dict[str, Any]) -> bytes:
    tpl = DocxTemplate(TEMPLATE_PATH)
    tpl.render(context, autoescape=True)
    buf = io.BytesIO()
    tpl.save(buf)
    return buf.getvalue()
