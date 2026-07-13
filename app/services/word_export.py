"""Word 导出：Summary 的 content_json 填公司纪要模板（PRD Feature 6）。

纯程序步骤，与 LLM 完全解耦；用户点击导出时实时渲染，不占管道。
负责人名字在导出时通过来源 segment 的 person_id 现算——
Speaker 重命名后无需重新摘要，导出立即用真名。

模板为公司表格式公文（会议主题/时间/地点/主持/参会/记录人/重要程度抬头
+「会议主要内容：」正文按议题分组），由 scripts/generate_word_template.py 生成。
"""
import io
from datetime import datetime
from pathlib import Path
from typing import Any

from docxtpl import DocxTemplate

TEMPLATE_PATH = Path(__file__).resolve().parent.parent / "templates" / "summary_template.docx"

_CN_NUMS = "一二三四五六七八九十"
_CN_WEEKDAYS = "一二三四五六日"


def _cn_num(i: int) -> str:
    """0-based 序号 → 中文序号（超出 20 个议题的会议不存在于现实中）。"""
    if i < 10:
        return _CN_NUMS[i]
    return "十" + (_CN_NUMS[i - 11] if i > 10 else "")


def _fmt_meeting_time(iso: str) -> str:
    dt = datetime.fromisoformat(iso)
    return (
        f"{dt.year}年{dt.month}月{dt.day}日"
        f"星期{_CN_WEEKDAYS[dt.weekday()]}  {dt.hour}:{dt.minute:02d}"
    )


def _fmt_ts(seconds: float) -> str:
    total = int(seconds)
    return f"{total // 3600:02d}:{total % 3600 // 60:02d}:{total % 60:02d}"


def _body_topics(content: dict[str, Any]) -> list[dict[str, Any]]:
    """content_json → 正文议题分组；兼容旧版（无 topics）已存 JSON。"""
    groups: list[dict[str, Any]] = []
    for t in content.get("topics") or []:
        groups.append(
            {"title": t.get("title", ""), "owner": t.get("owner"), "items": t.get("items") or []}
        )
    if not groups and content.get("discussions"):
        # 旧结构：平铺讨论事项归入单一议题
        groups.append({"title": "会议讨论事项", "owner": None, "items": content["discussions"]})
    if content.get("decisions"):
        groups.append({"title": "会议决定事项", "owner": None, "items": content["decisions"]})
    return groups


def build_context(
    content: dict[str, Any],
    todo_rows: list[tuple[Any, float | None, str | None]],
    meeting: Any = None,
) -> dict[str, Any]:
    """content_json + (ActionItem, 来源时间戳, 负责人真名) + Meeting → 模板上下文。"""
    meta = content.get("_meta", {})
    degraded = bool(meta.get("degraded"))
    meeting_time = ""
    if meta.get("meeting_time"):
        meeting_time = _fmt_meeting_time(meta["meeting_time"])

    todos = [
        {
            "task": item.task,
            "owner": person_name or item.owner_text or "待定",
            "deadline": item.deadline or "",
            "source": _fmt_ts(start_time) if start_time is not None else "",
        }
        for item, start_time, person_name in todo_rows
    ]

    # 键名用 entries 而非 items：Jinja 的 t.items 会解析成 dict.items 方法
    topics = [
        {
            "num": _cn_num(i),
            "title": g["title"],
            "owner_suffix": f"（责任人：{g['owner']}）" if g.get("owner") else "",
            "entries": g["items"],
        }
        for i, g in enumerate(_body_topics(content))
    ]
    # 待办汇总作为末尾议题（公司格式：正文条目自带责任人与时限）
    if todos:
        todo_lines = []
        for t in todos:
            suffix = f"（负责人：{t['owner']}"
            suffix += f"，截止：{t['deadline']}" if t["deadline"] else ""
            suffix += f"，出处 {t['source']}" if t["source"] else ""
            suffix += "）"
            todo_lines.append(f"{t['task']}{suffix}")
        topics.append(
            {
                "num": _cn_num(len(topics)),
                "title": "待办事项汇总",
                "owner_suffix": "",
                "entries": todo_lines,
            }
        )

    importance = getattr(meeting, "importance", None) if meeting else None
    return {
        "degraded": degraded,
        "title": content.get("title", "会议纪要"),
        "meeting_time": meeting_time,
        "location": (getattr(meeting, "location", None) if meeting else None) or "",
        "host": (getattr(meeting, "host", None) if meeting else None) or "",
        "recorder": (getattr(meeting, "recorder", None) if meeting else None) or "",
        # 未指定时三档并列印出，打印后手工圈选（公司模板惯例）
        "importance_line": importance or "一般     重要     加急",
        "participants": "、".join(content.get("participants", [])),
        "plain_text": content.get("text", ""),
        "summary": content.get("summary", ""),
        "topics": topics,
        # 旧模板字段保留（自定义模板可能仍引用）
        "discussions": content.get("discussions", []),
        "decisions": content.get("decisions", []),
        "todos": todos,
    }


def render_summary_docx(context: dict[str, Any]) -> bytes:
    tpl = DocxTemplate(TEMPLATE_PATH)
    tpl.render(context, autoescape=True)
    buf = io.BytesIO()
    tpl.save(buf)
    return buf.getvalue()
