"""生成默认 Word 纪要模板（docxtpl/jinja2 标签）。

公司固定模板落地时直接替换 app/templates/summary_template.docx 即可，
标签写法见 docxtpl 文档；本脚本保证默认模板可复现、可调整。
用法：uv run python scripts/generate_word_template.py
"""
from pathlib import Path

from docx import Document
from docx.shared import Pt

OUT = Path(__file__).resolve().parent.parent / "app" / "templates" / "summary_template.docx"


def main() -> None:
    doc = Document()
    doc.add_heading("{{ title }}", 0)
    meta = doc.add_paragraph()
    meta.add_run("会议时间：{{ meeting_time }}    参会人员：{{ participants }}").font.size = Pt(10)

    # 降级（纯文本）分支
    doc.add_paragraph("{%p if degraded %}")
    doc.add_heading("会议纪要", level=1)
    doc.add_paragraph("{{ plain_text }}")
    doc.add_paragraph("{%p endif %}")

    # 结构化分支
    doc.add_paragraph("{%p if not degraded %}")
    doc.add_heading("会议总结", level=1)
    doc.add_paragraph("{{ summary }}")

    doc.add_heading("讨论事项", level=1)
    doc.add_paragraph("{%p for d in discussions %}")
    doc.add_paragraph("{{ loop.index }}. {{ d }}")
    doc.add_paragraph("{%p endfor %}")

    doc.add_heading("决策事项", level=1)
    doc.add_paragraph("{%p for d in decisions %}")
    doc.add_paragraph("{{ loop.index }}. {{ d }}")
    doc.add_paragraph("{%p endfor %}")

    doc.add_heading("TODO", level=1)
    table = doc.add_table(rows=4, cols=4)
    table.style = "Table Grid"
    header = table.rows[0].cells
    for cell, text in zip(header, ("事项", "负责人", "截止时间", "来源")):
        cell.text = text
    table.rows[1].cells[0].text = "{%tr for t in todos %}"
    body = table.rows[2].cells
    body[0].text = "{{ t.task }}"
    body[1].text = "{{ t.owner }}"
    body[2].text = "{{ t.deadline }}"
    body[3].text = "{{ t.source }}"
    table.rows[3].cells[0].text = "{%tr endfor %}"
    doc.add_paragraph("{%p endif %}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(OUT)
    print(f"template written: {OUT}")


if __name__ == "__main__":
    main()
