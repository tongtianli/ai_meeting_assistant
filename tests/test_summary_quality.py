"""纪要质量诊断（Phase 4）：确定性规则分级 + hybrid 裁判过滤/fail-open。

纯逻辑 + mock LLM，不依赖数据库（record_usage 无库时自吞）。
"""
import asyncio

from app.schemas.summary import SummaryContent, TodoItem, TopicGroup
from app.services.summary_quality import check_rules, run_quality_check


def _content(**kw) -> SummaryContent:
    base = dict(title="会议", participants=[], summary="", topics=[], decisions=[], todos=[])
    base.update(kw)
    return SummaryContent(**base)


# ---- 确定性规则 ----


def test_clean_summary_no_flags() -> None:
    c = _content(
        summary="讨论了项目进展",
        topics=[TopicGroup(title="进展", items=["接口完成八成"])],
    )
    r = check_rules(c, "接口完成八成，项目进展顺利", {0, 1}, None)
    assert r.is_empty and not r.has_high_risk


def test_ungrounded_arabic_number_is_suspect_only() -> None:
    c = _content(summary="销售额增长 45%")
    r = check_rules(c, "会议讨论了销售情况", {0}, None)
    assert any("45" in n for n in r.ungrounded_numbers)
    assert not r.has_high_risk  # 数字仅疑似，不触发重跑


def test_chinese_numeral_not_flagged() -> None:
    c = _content(summary="完成了八成工作")
    r = check_rules(c, "会议里没有阿拉伯数字", {0}, None)
    assert r.ungrounded_numbers == []


def test_grounded_number_ok() -> None:
    c = _content(summary="增长 30%")
    r = check_rules(c, "他说增长 30% 左右", {0}, None)
    assert r.ungrounded_numbers == []


def test_example_leak_is_high_risk() -> None:
    c = _content(summary="采用 Alpha 方案推进")
    r = check_rules(c, "会议讨论了技术方案", {0}, examples=["历史纪要：Alpha 项目已上线"])
    assert "Alpha" in r.example_leaks
    assert r.has_high_risk


def test_invalid_todo_seq_is_high_risk() -> None:
    c = _content(todos=[TodoItem(task="做事", source_segment_seq=99)])
    r = check_rules(c, "转录内容", {0, 1, 2}, None)
    assert r.invalid_todo_seqs == [99]
    assert r.has_high_risk


def test_ungrounded_owner_and_deadline_high_risk() -> None:
    c = _content(todos=[TodoItem(task="做事", owner="张三", deadline="下周三")])
    r = check_rules(c, "会议里只提到李四，没说时间", {0}, None)
    joined = " ".join(r.ungrounded_owner_or_deadline)
    assert "owner:张三" in joined and "deadline:下周三" in joined
    assert r.has_high_risk


def test_grounded_owner_not_flagged() -> None:
    c = _content(todos=[TodoItem(task="做事", owner="speaker_001", deadline="周五")])
    r = check_rules(c, "[0] speaker_001: 周五之前给结论", {0}, None)
    assert r.ungrounded_owner_or_deadline == []


# ---- 模式分派 ----


def test_mode_off_returns_empty() -> None:
    c = _content(todos=[TodoItem(task="做事", source_segment_seq=99)])
    r = asyncio.run(run_quality_check(c, "转录", {0}, None, mode="off"))
    assert r.is_empty


def test_mode_rules_keeps_flags() -> None:
    c = _content(todos=[TodoItem(task="做事", source_segment_seq=99)])
    r = asyncio.run(run_quality_check(c, "转录", {0}, None, mode="rules"))
    assert r.invalid_todo_seqs == [99]


def test_hybrid_judge_filters_false_positives() -> None:
    # mock 裁判 confirmed=[] → 模糊疑似项被判为误报清掉；确定性事实仍保留
    c = _content(
        summary="增长 99%",
        todos=[TodoItem(task="做事", source_segment_seq=99)],
    )
    r = asyncio.run(run_quality_check(c, "会议无此数字", {0}, None, mode="hybrid"))
    assert r.ungrounded_numbers == []  # 裁判过滤
    assert r.invalid_todo_seqs == [99]  # 硬事实不受裁判影响


def test_hybrid_fail_open_keeps_rules_result(monkeypatch) -> None:
    import app.services.summary_quality as q

    async def _judge_down(*a, **k):
        return None  # 裁判不可用

    monkeypatch.setattr(q, "_judge", _judge_down)
    c = _content(summary="增长 99%")
    r = asyncio.run(run_quality_check(c, "会议无此数字", {0}, None, mode="hybrid"))
    assert any("99" in n for n in r.ungrounded_numbers)  # 保留规则结果
