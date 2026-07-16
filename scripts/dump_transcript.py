"""导出带 seq 的会议转录（评估数据集手工整理用，RAG 设计 §6.2）。

用法：
    uv run python scripts/dump_transcript.py --list          # 列出近期会议
    uv run python scripts/dump_transcript.py <meeting_id>    # 导出该会议转录

输出格式 `[seq] [时间] 说话人: 文本` ——直接粘贴给大模型出题，并要求它
引用行首 seq 作为 expected_segment_seqs（见 eval/qa_dataset.example.json）。
"""
import argparse
import asyncio
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


async def _list_meetings() -> None:
    from sqlalchemy import func, select

    from app.db.session import SessionLocal
    from app.models import Meeting, TranscriptSegment

    async with SessionLocal() as session:
        rows = await session.execute(
            select(
                Meeting.id,
                Meeting.title,
                Meeting.created_at,
                func.count(TranscriptSegment.id),
            )
            .outerjoin(TranscriptSegment, TranscriptSegment.meeting_id == Meeting.id)
            .group_by(Meeting.id)
            .order_by(Meeting.created_at.desc())
            .limit(30)
        )
        for mid, title, created, seg_count in rows:
            print(f"{mid}  {created:%Y-%m-%d %H:%M}  segments={seg_count:<4d}  {title}")


async def _dump(meeting_id: uuid.UUID) -> None:
    from sqlalchemy import select

    from app.db.session import SessionLocal
    from app.models import Meeting, TranscriptSegment
    from app.services.speakers import active_speaker_names
    from app.services.summarize import _fmt_ts

    async with SessionLocal() as session:
        meeting = await session.get(Meeting, meeting_id)
        if meeting is None:
            print(f"meeting {meeting_id} 不存在", file=sys.stderr)
            raise SystemExit(1)
        names = await active_speaker_names(session, meeting_id)
        segments = await session.scalars(
            select(TranscriptSegment)
            .where(TranscriptSegment.meeting_id == meeting_id)
            .order_by(TranscriptSegment.seq)
        )
        print(f"# meeting_id: {meeting_id}")
        print(f"# title: {meeting.title}")
        for s in segments:
            speaker = names.get(s.speaker_label, (None, s.speaker_label))[1]
            print(f"[{s.seq}] [{_fmt_ts(s.start_time)}] {speaker}: {s.text}")


def main() -> None:
    parser = argparse.ArgumentParser(description="导出带 seq 的会议转录")
    parser.add_argument("meeting_id", nargs="?", help="会议 UUID")
    parser.add_argument("--list", action="store_true", help="列出近期会议")
    args = parser.parse_args()
    try:
        if args.list or not args.meeting_id:
            asyncio.run(_list_meetings())
        else:
            asyncio.run(_dump(uuid.UUID(args.meeting_id)))
    except BrokenPipeError:  # 管道被 head 等截断属正常使用
        sys.stderr.close()


if __name__ == "__main__":
    main()
