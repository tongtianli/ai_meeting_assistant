import type { Segment } from "../api/types";

function fmt(seconds: number): string {
  const t = Math.floor(seconds);
  const h = Math.floor(t / 3600);
  const m = Math.floor((t % 3600) / 60);
  const s = t % 60;
  const mm = `${m}`.padStart(2, "0");
  const ss = `${s}`.padStart(2, "0");
  return h > 0 ? `${h}:${mm}:${ss}` : `${mm}:${ss}`;
}

interface Props {
  segments: Segment[];
  currentTime: number;
  onSeek: (seconds: number) => void;
  onRename: (label: string, currentName: string) => void;
}

export default function TranscriptView({
  segments,
  currentTime,
  onSeek,
  onRename,
}: Props) {
  return (
    <div className="card">
      {segments.map((s) => (
        <div
          key={s.seq}
          className={`segment ${
            currentTime >= s.start_time && currentTime < s.end_time
              ? "current"
              : ""
          }`}
          onClick={() => onSeek(s.start_time)}
        >
          <span className="time">{fmt(s.start_time)}</span>
          <span
            className="speaker"
            title="点击重命名"
            onClick={(e) => {
              e.stopPropagation(); // 不触发跳转播放
              onRename(s.speaker_label, s.speaker_name);
            }}
          >
            {s.speaker_name}
          </span>
          <span>{s.text}</span>
        </div>
      ))}
      {segments.length === 0 && <div className="muted">暂无转录</div>}
    </div>
  );
}
