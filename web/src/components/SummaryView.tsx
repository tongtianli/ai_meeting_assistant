import type { Segment, Summary } from "../api/types";

function fmt(seconds: number): string {
  const t = Math.floor(seconds);
  const m = Math.floor(t / 60);
  const s = t % 60;
  return `${`${m}`.padStart(2, "0")}:${`${s}`.padStart(2, "0")}`;
}

interface Props {
  summary: Summary;
  segments: Segment[];
  onSeek: (seconds: number) => void;
}

export default function SummaryView({ summary, segments, onSeek }: Props) {
  const c = summary.content_json;
  const bySeq = new Map(segments.map((s) => [s.seq, s]));

  if (c._meta?.degraded) {
    return (
      <div className="card">
        <div className="muted">结构化纪要生成失败，以下为纯文本纪要（降级）</div>
        <p style={{ whiteSpace: "pre-wrap" }}>{c.text}</p>
      </div>
    );
  }

  return (
    <div className="card">
      <div className="muted">
        v{summary.version} · {c._meta?.model} · 参会人：
        {c.participants?.join("、") || "—"}
      </div>

      <h3>会议总结</h3>
      <p>{c.summary}</p>

      {!!c.discussions?.length && (
        <>
          <h3>讨论事项</h3>
          <ol>
            {c.discussions.map((d, i) => (
              <li key={i}>{d}</li>
            ))}
          </ol>
        </>
      )}

      {!!c.decisions?.length && (
        <>
          <h3>决策事项</h3>
          <ol>
            {c.decisions.map((d, i) => (
              <li key={i}>{d}</li>
            ))}
          </ol>
        </>
      )}

      {!!c.todos?.length && (
        <>
          <h3>TODO</h3>
          <table>
            <thead>
              <tr>
                <th>事项</th>
                <th>负责人</th>
                <th>截止时间</th>
                <th>来源</th>
              </tr>
            </thead>
            <tbody>
              {c.todos.map((t, i) => {
                const src =
                  t.source_segment_seq != null
                    ? bySeq.get(t.source_segment_seq)
                    : undefined;
                const seg = src ? bySeq.get(src.seq) : undefined;
                return (
                  <tr key={i}>
                    <td>{t.task}</td>
                    <td>{(seg && seg.person_id && seg.speaker_name) || t.owner || "待定"}</td>
                    <td>{t.deadline || "—"}</td>
                    <td>
                      {src ? (
                        // 溯源：点击跳到音频对应位置（PRD 设计原则 1）
                        <span
                          className="todo-source"
                          title={src.text}
                          onClick={() => onSeek(src.start_time)}
                        >
                          {fmt(src.start_time)} ▶
                        </span>
                      ) : (
                        "—"
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </>
      )}
    </div>
  );
}
