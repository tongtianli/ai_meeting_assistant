import { useCallback, useEffect, useRef, useState } from "react";
import { askQuestion, getChatHistory } from "../api/chat";
import type { ChatMessage } from "../api/types";

function fmt(seconds: number): string {
  const t = Math.floor(seconds);
  const m = Math.floor(t / 60);
  const s = t % 60;
  return `${`${m}`.padStart(2, "0")}:${`${s}`.padStart(2, "0")}`;
}

interface Props {
  meetingId: string;
  onSeek: (seconds: number) => void;
}

export default function QaPanel({ meetingId, onSeek }: Props) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const endRef = useRef<HTMLDivElement>(null);

  const refresh = useCallback(async () => {
    try {
      setMessages(await getChatHistory(meetingId));
    } catch (err) {
      setError(err instanceof Error ? err.message : "加载失败");
    }
  }, [meetingId]);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, busy]);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    const q = input.trim();
    if (!q || busy) return;
    setBusy(true);
    setError("");
    // 乐观插入用户气泡
    const optimistic: ChatMessage = {
      id: `tmp-${Date.now()}`,
      role: "user",
      content: q,
      citations: [],
      created_at: new Date().toISOString(),
    };
    setMessages((prev) => [...prev, optimistic]);
    setInput("");
    try {
      await askQuestion(meetingId, q);
      await refresh(); // 拉回权威历史（含 assistant 回答与引用）
    } catch (err) {
      setError(err instanceof Error ? err.message : "提问失败");
      await refresh();
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="card">
      <div className="muted" style={{ marginBottom: 12 }}>
        基于本次会议转录内容问答，回答附带原文引用（点击时间戳跳转播放）。
        仅依据会议记录作答，不引入外部知识。
      </div>

      <div className="qa-messages">
        {messages.map((m) => (
          <div key={m.id} className={`qa-msg qa-${m.role}`}>
            <div className="qa-bubble">
              <div style={{ whiteSpace: "pre-wrap" }}>{m.content}</div>
              {m.citations.length > 0 && (
                <div className="qa-citations">
                  {m.citations.map((c) => (
                    <span
                      key={c.seq}
                      className="todo-source"
                      title={`${c.speaker_name}: ${c.text}`}
                      onClick={() => onSeek(c.start_time)}
                    >
                      {c.speaker_name} {fmt(c.start_time)} ▶
                    </span>
                  ))}
                </div>
              )}
            </div>
          </div>
        ))}
        {busy && <div className="qa-msg qa-assistant"><div className="qa-bubble muted">思考中…</div></div>}
        {messages.length === 0 && !busy && (
          <div className="muted">还没有提问。试试「XX 任务谁负责？」「为什么决定用方案 B？」</div>
        )}
        <div ref={endRef} />
      </div>

      {error && <div className="error">{error}</div>}

      <form onSubmit={submit} className="form-row" style={{ marginTop: 12 }}>
        <input
          type="text"
          placeholder="就本次会议内容提问…"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          disabled={busy}
        />
        <button disabled={busy || !input.trim()}>发送</button>
      </form>
    </div>
  );
}
