import { useCallback, useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { clearToken } from "../api/client";
import { listMeetings, uploadMeeting } from "../api/meetings";
import type { Meeting } from "../api/types";
import { PROCESSING_STATUSES } from "../api/types";
import StatusBadge from "../components/StatusBadge";

export default function HomePage() {
  const [meetings, setMeetings] = useState<Meeting[]>([]);
  const [title, setTitle] = useState("");
  const [location, setLocation] = useState("");
  const [host, setHost] = useState("");
  const [recorder, setRecorder] = useState("");
  const [importance, setImportance] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);

  const refresh = useCallback(async () => {
    try {
      setMeetings(await listMeetings());
    } catch (err) {
      setError(err instanceof Error ? err.message : "加载失败");
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  // 有处理中的会议时轮询进度（PRD：前端可实时查看进度）
  const hasProcessing = meetings.some((m) =>
    PROCESSING_STATUSES.includes(m.status),
  );
  useEffect(() => {
    if (!hasProcessing) return;
    const timer = setInterval(refresh, 3000);
    return () => clearInterval(timer);
  }, [hasProcessing, refresh]);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    const file = fileRef.current?.files?.[0];
    if (!file) return;
    setBusy(true);
    setError("");
    try {
      await uploadMeeting(title || file.name, file, {
        location,
        host,
        recorder,
        importance,
      });
      setTitle("");
      setLocation("");
      setHost("");
      setRecorder("");
      setImportance("");
      if (fileRef.current) fileRef.current.value = "";
      await refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : "上传失败");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="container">
      <div className="topbar">
        <h1>会议列表</h1>
        <div>
          <Link to="/examples" style={{ marginRight: 12 }}>
            纪要范例库
          </Link>
          <Link to="/persons" style={{ marginRight: 12 }}>
            人员与声纹
          </Link>
          <button
            className="secondary"
            onClick={() => {
              clearToken();
              window.location.hash = "#/login";
            }}
          >
            退出
          </button>
        </div>
      </div>

      <div className="card">
        <h3 style={{ marginTop: 0 }}>上传会议录音</h3>
        <form onSubmit={submit}>
          <div className="form-row">
            <input
              type="text"
              placeholder="会议标题（留空用文件名）"
              value={title}
              onChange={(e) => setTitle(e.target.value)}
            />
            <input ref={fileRef} type="file" accept=".mp3,.wav,.m4a,.mp4" required />
            <button disabled={busy}>{busy ? "上传中…" : "上传"}</button>
          </div>
          <div className="form-row" style={{ marginTop: 8 }}>
            <input
              type="text"
              placeholder="会议地点（选填）"
              value={location}
              onChange={(e) => setLocation(e.target.value)}
            />
            <input
              type="text"
              placeholder="主持人（选填）"
              value={host}
              onChange={(e) => setHost(e.target.value)}
            />
            <input
              type="text"
              placeholder="记录人（选填）"
              value={recorder}
              onChange={(e) => setRecorder(e.target.value)}
            />
            <select
              value={importance}
              onChange={(e) => setImportance(e.target.value)}
            >
              <option value="">重要程度（选填）</option>
              <option value="一般">一般</option>
              <option value="重要">重要</option>
              <option value="加急">加急</option>
            </select>
          </div>
        </form>
        {error && <div className="error">{error}</div>}
        <div className="muted">
          支持 mp3 / wav / m4a / mp4；抬头选填项留空时导出 Word 留白手写
        </div>
      </div>

      <div className="card">
        <table>
          <thead>
            <tr>
              <th>标题</th>
              <th>状态</th>
              <th>时长</th>
              <th>创建时间</th>
            </tr>
          </thead>
          <tbody>
            {meetings.map((m) => (
              <tr key={m.id}>
                <td>
                  <Link to={`/meetings/${m.id}`}>{m.title}</Link>
                </td>
                <td>
                  <StatusBadge status={m.status} />
                </td>
                <td>{m.duration ? `${Math.round(m.duration / 60)} 分钟` : "—"}</td>
                <td>{new Date(m.created_at).toLocaleString()}</td>
              </tr>
            ))}
            {meetings.length === 0 && (
              <tr>
                <td colSpan={4} className="muted">
                  还没有会议，上传第一段录音吧
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
