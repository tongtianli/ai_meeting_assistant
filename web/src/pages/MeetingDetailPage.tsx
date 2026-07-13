import { useCallback, useEffect, useRef, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { createExampleFromMeeting } from "../api/examples";
import {
  downloadTranscript,
  downloadWord,
  getAudioUrl,
  getMeeting,
  getSummary,
  getTranscript,
  renameSpeaker,
  retryMeeting,
} from "../api/meetings";
import type { Meeting, Segment, Summary } from "../api/types";
import { PROCESSING_STATUSES } from "../api/types";
import StatusBadge from "../components/StatusBadge";
import SummaryView from "../components/SummaryView";
import TranscriptView from "../components/TranscriptView";

export default function MeetingDetailPage() {
  const { id = "" } = useParams();
  const [meeting, setMeeting] = useState<Meeting | null>(null);
  const [segments, setSegments] = useState<Segment[]>([]);
  const [summary, setSummary] = useState<Summary | null>(null);
  const [tab, setTab] = useState<"summary" | "transcript">("summary");
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [currentTime, setCurrentTime] = useState(0);
  const audioRef = useRef<HTMLAudioElement>(null);
  const [audioSrc, setAudioSrc] = useState("");

  const refresh = useCallback(async () => {
    try {
      const m = await getMeeting(id);
      setMeeting(m);
      if (m.status === "done" || m.status === "summarizing") {
        setSegments((await getTranscript(id)).segments);
      }
      if (m.status === "done") {
        setSummary(await getSummary(id));
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "加载失败");
    }
  }, [id]);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const processing =
    meeting !== null && PROCESSING_STATUSES.includes(meeting.status);
  useEffect(() => {
    if (!processing) return;
    const timer = setInterval(refresh, 3000);
    return () => clearInterval(timer);
  }, [processing, refresh]);

  // 音频短时签名 URL：按需换发（PRD §9.2）
  useEffect(() => {
    if (meeting?.status !== "done" || audioSrc) return;
    void getAudioUrl(id).then((r) => setAudioSrc(r.url));
  }, [meeting?.status, audioSrc, id]);

  function seek(seconds: number) {
    const audio = audioRef.current;
    if (!audio) return;
    audio.currentTime = seconds;
    void audio.play();
  }

  async function saveAsExample() {
    setError("");
    setNotice("");
    try {
      const ex = await createExampleFromMeeting(id);
      setNotice(`已存为范例「${ex.title}」，之后生成纪要将模仿其文风（范例库可润色）`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "存为范例失败");
    }
  }

  async function handleRename(label: string, currentName: string) {
    const name = window.prompt(`将 ${label} 重命名为：`, currentName)?.trim();
    if (!name) return;
    try {
      await renameSpeaker(id, label, name);
      await refresh(); // 转录与纪要归属立即用真名
    } catch (err) {
      setError(err instanceof Error ? err.message : "重命名失败");
    }
  }

  if (!meeting) {
    return (
      <div className="container">
        {error ? <div className="error">{error}</div> : "加载中…"}
      </div>
    );
  }

  return (
    <div className="container">
      <div className="topbar">
        <h1>
          <Link to="/">←</Link> {meeting.title}
        </h1>
        <StatusBadge status={meeting.status} />
      </div>
      {error && <div className="error">{error}</div>}
      {notice && <div className="card muted">{notice}</div>}

      {meeting.status === "failed" && (
        <div className="card">
          <div className="error">处理失败：{meeting.error_message}</div>
          <button
            onClick={async () => {
              await retryMeeting(id);
              await refresh();
            }}
          >
            重试
          </button>
        </div>
      )}

      {processing && (
        <div className="card muted">处理中，页面会自动刷新进度…</div>
      )}

      {audioSrc && (
        <div className="audio-bar">
          <audio
            ref={audioRef}
            src={audioSrc}
            controls
            onTimeUpdate={(e) => setCurrentTime(e.currentTarget.currentTime)}
          />
        </div>
      )}

      {meeting.status === "done" && (
        <>
          <div className="tabs">
            <button
              className={tab === "summary" ? "active" : ""}
              onClick={() => setTab("summary")}
            >
              会议纪要
            </button>
            <button
              className={tab === "transcript" ? "active" : ""}
              onClick={() => setTab("transcript")}
            >
              原始转录
            </button>
            <span style={{ flex: 1 }} />
            <button className="secondary" onClick={() => void saveAsExample()}>
              存为范例
            </button>
            <button
              className="secondary"
              onClick={() => downloadTranscript(id, meeting.title)}
            >
              下载转录
            </button>
            <button onClick={() => downloadWord(id, meeting.title)}>
              导出 Word
            </button>
          </div>

          {tab === "summary" && summary && (
            <SummaryView
              summary={summary}
              segments={segments}
              onSeek={seek}
            />
          )}
          {tab === "transcript" && (
            <TranscriptView
              segments={segments}
              currentTime={currentTime}
              onSeek={seek}
              onRename={handleRename}
            />
          )}
        </>
      )}
    </div>
  );
}
