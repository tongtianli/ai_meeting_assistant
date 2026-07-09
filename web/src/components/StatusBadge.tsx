import type { MeetingStatus } from "../api/types";
import { PROCESSING_STATUSES } from "../api/types";

const LABELS: Record<MeetingStatus, string> = {
  uploaded: "已上传",
  transcoding: "转码中",
  transcribing: "转写中",
  summarizing: "生成纪要中",
  done: "完成",
  failed: "失败",
};

export default function StatusBadge({ status }: { status: MeetingStatus }) {
  const cls = PROCESSING_STATUSES.includes(status) ? "processing" : status;
  return <span className={`badge ${cls}`}>{LABELS[status]}</span>;
}
