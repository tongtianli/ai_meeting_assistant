export type MeetingStatus =
  | "uploaded"
  | "transcoding"
  | "transcribing"
  | "summarizing"
  | "done"
  | "failed";

export interface Meeting {
  id: string;
  title: string;
  status: MeetingStatus;
  duration: number | null;
  error_message: string | null;
  location: string | null;
  host: string | null;
  recorder: string | null;
  importance: string | null;
  created_at: string;
}

export interface Segment {
  seq: number;
  start_time: number;
  end_time: number;
  speaker_label: string;
  speaker_name: string;
  person_id: string | null;
  text: string;
}

export interface Transcript {
  meeting_id: string;
  segments: Segment[];
}

export interface TodoItem {
  task: string;
  owner: string | null;
  deadline: string | null;
  source_segment_seq: number | null;
}

export interface TopicGroup {
  title: string;
  owner: string | null;
  items: string[];
}

export interface SummaryContent {
  title?: string;
  participants?: string[];
  summary?: string;
  topics?: TopicGroup[]; // 新版：议题分组（公司纪要格式）
  discussions?: string[]; // 旧版纪要兼容
  decisions?: string[];
  todos?: TodoItem[];
  text?: string; // 降级纯文本
  _meta?: { model: string; degraded: boolean };
}

export interface Summary {
  id: string;
  meeting_id: string;
  version: number;
  content_json: SummaryContent;
  created_at: string;
}

export interface Citation {
  seq: number;
  start_time: number;
  speaker_name: string;
  text: string;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  citations: Citation[];
  created_at: string;
}

export const PROCESSING_STATUSES: MeetingStatus[] = [
  "uploaded",
  "transcoding",
  "transcribing",
  "summarizing",
];
