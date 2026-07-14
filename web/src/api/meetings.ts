import { request, requestBlob, setToken } from "./client";

// anchor 必须挂到 DOM，否则 Chromium 忽略 download 属性（文件名变成 "download"）
function triggerDownload(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}
import type { Meeting, Summary, Transcript } from "./types";

export async function login(password: string): Promise<void> {
  const data = await request<{ access_token: string }>("/api/auth/token", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ password }),
  });
  setToken(data.access_token);
}

export function listMeetings(): Promise<Meeting[]> {
  return request<Meeting[]>("/api/meetings");
}

export function getMeeting(id: string): Promise<Meeting> {
  return request<Meeting>(`/api/meetings/${id}`);
}

export interface MeetingHeaderFields {
  location?: string;
  host?: string;
  recorder?: string;
  importance?: string;
}

export function uploadMeeting(
  title: string,
  file: File,
  header: MeetingHeaderFields = {},
): Promise<Meeting> {
  const form = new FormData();
  form.append("title", title);
  form.append("file", file);
  for (const [key, value] of Object.entries(header)) {
    if (value) form.append(key, value);
  }
  return request<Meeting>("/api/meetings", { method: "POST", body: form });
}

export function retryMeeting(id: string): Promise<Meeting> {
  return request<Meeting>(`/api/meetings/${id}/retry`, { method: "POST" });
}

export function resummarizeMeeting(id: string): Promise<Meeting> {
  return request<Meeting>(`/api/meetings/${id}/resummarize`, {
    method: "POST",
  });
}

export function deleteMeeting(id: string): Promise<void> {
  return request<void>(`/api/meetings/${id}`, { method: "DELETE" });
}

export function getTranscript(id: string): Promise<Transcript> {
  return request<Transcript>(`/api/meetings/${id}/segments`);
}

export function getSummary(id: string): Promise<Summary> {
  return request<Summary>(`/api/meetings/${id}/summary`);
}

export function renameSpeaker(
  id: string,
  speakerLabel: string,
  name: string,
): Promise<unknown> {
  return request(`/api/meetings/${id}/speaker-bindings`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ speaker_label: speakerLabel, name }),
  });
}

export function getAudioUrl(id: string): Promise<{ url: string }> {
  return request<{ url: string }>(`/api/meetings/${id}/audio-url`);
}

export async function downloadWord(id: string, title: string): Promise<void> {
  const blob = await requestBlob(`/api/meetings/${id}/export.docx`);
  triggerDownload(blob, `${title}-纪要.docx`);
}

export async function downloadTranscript(
  id: string,
  title: string,
): Promise<void> {
  const blob = await requestBlob(`/api/meetings/${id}/transcript`);
  triggerDownload(blob, `${title}-转录.txt`);
}
