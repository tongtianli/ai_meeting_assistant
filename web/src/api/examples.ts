import { getToken, request } from "./client";

export interface SummaryExample {
  id: string;
  title: string;
  content: string;
  source_meeting_id: string | null;
  enabled: boolean;
  created_at: string;
  updated_at: string;
}

export function listExamples(): Promise<SummaryExample[]> {
  return request<SummaryExample[]>("/api/summary-examples");
}

export function createExample(
  title: string,
  content: string,
): Promise<SummaryExample> {
  return request<SummaryExample>("/api/summary-examples", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ title, content }),
  });
}

export function createExampleFromMeeting(
  meetingId: string,
): Promise<SummaryExample> {
  return request<SummaryExample>(
    `/api/summary-examples/from-meeting/${meetingId}`,
    { method: "POST" },
  );
}

export function updateExample(
  id: string,
  patch: Partial<Pick<SummaryExample, "title" | "content" | "enabled">>,
): Promise<SummaryExample> {
  return request<SummaryExample>(`/api/summary-examples/${id}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(patch),
  });
}

export async function deleteExample(id: string): Promise<void> {
  // 204 无响应体，不能走统一的 JSON request
  const headers = new Headers();
  const token = getToken();
  if (token) headers.set("Authorization", `Bearer ${token}`);
  const resp = await fetch(`/api/summary-examples/${id}`, {
    method: "DELETE",
    headers,
  });
  if (!resp.ok) throw new Error(`删除失败（${resp.status}）`);
}
