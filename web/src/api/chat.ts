import { request } from "./client";
import type { ChatMessage } from "./types";

export function getChatHistory(meetingId: string): Promise<ChatMessage[]> {
  return request<ChatMessage[]>(`/api/meetings/${meetingId}/chat`);
}

export function askQuestion(
  meetingId: string,
  question: string,
): Promise<ChatMessage> {
  return request<ChatMessage>(`/api/meetings/${meetingId}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
  });
}
