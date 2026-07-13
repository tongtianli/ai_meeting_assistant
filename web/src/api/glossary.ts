import { request } from "./client";

export interface GlossaryTerm {
  id: string;
  term: string;
  enabled: boolean;
  created_at: string;
  updated_at: string;
}

export function listTerms(): Promise<GlossaryTerm[]> {
  return request<GlossaryTerm[]>("/api/glossary");
}

export function addTerms(terms: string[]): Promise<GlossaryTerm[]> {
  return request<GlossaryTerm[]>("/api/glossary", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ terms }),
  });
}

export function setTermEnabled(
  id: string,
  enabled: boolean,
): Promise<GlossaryTerm> {
  return request<GlossaryTerm>(`/api/glossary/${id}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ enabled }),
  });
}

export function deleteTerm(id: string): Promise<void> {
  return request<void>(`/api/glossary/${id}`, { method: "DELETE" });
}
