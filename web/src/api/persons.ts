import { request } from "./client";

export interface Person {
  id: string;
  name: string;
  voiceprint_id: string | null;
  consent_record: { note: string; recorded_at: string } | null;
  created_at: string;
}

export interface PersonDeleteResult {
  deleted: boolean;
  cloud_cleanup_required: boolean;
  message: string;
}

export function listPersons(): Promise<Person[]> {
  return request<Person[]>("/api/persons");
}

export function createPerson(
  name: string,
  voiceprintId?: string,
  consentNote?: string,
): Promise<Person> {
  return request<Person>("/api/persons", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      name,
      voiceprint_id: voiceprintId || null,
      consent_note: consentNote || null,
    }),
  });
}

export function deletePerson(id: string): Promise<PersonDeleteResult> {
  return request<PersonDeleteResult>(`/api/persons/${id}`, {
    method: "DELETE",
  });
}
