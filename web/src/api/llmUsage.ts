import { request } from "./client";

export interface ProviderStat {
  provider: string;
  calls: number;
  success_calls: number;
  success_rate: number;
  fallback_calls: number;
  fallback_rate: number;
  input_tokens: number;
  output_tokens: number;
  total_tokens: number;
}

export interface TaskStat {
  task_type: string;
  calls: number;
  success_calls: number;
  success_rate: number;
  input_tokens: number;
  output_tokens: number;
  total_tokens: number;
  avg_latency_ms: number | null;
}

export interface GrantStatus {
  name: string;
  providers: string[];
  grant_total_tokens: number;
  tracked_total_tokens: number;
  estimated_remaining_tokens: number;
  usage_ratio: number;
  usage_level: string | null;
  avg_tokens_per_meeting: number | null;
  estimated_remaining_meetings: number | null;
  avg_daily_tokens_7d: number | null;
  estimated_exhaustion_date: string | null;
  expires_at: string;
  days_until_expiry: number;
  expiry_warning: string | null;
  soft_limit_tokens: number | null;
  soft_limit_reached: boolean;
  hard_limit_reached: boolean;
  expired: boolean;
  allow_paid_after_grant: boolean;
  enforcement: string; // none | high_value_only | blocked
  enforcement_reason: string | null; // exhausted | expired
}

export interface UsageFailure {
  created_at: string;
  meeting_id: string | null;
  task_type: string;
  provider: string;
  model: string;
  fallback_index: number;
  error_type: string | null;
  error_message: string | null;
}

export interface UsageStats {
  total_calls: number;
  total_tokens: number;
  by_provider: ProviderStat[];
  by_task: TaskStat[];
  grants: GrantStatus[];
  recent_failures: UsageFailure[];
}

export function getUsageStats(): Promise<UsageStats> {
  return request<UsageStats>("/api/llm-usage/stats");
}
