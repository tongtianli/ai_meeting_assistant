import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import {
  getUsageStats,
  type GrantStatus,
  type UsageStats,
} from "../api/llmUsage";

function fmt(n: number | null | undefined): string {
  if (n === null || n === undefined) return "—";
  return n.toLocaleString("zh-CN");
}

function pct(ratio: number): string {
  return `${(ratio * 100).toFixed(1)}%`;
}

const GRANT_LABELS: Record<string, string> = {
  glm_air: "GLM-4.5-Air 资源包",
  glm_general: "GLM 通用资源包（Flash + Embedding）",
};

const LEVEL_LABELS: Record<string, string> = {
  observe: "已用超 70%：建议观察",
  evaluate_routing: "已用超 85%：建议评估路由配置",
  high_value_only: "已用超 95%：建议仅高价值任务使用",
  exhausted: "应用侧估算已耗尽（以控制台余额为准）",
};

function GrantCard({ g }: { g: GrantStatus }) {
  const ratio = Math.min(g.usage_ratio, 1);
  const barColor =
    g.usage_ratio >= 0.95 ? "#c0392b" : g.usage_ratio >= 0.7 ? "#e67e22" : "#27ae60";
  return (
    <div className="card">
      <h3 style={{ marginTop: 0 }}>{GRANT_LABELS[g.name] ?? g.name}</h3>
      {g.expiry_warning && (
        <div className="error" style={{ marginBottom: 8 }}>
          ⚠ {g.expiry_warning}
        </div>
      )}
      {g.usage_level && (
        <div className="error" style={{ marginBottom: 8 }}>
          {LEVEL_LABELS[g.usage_level] ?? g.usage_level}
        </div>
      )}
      <div
        style={{
          background: "#eee",
          borderRadius: 4,
          height: 10,
          overflow: "hidden",
          marginBottom: 8,
        }}
      >
        <div
          style={{
            width: `${ratio * 100}%`,
            background: barColor,
            height: "100%",
          }}
        />
      </div>
      <div className="muted" style={{ marginBottom: 8 }}>
        应用侧累计 {fmt(g.tracked_total_tokens)} / 配置总量{" "}
        {fmt(g.grant_total_tokens)} tokens（{pct(g.usage_ratio)}）——额度为
        账号级共享（跨用户合计），且仅为应用侧估算，控制台余额是最终真值
      </div>
      <table>
        <tbody>
          <tr>
            <td>预计剩余额度</td>
            <td>{fmt(g.estimated_remaining_tokens)} tokens</td>
          </tr>
          <tr>
            <td>每场会议平均消耗</td>
            <td>{fmt(g.avg_tokens_per_meeting)} tokens</td>
          </tr>
          <tr>
            <td>预计还能生成会议</td>
            <td>
              {g.estimated_remaining_meetings === null
                ? "—"
                : `约 ${fmt(g.estimated_remaining_meetings)} 场`}
            </td>
          </tr>
          <tr>
            <td>最近 7 天日均消耗（按活跃天数）</td>
            <td>{fmt(g.avg_daily_tokens_7d)} tokens</td>
          </tr>
          <tr>
            <td>按当前速度预计耗尽</td>
            <td>{g.estimated_exhaustion_date ?? "—"}</td>
          </tr>
          <tr>
            <td>资源包到期</td>
            <td>
              {new Date(g.expires_at).toLocaleDateString()}（剩{" "}
              {g.days_until_expiry} 天）
            </td>
          </tr>
        </tbody>
      </table>
    </div>
  );
}

export default function LlmUsagePage() {
  const [stats, setStats] = useState<UsageStats | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    getUsageStats()
      .then(setStats)
      .catch((err) =>
        setError(err instanceof Error ? err.message : "加载失败"),
      );
  }, []);

  return (
    <div className="container">
      <div className="topbar">
        <h1>模型用量</h1>
        <Link to="/">← 返回会议列表</Link>
      </div>
      {error && <div className="error">{error}</div>}
      {!stats && !error && <div className="muted">加载中…</div>}
      {stats && (
        <>
          <div className="muted" style={{ marginBottom: 12 }}>
            当前用户累计 {fmt(stats.total_calls)} 次调用，
            {fmt(stats.total_tokens)} tokens（成功与失败调用均计入审计；
            以下统计仅含当前用户，资源包卡片为账号级）
          </div>

          {stats.grants.map((g) => (
            <GrantCard key={g.name} g={g} />
          ))}

          <div className="card">
            <h3 style={{ marginTop: 0 }}>按任务统计</h3>
            <table>
              <thead>
                <tr>
                  <th>任务</th>
                  <th>调用数</th>
                  <th>成功率</th>
                  <th>输入 tokens</th>
                  <th>输出 tokens</th>
                  <th>合计 tokens</th>
                  <th>平均耗时</th>
                </tr>
              </thead>
              <tbody>
                {stats.by_task.map((t) => (
                  <tr key={t.task_type}>
                    <td>{t.task_type}</td>
                    <td>{fmt(t.calls)}</td>
                    <td>{pct(t.success_rate)}</td>
                    <td>{fmt(t.input_tokens)}</td>
                    <td>{fmt(t.output_tokens)}</td>
                    <td>{fmt(t.total_tokens)}</td>
                    <td>
                      {t.avg_latency_ms === null
                        ? "—"
                        : `${(t.avg_latency_ms / 1000).toFixed(1)}s`}
                    </td>
                  </tr>
                ))}
                {stats.by_task.length === 0 && (
                  <tr>
                    <td colSpan={7} className="muted">
                      暂无调用记录
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>

          <div className="card">
            <h3 style={{ marginTop: 0 }}>按 Provider 统计</h3>
            <table>
              <thead>
                <tr>
                  <th>Provider</th>
                  <th>调用数</th>
                  <th>成功率</th>
                  <th>作为降级被调用</th>
                  <th>合计 tokens</th>
                </tr>
              </thead>
              <tbody>
                {stats.by_provider.map((p) => (
                  <tr key={p.provider}>
                    <td>{p.provider}</td>
                    <td>{fmt(p.calls)}</td>
                    <td>{pct(p.success_rate)}</td>
                    <td>
                      {fmt(p.fallback_calls)}（{pct(p.fallback_rate)}）
                    </td>
                    <td>{fmt(p.total_tokens)}</td>
                  </tr>
                ))}
                {stats.by_provider.length === 0 && (
                  <tr>
                    <td colSpan={5} className="muted">
                      暂无调用记录
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>

          <div className="card">
            <h3 style={{ marginTop: 0 }}>最近失败（fallback 原因）</h3>
            <table>
              <thead>
                <tr>
                  <th>时间</th>
                  <th>任务</th>
                  <th>Provider</th>
                  <th>错误类型</th>
                  <th>错误信息</th>
                </tr>
              </thead>
              <tbody>
                {stats.recent_failures.map((f, i) => (
                  <tr key={i}>
                    <td>{new Date(f.created_at).toLocaleString()}</td>
                    <td>{f.task_type}</td>
                    <td>{f.provider}</td>
                    <td>{f.error_type ?? "—"}</td>
                    <td className="muted">{f.error_message ?? "—"}</td>
                  </tr>
                ))}
                {stats.recent_failures.length === 0 && (
                  <tr>
                    <td colSpan={5} className="muted">
                      暂无失败记录
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </>
      )}
    </div>
  );
}
