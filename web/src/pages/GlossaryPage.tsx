import { useCallback, useEffect, useState } from "react";
import { Link } from "react-router-dom";
import {
  addTerms,
  deleteTerm,
  listTerms,
  setTermEnabled,
  type GlossaryTerm,
} from "../api/glossary";

export default function GlossaryPage() {
  const [terms, setTerms] = useState<GlossaryTerm[]>([]);
  const [input, setInput] = useState("");
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [busy, setBusy] = useState(false);

  const refresh = useCallback(async () => {
    try {
      setTerms(await listTerms());
    } catch (err) {
      setError(err instanceof Error ? err.message : "加载失败");
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    // 按换行或逗号拆成多条
    const parsed = input
      .split(/[\n,，]/)
      .map((t) => t.trim())
      .filter(Boolean);
    if (parsed.length === 0) return;
    setBusy(true);
    setError("");
    setNotice("");
    try {
      const created = await addTerms(parsed);
      setInput("");
      const skipped = parsed.length - created.length;
      setNotice(
        `新增 ${created.length} 个术语` +
          (skipped > 0 ? `，${skipped} 个已存在或重复已跳过` : ""),
      );
      await refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : "添加失败");
    } finally {
      setBusy(false);
    }
  }

  async function toggle(t: GlossaryTerm) {
    setError("");
    try {
      await setTermEnabled(t.id, !t.enabled);
      await refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : "更新失败");
    }
  }

  async function remove(t: GlossaryTerm) {
    if (!window.confirm(`删除术语「${t.term}」？`)) return;
    setError("");
    try {
      await deleteTerm(t.id);
      await refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : "删除失败");
    }
  }

  const enabledCount = terms.filter((t) => t.enabled).length;

  return (
    <div className="container">
      <div className="topbar">
        <h1>术语表</h1>
        <Link to="/">← 返回会议列表</Link>
      </div>

      <div className="card">
        <h3 style={{ marginTop: 0 }}>添加术语</h3>
        <div className="muted" style={{ marginBottom: 8 }}>
          人名、公司专名、行业术语等——启用中的术语会在每次转写时作为热词
          注入语音识别，提升这些词的识别准确率。跨会议全局生效，切换 ASR
          引擎（云端 / 本地）同样有效。可一次粘贴多个，按换行或逗号分隔。
        </div>
        <form onSubmit={submit}>
          <textarea
            placeholder={"声纹\npgvector\n每行一个，或用逗号分隔"}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            rows={5}
            style={{ width: "100%", boxSizing: "border-box" }}
          />
          <div style={{ marginTop: 8 }}>
            <button disabled={busy}>{busy ? "提交中…" : "添加"}</button>
          </div>
        </form>
        {error && <div className="error">{error}</div>}
        {notice && <div className="muted">{notice}</div>}
      </div>

      <div className="card">
        <div className="muted" style={{ marginBottom: 8 }}>
          共 {terms.length} 个术语，{enabledCount} 个启用中
        </div>
        <table>
          <thead>
            <tr>
              <th>术语</th>
              <th>启用</th>
              <th>添加时间</th>
              <th>操作</th>
            </tr>
          </thead>
          <tbody>
            {terms.map((t) => (
              <tr key={t.id}>
                <td>{t.term}</td>
                <td>
                  <input
                    type="checkbox"
                    checked={t.enabled}
                    onChange={() => void toggle(t)}
                  />
                </td>
                <td>{new Date(t.created_at).toLocaleString()}</td>
                <td>
                  <button className="danger" onClick={() => void remove(t)}>
                    删除
                  </button>
                </td>
              </tr>
            ))}
            {terms.length === 0 && (
              <tr>
                <td colSpan={4} className="muted">
                  还没有术语；添加人名/专名/术语可提升识别准确率
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
