import { Fragment, useCallback, useEffect, useState } from "react";
import { Link } from "react-router-dom";
import {
  createExample,
  deleteExample,
  listExamples,
  updateExample,
  type SummaryExample,
} from "../api/examples";

export default function ExamplesPage() {
  const [examples, setExamples] = useState<SummaryExample[]>([]);
  const [title, setTitle] = useState("");
  const [content, setContent] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editText, setEditText] = useState("");

  const refresh = useCallback(async () => {
    try {
      setExamples(await listExamples());
    } catch (err) {
      setError(err instanceof Error ? err.message : "加载失败");
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError("");
    try {
      await createExample(title, content);
      setTitle("");
      setContent("");
      await refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : "创建失败");
    } finally {
      setBusy(false);
    }
  }

  async function toggle(ex: SummaryExample) {
    setError("");
    try {
      await updateExample(ex.id, { enabled: !ex.enabled });
      await refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : "更新失败");
    }
  }

  function startEdit(ex: SummaryExample) {
    setEditingId(ex.id);
    setEditText(ex.content);
    setExpandedId(ex.id);
  }

  async function saveEdit(ex: SummaryExample) {
    setError("");
    try {
      await updateExample(ex.id, { content: editText });
      setEditingId(null);
      await refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : "保存失败");
    }
  }

  async function remove(ex: SummaryExample) {
    if (!window.confirm(`删除范例「${ex.title}」？`)) return;
    setError("");
    try {
      await deleteExample(ex.id);
      await refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : "删除失败");
    }
  }

  return (
    <div className="container">
      <div className="topbar">
        <h1>纪要范例库</h1>
        <Link to="/">← 返回会议列表</Link>
      </div>

      <div className="card">
        <h3 style={{ marginTop: 0 }}>新增范例</h3>
        <div className="muted" style={{ marginBottom: 8 }}>
          粘贴一份公司格式的优质会议纪要作为文风范例；也可以在会议详情页把
          生成效果好的纪要一键「存为范例」。启用中的范例会在每次生成纪要时
          作为 few-shot 参照注入，模型将模仿其议题命名、句式与用语习惯。
        </div>
        <form onSubmit={submit}>
          <div className="form-row">
            <input
              type="text"
              placeholder="范例标题（如：7月安全生产例会）"
              value={title}
              onChange={(e) => setTitle(e.target.value)}
              required
            />
            <button disabled={busy}>{busy ? "提交中…" : "添加"}</button>
          </div>
          <textarea
            placeholder="粘贴纪要正文（公司格式）"
            value={content}
            onChange={(e) => setContent(e.target.value)}
            required
            rows={8}
            style={{ width: "100%", marginTop: 8, boxSizing: "border-box" }}
          />
        </form>
        {error && <div className="error">{error}</div>}
      </div>

      <div className="card">
        <table>
          <thead>
            <tr>
              <th>标题</th>
              <th>来源</th>
              <th>启用</th>
              <th>更新时间</th>
              <th>操作</th>
            </tr>
          </thead>
          <tbody>
            {examples.map((ex) => (
              <Fragment key={ex.id}>
                <tr>
                  <td>
                    <a
                      href="#"
                      onClick={(e) => {
                        e.preventDefault();
                        setExpandedId(expandedId === ex.id ? null : ex.id);
                      }}
                    >
                      {ex.title}
                    </a>
                  </td>
                  <td>
                    {ex.source_meeting_id ? (
                      <Link to={`/meetings/${ex.source_meeting_id}`}>
                        采纳自会议
                      </Link>
                    ) : (
                      "手工添加"
                    )}
                  </td>
                  <td>
                    <input
                      type="checkbox"
                      checked={ex.enabled}
                      onChange={() => void toggle(ex)}
                    />
                  </td>
                  <td>{new Date(ex.updated_at).toLocaleString()}</td>
                  <td>
                    <button
                      className="secondary"
                      onClick={() => startEdit(ex)}
                      style={{ marginRight: 8 }}
                    >
                      编辑
                    </button>
                    <button className="secondary" onClick={() => void remove(ex)}>
                      删除
                    </button>
                  </td>
                </tr>
                {expandedId === ex.id && (
                  <tr>
                    <td colSpan={5}>
                      {editingId === ex.id ? (
                        <>
                          <textarea
                            value={editText}
                            onChange={(e) => setEditText(e.target.value)}
                            rows={12}
                            style={{ width: "100%", boxSizing: "border-box" }}
                          />
                          <div style={{ marginTop: 8 }}>
                            <button
                              onClick={() => void saveEdit(ex)}
                              style={{ marginRight: 8 }}
                            >
                              保存
                            </button>
                            <button
                              className="secondary"
                              onClick={() => setEditingId(null)}
                            >
                              取消
                            </button>
                          </div>
                        </>
                      ) : (
                        <pre
                          style={{
                            whiteSpace: "pre-wrap",
                            margin: 0,
                            fontFamily: "inherit",
                          }}
                        >
                          {ex.content}
                        </pre>
                      )}
                    </td>
                  </tr>
                )}
              </Fragment>
            ))}
            {examples.length === 0 && (
              <tr>
                <td colSpan={5} className="muted">
                  还没有范例；生成效果好的纪要可在详情页「存为范例」
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
