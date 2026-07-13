import { useCallback, useEffect, useState } from "react";
import { Link } from "react-router-dom";
import {
  createPerson,
  deletePerson,
  listPersons,
  type Person,
} from "../api/persons";

export default function PersonsPage() {
  const [persons, setPersons] = useState<Person[]>([]);
  const [name, setName] = useState("");
  const [voiceprintId, setVoiceprintId] = useState("");
  const [consentNote, setConsentNote] = useState("");
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [busy, setBusy] = useState(false);

  const refresh = useCallback(async () => {
    try {
      setPersons(await listPersons());
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
    setNotice("");
    try {
      await createPerson(name, voiceprintId || undefined, consentNote || undefined);
      setName("");
      setVoiceprintId("");
      setConsentNote("");
      await refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : "创建失败");
    } finally {
      setBusy(false);
    }
  }

  async function remove(p: Person) {
    const warning = p.voiceprint_id
      ? `删除「${p.name}」？其云端声纹样本（${p.voiceprint_id}）需在火山控制台手动删除。`
      : `删除「${p.name}」？`;
    if (!window.confirm(warning)) return;
    setError("");
    try {
      const result = await deletePerson(p.id);
      setNotice(result.cloud_cleanup_required ? result.message : "");
      await refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : "删除失败");
    }
  }

  return (
    <div className="container">
      <div className="topbar">
        <h1>人员与声纹</h1>
        <Link to="/">← 返回会议列表</Link>
      </div>

      <div className="card">
        <h3 style={{ marginTop: 0 }}>登记人员</h3>
        <div className="muted" style={{ marginBottom: 8 }}>
          声纹 ID 在火山控制台「声纹管理」上传样本（16kHz 单声道
          wav，纯净人声 ≥10 秒）后获得；登记后转写将自动识别并绑定该说话人。
          声纹属敏感个人信息，登记时必须填写本人同意记录。
        </div>
        <form onSubmit={submit}>
          <div className="form-row">
            <input
              type="text"
              placeholder="姓名"
              value={name}
              onChange={(e) => setName(e.target.value)}
              required
            />
            <input
              type="text"
              placeholder="声纹 ID（选填）"
              value={voiceprintId}
              onChange={(e) => setVoiceprintId(e.target.value)}
            />
            <input
              type="text"
              placeholder="同意记录（登记声纹时必填）"
              value={consentNote}
              onChange={(e) => setConsentNote(e.target.value)}
              required={Boolean(voiceprintId)}
            />
            <button disabled={busy}>{busy ? "提交中…" : "登记"}</button>
          </div>
        </form>
        {error && <div className="error">{error}</div>}
        {notice && <div className="muted">{notice}</div>}
      </div>

      <div className="card">
        <table>
          <thead>
            <tr>
              <th>姓名</th>
              <th>声纹 ID</th>
              <th>同意记录</th>
              <th>操作</th>
            </tr>
          </thead>
          <tbody>
            {persons.map((p) => (
              <tr key={p.id}>
                <td>{p.name}</td>
                <td>{p.voiceprint_id ?? "—"}</td>
                <td>
                  {p.consent_record
                    ? `${p.consent_record.note}（${new Date(
                        p.consent_record.recorded_at,
                      ).toLocaleDateString()}）`
                    : "—"}
                </td>
                <td>
                  <button className="secondary" onClick={() => void remove(p)}>
                    删除
                  </button>
                </td>
              </tr>
            ))}
            {persons.length === 0 && (
              <tr>
                <td colSpan={4} className="muted">
                  还没有登记人员
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
