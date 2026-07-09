import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { login } from "../api/meetings";

export default function LoginPage() {
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const navigate = useNavigate();

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError("");
    try {
      await login(password);
      navigate("/");
    } catch (err) {
      setError(err instanceof Error ? err.message : "登录失败");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="container" style={{ maxWidth: 420, paddingTop: 96 }}>
      <div className="card">
        <h1 style={{ marginBottom: 16 }}>AI Meeting Assistant</h1>
        <form onSubmit={submit}>
          <div className="form-row">
            <input
              type="password"
              placeholder="访问口令"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              autoFocus
            />
          </div>
          {error && <div className="error">{error}</div>}
          <button disabled={busy || !password}>登录</button>
        </form>
      </div>
    </div>
  );
}
