"""任务 3：会议详情——完整转录、Speaker 重命名、原文下载、音频签名 URL。"""
import io
import wave

from fastapi.testclient import TestClient

from app.core.config import settings
from app.main import app
from tests.conftest import auth_headers, requires_db

pytestmark = requires_db


def _wav_bytes(seconds: float = 2.0, rate: int = 16000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x00" * int(rate * seconds))
    return buf.getvalue()


def _create_done_meeting(client, headers) -> str:
    resp = client.post(
        "/api/meetings",
        files={"file": ("m.wav", _wav_bytes(), "audio/wav")},
        data={"title": "详情测试"},
        headers=headers,
    )
    assert resp.status_code == 201
    return resp.json()["id"]


def test_transcript_rename_download_flow(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    with TestClient(app) as client:
        headers = auth_headers(client)
        mid = _create_done_meeting(client, headers)

        # 1. 完整转录：初始 speaker_name 回退为 label，person_id 为空（暗桩未绑定）
        resp = client.get(f"/api/meetings/{mid}/segments", headers=headers)
        assert resp.status_code == 200
        segs = resp.json()["segments"]
        assert len(segs) > 0
        assert [s["seq"] for s in segs] == list(range(len(segs)))
        first = segs[0]
        assert first["speaker_name"] == first["speaker_label"]
        assert first["person_id"] is None

        # 2. Speaker 重命名：speaker_001 → Tim
        resp = client.post(
            f"/api/meetings/{mid}/speaker-bindings",
            json={"speaker_label": "speaker_001", "name": "Tim"},
            headers=headers,
        )
        assert resp.status_code == 201, resp.text
        binding = resp.json()
        assert binding["person_name"] == "Tim"
        assert binding["confirmed_by"] == "human"

        # 3. 重命名立即生效：speaker_name 与物化的 person_id
        segs = client.get(f"/api/meetings/{mid}/segments", headers=headers).json()[
            "segments"
        ]
        for s in segs:
            if s["speaker_label"] == "speaker_001":
                assert s["speaker_name"] == "Tim"
                assert s["person_id"] == binding["person_id"]
            else:
                assert s["person_id"] is None

        # 4. 重复绑定 → 追加式取代（可审计）
        resp = client.post(
            f"/api/meetings/{mid}/speaker-bindings",
            json={"speaker_label": "speaker_001", "name": "王小明"},
            headers=headers,
        )
        assert resp.status_code == 201
        segs = client.get(f"/api/meetings/{mid}/segments", headers=headers).json()[
            "segments"
        ]
        renamed = [s for s in segs if s["speaker_label"] == "speaker_001"]
        assert all(s["speaker_name"] == "王小明" for s in renamed)

        # 5. 未知 label 返回 404
        resp = client.post(
            f"/api/meetings/{mid}/speaker-bindings",
            json={"speaker_label": "speaker_999", "name": "无此人"},
            headers=headers,
        )
        assert resp.status_code == 404

        # 6. 原文下载：带时间戳与真名，attachment 头
        resp = client.get(f"/api/meetings/{mid}/transcript", headers=headers)
        assert resp.status_code == 200
        assert "attachment" in resp.headers["content-disposition"]
        text = resp.text
        assert "[00:00:00] 王小明:" in text
        assert "speaker_002" in text  # 未绑定的保持 label


def test_audio_signed_url_and_range(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    with TestClient(app) as client:
        headers = auth_headers(client)
        mid = _create_done_meeting(client, headers)

        resp = client.get(f"/api/meetings/{mid}/audio-url", headers=headers)
        assert resp.status_code == 200
        body = resp.json()
        assert body["expires_in"] > 0
        url = body["url"]

        # 签名 URL 无需 Authorization 头（audio 标签场景）
        resp = client.get(url)
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("audio/")

        # range 请求（音频定位播放，PRD §9.2）
        resp = client.get(url, headers={"Range": "bytes=0-99"})
        assert resp.status_code == 206
        assert len(resp.content) == 100

        # 篡改 token 拒绝
        resp = client.get(f"/api/audio/{mid}?token=forged")
        assert resp.status_code == 403
