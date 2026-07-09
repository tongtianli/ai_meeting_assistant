"""OpenAI 兼容 provider：Gemini 与 GLM 都暴露 OpenAI 兼容端点，
一份实现 + 两组配置即覆盖两家（base_url / api_key / model）。
"""
import httpx

from app.services.llm.base import LLMError, LLMProvider, LLMResponse

_TIMEOUT = httpx.Timeout(180.0, connect=10.0)


class OpenAICompatProvider(LLMProvider):
    def __init__(self, name: str, base_url: str, api_key: str, model: str) -> None:
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model

    async def complete(
        self,
        system: str,
        user: str,
        json_mode: bool = True,
        temperature: float = 0.2,
    ) -> LLMResponse:
        payload: dict = {
            "model": self.model,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                resp = await client.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                )
        except httpx.HTTPError as exc:
            raise LLMError(f"{self.name}: transport error: {exc}") from exc
        if resp.status_code != 200:
            raise LLMError(
                f"{self.name}: HTTP {resp.status_code}: {resp.text[:300]}"
            )
        data = resp.json()
        try:
            text = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"{self.name}: malformed response: {data}") from exc
        if not text:
            raise LLMError(f"{self.name}: empty completion")
        usage = data.get("usage") or {}
        return LLMResponse(
            text=text,
            provider=self.name,
            model=data.get("model", self.model),
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
        )
