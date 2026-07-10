"""统一 LLM provider 接口（PRD §4：屏蔽多厂商差异）。"""
from abc import ABC, abstractmethod
from dataclasses import dataclass


class LLMError(RuntimeError):
    """provider 调用失败（网络、限流、非 2xx、响应缺失）。"""


@dataclass
class LLMResponse:
    text: str
    provider: str
    model: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class LLMProvider(ABC):
    name: str
    model: str

    @abstractmethod
    async def complete(
        self,
        system: str,
        user: str,
        json_mode: bool = True,
        temperature: float = 0.2,
    ) -> LLMResponse:
        """单轮补全；json_mode 时要求模型输出 JSON 对象。"""
