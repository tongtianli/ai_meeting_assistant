"""统一 LLM provider 接口（PRD §4：屏蔽多厂商差异）。"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum


class LLMTaskType(str, Enum):
    """任务级路由的任务类型（Tech Design M4 §8.1）。

    不同任务价值密度不同：最终纪要/纪要编辑优先消耗 GLM-4.5-Air 赠送额度，
    问答/改写/分类走免费 Flash——路由顺序按任务在配置中独立指定。
    """

    SUMMARY_MAP = "summary_map"
    SUMMARY_FINAL = "summary_final"
    SUMMARY_EDIT = "summary_edit"
    QA_ANSWER = "qa_answer"
    QUERY_REWRITE = "query_rewrite"
    INTENT_CLASSIFY = "intent_classify"


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
