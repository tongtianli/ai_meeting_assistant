import logging

from app.core.config import settings
from app.services.llm.base import (
    LLMError,
    LLMProvider,
    LLMResponse,
    LLMTaskType,
)
from app.services.llm.mock import MockLLMProvider
from app.services.llm.openai_compat import OpenAICompatProvider
from app.services.llm.router import LLMExhaustedError, LLMRouter

logger = logging.getLogger(__name__)

# 任务类型 → Settings 中的 provider 顺序字段（task_based 模式）
_TASK_PROVIDER_FIELDS: dict[LLMTaskType, str] = {
    LLMTaskType.SUMMARY_MAP: "summary_map_providers",
    LLMTaskType.SUMMARY_FINAL: "summary_final_providers",
    LLMTaskType.SUMMARY_EDIT: "summary_edit_providers",
    LLMTaskType.QA_ANSWER: "qa_providers",
    LLMTaskType.QUERY_REWRITE: "query_rewrite_providers",
    LLMTaskType.INTENT_CLASSIFY: "intent_classify_providers",
    LLMTaskType.QUALITY_CHECK: "quality_check_providers",
}


def _build_provider(name: str) -> LLMProvider | None:
    if name == "mock":
        return MockLLMProvider()
    if name == "gemini":
        if not settings.gemini_api_key:
            return None
        return OpenAICompatProvider(
            "gemini", settings.gemini_base_url, settings.gemini_api_key, settings.gemini_model
        )
    # 同一 GLM key 下的多个路由节点：模型不同、名称不同，
    # 额度消耗与 fallback 统计才能按节点区分（Tech Design M4 §8.2）
    if name in ("glm", "glm_air", "glm_flash"):
        if not settings.glm_api_key:
            return None
        model = {
            "glm": settings.glm_model,
            "glm_air": settings.glm_air_model,
            "glm_flash": settings.glm_flash_model,
        }[name]
        return OpenAICompatProvider(
            name, settings.glm_base_url, settings.glm_api_key, model
        )
    raise ValueError(f"unknown LLM provider: {name!r}")


def _build_providers(order: str) -> list[LLMProvider]:
    providers: list[LLMProvider] = []
    for name in [n.strip() for n in order.split(",") if n.strip()]:
        provider = _build_provider(name)
        if provider is None:
            logger.warning("LLM provider %r skipped: api key not configured", name)
        else:
            providers.append(provider)
    return providers


def provider_order_for(task: LLMTaskType | None) -> str:
    """返回任务的 provider 顺序串；legacy 模式或未指定任务时用 LLM_PROVIDERS。"""
    if task is None or settings.llm_routing_mode != "task_based":
        return settings.llm_providers
    return getattr(settings, _TASK_PROVIDER_FIELDS[task])


def build_router(task: LLMTaskType | None = None) -> LLMRouter:
    """按任务类型构建 Router；缺 key 的 provider 跳过并告警。

    task_based 模式下任务路由全部不可用时回退 LLM_PROVIDERS——
    保证 mock 联调（LLM_PROVIDERS=mock）无需逐任务配置也能工作。
    """
    order = provider_order_for(task)
    providers = _build_providers(order)
    if not providers and order != settings.llm_providers:
        logger.warning(
            "no provider available for task %s (order=%r), "
            "falling back to LLM_PROVIDERS",
            task.value if task else "-",
            order,
        )
        providers = _build_providers(settings.llm_providers)
    return LLMRouter(providers, task=task)


__all__ = [
    "LLMError",
    "LLMExhaustedError",
    "LLMProvider",
    "LLMResponse",
    "LLMRouter",
    "LLMTaskType",
    "MockLLMProvider",
    "OpenAICompatProvider",
    "build_router",
    "provider_order_for",
]
