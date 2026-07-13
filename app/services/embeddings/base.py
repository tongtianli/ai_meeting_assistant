"""文本 embedding 抽象（RAG 检索用，PRD Feature 5）。

与 LLM Router 不同：embedding 不能降级混用——不同模型的向量空间不通，
检索时问题向量与 segment 向量必须来自同一模型。故做成单一 provider、
无 fallback；缺 key 时明确报错而非静默切换。
"""
from abc import ABC, abstractmethod


class EmbeddingError(RuntimeError):
    pass


class Embedder(ABC):
    name: str
    model: str

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """把一批文本转成向量；返回顺序与输入一致。"""
