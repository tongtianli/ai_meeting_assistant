"""确定性 Mock embedder：无 key 联调与测试用。

同文本恒得同向量；词重叠越多向量越接近（哈希词袋 + L2 归一化），
使 cosine 检索在测试里可区分相关/无关 segment。
"""
import hashlib
import math
import re

from app.services.embeddings.base import Embedder

_DIM = 64


def _vectorize(text: str, dim: int = _DIM) -> list[float]:
    vec = [0.0] * dim
    for token in re.findall(r"\w+", text.lower()):
        h = int(hashlib.md5(token.encode()).hexdigest(), 16)
        vec[h % dim] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


class MockEmbedder(Embedder):
    name = "mock"
    model = "mock-embed"

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [_vectorize(t) for t in texts]
