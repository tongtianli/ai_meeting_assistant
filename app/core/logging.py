"""应用日志配置：让 app.* 的结构化日志真正落到 stdout。

uvicorn 只配置自己的 logger（uvicorn/uvicorn.access，propagate=False），
root logger 无 handler——检索元数据、用量审计、质量诊断等 INFO 日志
此前在生产一条都看不到。这里给 root 挂一个 StreamHandler(stdout)：
- 幂等：root 已有 handler（pytest caplog、重复 import）时只调级别不重复挂；
- 级别由 LOG_LEVEL 控制（默认 INFO）；
- 不动 uvicorn 的 logger，不会重复打印 access 日志。
"""
import logging
import sys

from app.core.config import settings

_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"


def setup_logging() -> None:
    root = logging.getLogger()
    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    root.setLevel(level)
    # 已有 handler（含 pytest caplog / 二次调用）就不再挂，防止重复输出
    if not any(isinstance(h, logging.StreamHandler) for h in root.handlers):
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(_FORMAT))
        root.addHandler(handler)
