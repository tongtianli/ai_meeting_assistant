"""应用日志配置：让 app.* 的结构化日志真正落到 stdout。

uvicorn 只配置自己的 logger（uvicorn/uvicorn.access，propagate=False），
root logger 无 handler——检索元数据、用量审计、质量诊断等 INFO 日志
此前在生产一条都看不到。这里给 root 挂一个 stdout handler：
- 幂等按**本模块打标的 handler** 识别（`_ama_handler`）：不会因环境里
  已有 FileHandler（StreamHandler 子类）或别人的 WARNING 级 handler
  而误判"已配置"，stdout 可见性是硬承诺；
- 重复调用只同步级别（root 与本模块 handler 一起调）；
- 级别由 LOG_LEVEL 控制（默认 INFO）；不动 uvicorn 的 logger，
  不会重复打印 access 日志。
"""
import logging
import sys

from app.core.config import settings

_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"


def setup_logging() -> None:
    root = logging.getLogger()
    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    root.setLevel(level)
    for handler in root.handlers:
        if getattr(handler, "_ama_handler", False):
            handler.setLevel(level)  # 重复调用：只更新级别，不重复挂
            return
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(_FORMAT))
    handler._ama_handler = True  # type: ignore[attr-defined]
    root.addHandler(handler)
