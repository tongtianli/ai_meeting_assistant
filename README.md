# AI Meeting Assistant

将会议录音自动转化为结构化会议纪要。详见 [PRD v2](docs/PRD_AI_Meeting_Assistant_v2.md)。

## 技术栈

- Python 3.11+ / FastAPI
- PostgreSQL 16 + pgvector（结构化数据 + 向量一库管理）
- SQLAlchemy 2.0 (async) + Alembic 迁移
- 依赖管理：[uv](https://docs.astral.sh/uv/)

## 本地开发

```bash
# 1. 启动数据库（pgvector PostgreSQL）
docker compose up -d db

# 2. 安装依赖
uv sync

# 3. 配置环境变量
cp .env.example .env

# 4. 执行数据库迁移（建全部表 + 种子默认用户）
uv run alembic upgrade head

# 5. 启动服务
uv run uvicorn app.main:app --reload
# 健康检查: curl http://localhost:8000/api/health

# 6. 运行测试
uv run pytest
```

## 项目结构

```
app/
  main.py            # FastAPI 入口
  core/config.py     # 环境变量配置（pydantic-settings）
  db/                # engine / session / Base
  models/            # SQLAlchemy 模型（PRD §5 全部 10 张表，含二期暗桩字段）
  api/routes/        # 路由
alembic/             # 数据库迁移
tests/               # pytest
```

## 数据模型要点（PRD §5）

- 全表 UUID 主键 + `created_at` / `updated_at`
- Speaker（会议内标签）与 Person（全局身份）分离；`speaker_bindings` 追加式可审计
- 暗桩字段一期即建：`transcript_segments.person_id / embedding`、`voice_samples`（`person_id` 可空 + 模型版本号）、多用户 `user_id`
- 删除 Person 物理级联删除全部 VoiceSample（含 embedding），满足合规要求（PRD §9.4）
- 迁移会种子一个默认用户（MVP 单用户，id `00000000-0000-0000-0000-000000000001`）
