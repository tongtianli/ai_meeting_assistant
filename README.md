# AI Meeting Assistant

将会议录音自动转化为结构化会议纪要。详见 [PRD v2](docs/PRD_AI_Meeting_Assistant_v2.md)。

## 技术栈

- Python 3.11+ / FastAPI
- PostgreSQL 16 + pgvector（结构化数据 + 向量一库管理）
- SQLAlchemy 2.0 (async) + Alembic 迁移
- ffmpeg（转码归一化：统一 16kHz 单声道 wav）
- 依赖管理：[uv](https://docs.astral.sh/uv/)

## 本地开发

```bash
# 0. 前置依赖：ffmpeg（macOS: brew install ffmpeg / Ubuntu: apt install ffmpeg）

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

# 6. 运行测试（需要数据库在跑，DB 相关用例会在无库时自动跳过）
uv run pytest
```

## API 速览

```bash
# 健康检查
curl http://localhost:8000/api/health

# 上传录音并触发处理管道（mp3/wav/m4a/mp4）
curl -X POST http://localhost:8000/api/meetings \
  -F "file=@meeting.mp3" -F "title=项目周会"

# 查询处理状态（uploaded → transcoding → transcribing → done/failed）
curl http://localhost:8000/api/meetings/{id}

# 失败后重试（各阶段幂等）
curl -X POST http://localhost:8000/api/meetings/{id}/retry
```

## 处理管道（PRD §2）

```
上传 → 转码归一化(ffmpeg 16kHz mono wav) → ASR(转写+说话人分离+对齐)
    → segments 入库 + speaker embedding 留存(voice_samples 暗桩)
    → [任务4接入] LLM 摘要 → Word 导出
```

- ASR provider 可替换（`ASR_PROVIDER` 环境变量），当前内置 `mock`
  （确定性剧本，用于本地开发与联调）；云 provider 在
  `app/services/asr/__init__.py` 注册即可接入
- 存储层抽象（`app/services/storage.py`）：MVP 本地磁盘，二期换对象存储
  预签名直传时管道不变

## 项目结构

```
app/
  main.py            # FastAPI 入口
  core/config.py     # 环境变量配置（pydantic-settings）
  db/                # engine / session / Base
  models/            # SQLAlchemy 模型（PRD §5 全部 10 张表，含二期暗桩字段）
  schemas/           # API 出入参（pydantic）
  api/routes/        # 路由（health / meetings）
  services/
    storage.py       # 音频存储抽象（本地磁盘实现）
    transcode.py     # ffmpeg 转码归一化
    asr/             # ASR provider 接口 + mock 实现
    pipeline.py      # 异步处理管道（状态机 + 阶段幂等重试）
alembic/             # 数据库迁移
tests/               # pytest（无库时 DB 用例自动跳过）
```

## 数据模型要点（PRD §5）

- 全表 UUID 主键 + `created_at` / `updated_at`
- Speaker（会议内标签）与 Person（全局身份）分离；`speaker_bindings` 追加式可审计
- 暗桩字段一期即建：`transcript_segments.person_id / embedding`、`voice_samples`（`person_id` 可空 + 模型版本号）、多用户 `user_id`
- 删除 Person 物理级联删除全部 VoiceSample（含 embedding），满足合规要求（PRD §9.4）
- 迁移会种子一个默认用户（MVP 单用户，id `00000000-0000-0000-0000-000000000001`）
