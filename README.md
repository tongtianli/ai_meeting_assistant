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
# 健康检查（无需鉴权）
curl http://localhost:8000/api/health

# 换取 Bearer Token（MVP 单默认用户，口令见 .env 的 AUTH_PASSWORD）
TOKEN=$(curl -s -X POST http://localhost:8000/api/auth/token \
  -H 'Content-Type: application/json' -d '{"password":"dev-password"}' \
  | python3 -c "import sys,json;print(json.load(sys.stdin)['access_token'])")
AUTH="Authorization: Bearer $TOKEN"

# 上传录音并触发处理管道（mp3/wav/m4a/mp4）
curl -X POST http://localhost:8000/api/meetings -H "$AUTH" \
  -F "file=@meeting.mp3" -F "title=项目周会"

# 会议列表
curl -H "$AUTH" http://localhost:8000/api/meetings

# 查询处理状态（uploaded → transcoding → transcribing → done/failed）
curl -H "$AUTH" http://localhost:8000/api/meetings/{id}

# 完整转录（segments 入库即可用，不依赖摘要阶段）
curl -H "$AUTH" http://localhost:8000/api/meetings/{id}/segments

# 原文下载（纯文本，带时间戳与说话人真名）
curl -H "$AUTH" -O http://localhost:8000/api/meetings/{id}/transcript

# Speaker 重命名（写 SpeakerBinding，追加式可审计；person_id 立即物化）
curl -X POST -H "$AUTH" -H 'Content-Type: application/json' \
  http://localhost:8000/api/meetings/{id}/speaker-bindings \
  -d '{"speaker_label":"speaker_001","name":"Tim"}'

# 换发音频短时签名播放 URL（支持 range，audio 标签直接可用）
curl -H "$AUTH" http://localhost:8000/api/meetings/{id}/audio-url

# 结构化会议纪要（最新版本；含总结/讨论/决策/TODO 与 segment 溯源）
curl -H "$AUTH" http://localhost:8000/api/meetings/{id}/summary

# Word 导出（docxtpl 实时渲染；重命名 Speaker 后导出立即用真名）
curl -H "$AUTH" -O -J http://localhost:8000/api/meetings/{id}/export.docx

# 失败后重试（各阶段幂等）
curl -X POST -H "$AUTH" http://localhost:8000/api/meetings/{id}/retry
```

## 处理管道（PRD §2）

```
上传 → 转码归一化(ffmpeg 16kHz mono wav) → ASR(转写+说话人分离+对齐)
    → segments 入库 + speaker embedding 留存(voice_samples 暗桩)
    → LLM 摘要(map-reduce → 结构化 JSON → Summary/ActionItem 入库)
    → Word 导出(docxtpl 模板实时渲染，用户点击时生成，不占管道)
```

- ASR provider 可替换（`ASR_PROVIDER` 环境变量）：
  - `mock`（默认）：确定性剧本，用于本地开发与联调
  - `funasr`：本地推理（Paraformer-zh 转写 + CAM++ 说话人分离/声纹），
    数据不出域。安装可选依赖后启用：
    ```bash
    uv sync --extra funasr        # 依赖较重（torch 等）
    ASR_PROVIDER=funasr uv run uvicorn app.main:app
    ```
    首次运行自动从 ModelScope 下载模型（约 1-2GB）；CPU 可推理，
    长音频耗时较长。支持热词注入（provider 接口 hotwords 参数）
  - 云 provider（阿里云/腾讯云等）在 `app/services/asr/__init__.py` 注册即可接入
- LLM Router（`LLM_PROVIDERS` 环境变量，逗号分隔优先级）：
  - `gemini`（默认主力，Gemini Flash）+ `glm`（GLM Flash 中文兜底），
    两家都走 OpenAI 兼容端点，一份实现两组配置；分别需要
    `GEMINI_API_KEY` / `GLM_API_KEY`
  - `mock`：无 key 联调用，输出确定性纪要 JSON
  - 传输错误/限流自动降级到下一个 provider；JSON schema 校验失败带错误
    反馈重试；全部失败时降级为纯文本纪要（`_meta.degraded=true`）
  - 注意：Gemini 免费档数据可能被用于训练，真实敏感会议建议付费档
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
  templates/         # Word 纪要模板（docxtpl；换公司模板直接替换 docx）
  services/
    storage.py       # 音频存储抽象（本地磁盘实现）
    word_export.py   # Summary JSON → Word（docxtpl 渲染）
    llm/             # LLM Router（Gemini/GLM/mock）+ prompt 管理
    summarize.py     # map-reduce 摘要 + schema 校验 + 降级
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
