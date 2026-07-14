# Tech Design：Mac mini M4 本地 LLM 与云端回退路由

> 状态：Draft for implementation  
> 目标分支：`dev`  
> 适用项目：AI Meeting Assistant  
> 部署位置：美国  
> 主要任务：中文会议纪要生成、结构化信息抽取、会议问答辅助

## 1. 背景

当前项目的 LLM 路由为：

```text
Gemini -> GLM
```

现有配置中，纪要生成默认使用 Gemini，失败后回退到智谱 `glm-4-flash`；RAG embedding 默认使用 `embedding-3`。

当前主要问题：

1. Gemini 免费模型在高峰期或免费配额下可用性不稳定，纪要生成经常触发 fallback。
2. GLM fallback 缺少调用成本、剩余额度、失败原因和月度预算保护。
3. 所有任务共用同一条 Provider 顺序，无法针对纪要、查询改写、问答等不同任务选择最合适的模型。
4. 项目计划部署在美国地区的 Mac mini M4 上，具备本地运行中小型开源模型的条件，但当前尚未利用本地推理能力。
5. 长会议直接将完整逐字稿送入单个模型，会放大上下文成本、内存占用、输出不稳定和单次失败影响。

## 2. 设计目标

本设计希望达到以下目标：

- 将本地模型作为会议纪要处理的主要计算资源，降低对免费云模型可用性的依赖。
- 保持 Gemini 和 GLM 作为云端 fallback，但只在本地模型失败或质量不合格时触发。
- 对不同任务使用独立路由，而不是所有请求共用 `LLM_PROVIDERS`。
- 对长会议使用分层摘要，避免本地模型一次处理超长上下文。
- 记录每次调用的 Provider、模型、Token、延迟、失败原因和估算成本。
- 对付费 Provider 设置硬预算和熔断，避免 GLM 或其他云模型意外超额。
- 不破坏现有 Provider Router、Summary Schema、版本管理和测试结构。

## 3. 非目标

本阶段不包含：

- 替换 ASR Provider。
- 重构整个 RAG 检索链路。
- 引入 Kubernetes、独立 GPU 集群或复杂调度系统。
- 在第一阶段直接运行 30B 以上 dense 模型。
- 自动购买或充值任何云模型额度。
- 将不同 embedding 模型生成的向量混用。

## 4. 硬件假设

本地节点为普通 Mac mini M4，物理部署在美国。

由于统一内存容量会显著影响可运行模型，设计按以下档位兼容：

| 统一内存 | 推荐本地模型 | 定位 |
|---|---|---|
| 16GB | Qwen 8B 级 4-bit | 分段摘要、行动项抽取、查询改写 |
| 24GB | Qwen 14B 级 4-bit | 推荐，可承担正式纪要生成 |
| 32GB | Qwen 14B 级 4-bit，较大上下文 | 更稳，适合生产主模型 |
| M4 Pro 48GB+ | 30B 左右量化模型可评估 | 非本阶段默认目标 |

第一版应支持 8B 和 14B 两档，通过环境变量切换，不把模型名写死在业务代码中。

## 5. 推荐总体架构

```text
会议逐字稿
   ↓
按议题或 Token 数切分
   ↓
本地模型逐块抽取
   ├── 局部摘要
   ├── 决策
   ├── 行动项
   ├── 风险
   └── 待确认事项
   ↓
本地模型合并为最终 SummaryContent
   ↓
Pydantic Schema 校验
   ├── 成功：保存 Summary 新版本
   └── 失败：本地重试一次
             ↓
        云端 Provider fallback
             ├── Gemini
             └── GLM
```

核心原则：

1. 本地优先。
2. 分层摘要优先于超长上下文直塞。
3. 云端只处理失败或质量不合格的任务。
4. 每次 fallback 都必须可追踪。
5. 付费模型必须受预算限制。

## 6. Provider 角色划分

### 6.1 纪要最终生成

推荐路由：

```text
local_qwen -> gemini -> glm
```

- `local_qwen`：主模型。
- `gemini`：第一云端 fallback。
- `glm`：最后 fallback。

如果 Mac mini 为 16GB：

```text
本地模型负责逐块摘要
Gemini 或 GLM 负责最终合并
```

如果 Mac mini 为 24GB 或 32GB：

```text
本地模型同时负责逐块摘要和最终合并
云端仅在 Schema 校验连续失败时启用
```

### 6.2 查询改写与意图分类

推荐路由：

```text
local_qwen -> glm_free
```

这类任务输入短、输出短，适合全部本地化。

### 6.3 会议问答生成

推荐路由：

```text
local_qwen -> gemini -> glm
```

问答依赖检索后的少量上下文，本地 8B/14B 通常足够。

### 6.4 Embedding

第一阶段维持当前单一 `embedding-3`，避免同时重构摘要与 RAG。

第二阶段可迁移为本地 embedding，例如 BGE-M3 或其他中文/多语模型，但必须：

1. 新增 embedding model/version 元数据。
2. 全量重建已有向量。
3. 禁止新旧模型向量混合检索。

## 7. 本地推理运行时

### 7.1 第一阶段：Ollama

优点：

- 接入快。
- 提供本地 HTTP API。
- 便于快速比较 8B 和 14B 模型。
- 运维成本低。

建议先用 Ollama 完成模型质量基准测试和业务接入。

### 7.2 第二阶段：MLX-LM

当模型和 Prompt 稳定后，可迁移到 MLX-LM：

- 更贴合 Apple Silicon。
- 可更细粒度控制量化、KV Cache 和 Prompt Cache。
- 更适合集成到 Python 服务。
- 可封装为 OpenAI 兼容接口，复用现有 Router。

第一版不要同时接入 Ollama 和 MLX-LM 两套运行时；建议先 Ollama，稳定后再评估迁移。

## 8. 长会议分层摘要

### 8.1 切块策略

输入块必须保留：

- segment seq
- speaker name / label
- start_ms
- end_ms
- text

建议初始参数：

```text
SUMMARY_CHUNK_TARGET_TOKENS=5000
SUMMARY_CHUNK_MAX_TOKENS=7000
SUMMARY_CHUNK_OVERLAP_SEGMENTS=2
```

优先按议题边界切分；议题信息不可用时，再按 Token 预算和 segment 边界切分。

禁止在一个 segment 中间截断。

### 8.2 局部摘要 Schema

每个块输出结构化 JSON：

```json
{
  "overview": "本段讨论概述",
  "topics": [],
  "decisions": [],
  "action_items": [],
  "risks": [],
  "open_questions": [],
  "source_segment_seqs": []
}
```

局部阶段应强调事实抽取，不追求最终文风。

### 8.3 最终合并

最终合并输入为所有局部 JSON，而不是再次输入完整逐字稿。

合并阶段负责：

- 去重。
- 合并同一议题。
- 区分建议与最终决定。
- 统一人名和术语。
- 生成最终 `SummaryContent`。
- 保留可追溯的 source segment seq。

## 9. Router 改造

### 9.1 环境变量

建议新增：

```env
# 按任务划分 Provider 顺序
SUMMARY_CHUNK_PROVIDERS=local_qwen,gemini,glm
SUMMARY_FINAL_PROVIDERS=local_qwen,gemini,glm
QA_PROVIDERS=local_qwen,gemini,glm
QUERY_REWRITE_PROVIDERS=local_qwen,glm
INTENT_CLASSIFY_PROVIDERS=local_qwen,glm

# 本地 OpenAI 兼容服务
LOCAL_LLM_BASE_URL=http://127.0.0.1:11434/v1
LOCAL_LLM_API_KEY=local
LOCAL_LLM_MODEL=qwen3:14b
LOCAL_LLM_TIMEOUT_SECONDS=180
LOCAL_LLM_MAX_CONCURRENCY=1

# 分层摘要
SUMMARY_CHUNK_TARGET_TOKENS=5000
SUMMARY_CHUNK_MAX_TOKENS=7000
SUMMARY_CHUNK_OVERLAP_SEGMENTS=2
SUMMARY_LOCAL_RETRIES=1
SUMMARY_MAX_PROVIDER_ATTEMPTS=3

# 成本保护
LLM_COST_GUARD_ENABLED=true
GLM_MONTHLY_BUDGET_CNY=5
GEMINI_MONTHLY_BUDGET_USD=3
PAID_FALLBACK_DISABLED_WHEN_BUDGET_EXCEEDED=true
```

现有 `LLM_PROVIDERS` 暂时保留，用作未迁移任务的默认路由。

### 9.2 任务类型

建议定义：

```python
class LLMTaskType(str, Enum):
    SUMMARY_CHUNK = "summary_chunk"
    SUMMARY_FINAL = "summary_final"
    QA_ANSWER = "qa_answer"
    QUERY_REWRITE = "query_rewrite"
    INTENT_CLASSIFY = "intent_classify"
    SUMMARY_EDIT = "summary_edit"
```

Router 根据 `task_type` 选择 Provider 列表。

### 9.3 Provider 接口

保持统一接口：

```python
class LLMProvider(Protocol):
    async def generate_structured(
        self,
        *,
        messages: list[Message],
        schema: type[BaseModel],
        task_type: LLMTaskType,
    ) -> LLMResult:
        ...
```

`LLMResult` 应包含：

```python
class LLMResult(BaseModel):
    content: BaseModel
    provider: str
    model: str
    input_tokens: int | None
    output_tokens: int | None
    latency_ms: int
    estimated_cost: Decimal | None
    fallback_index: int
```

## 10. 失败与 fallback 策略

### 10.1 可重试错误

只对以下错误重试同一 Provider：

- 网络超时。
- HTTP 429。
- HTTP 502/503/504。
- 本地服务临时不可达。

### 10.2 不应无限重试的错误

以下情况最多修复重试一次，然后切换 Provider：

- JSON 解析失败。
- Schema 校验失败。
- 输出为空。
- 输出被截断。

### 10.3 Provider 总尝试次数

同一任务建议：

```text
本地模型：最多 2 次
Gemini：最多 1 次
GLM：最多 1 次
整个任务总 Provider 尝试不超过 3 个
```

避免一个会议在多个模型间反复重试造成延迟和费用失控。

### 10.4 质量门槛

即使 Schema 校验通过，也应增加基础质量检查：

- `action_items` 中负责人和截止时间不得无依据补全。
- 纪要不得包含逐字稿中不存在的参与人。
- 输出不得为空或只有模板标题。
- 最终纪要必须包含至少一个 overview 或 topic。
- source segment seq 必须属于当前会议。

质量检查失败时，视为可 fallback 的生成失败。

## 11. GLM 使用与费用保护

当前配置中的纪要 fallback 模型为 `glm-4-flash`。该模型当前通常对应免费 Flash 系列，但免费状态、速率限制和模型别名可能变化，不能假设永久免费。

同时，RAG 使用的 `embedding-3` 是独立模型，其赠送额度和到期时间必须从智谱控制台确认，不能从应用侧推断。

建议：

1. 启动时记录实际调用的模型名。
2. 不依赖模糊别名长期运行，配置中使用明确版本号或稳定型号。
3. 为 GLM 设置月度预算。
4. 在控制台核对：
   - 赠送余额。
   - 资源包剩余 Token。
   - 到期时间。
   - `embedding-3` 实际计费。
5. 预算耗尽后禁止自动切到其他 GLM 付费模型。

## 12. 用量与成本审计

建议新增表：

```text
llm_usage_records
- id
- meeting_id nullable
- user_id nullable
- task_type
- provider
- model
- input_tokens nullable
- output_tokens nullable
- estimated_cost nullable
- currency nullable
- success
- fallback_index
- error_type nullable
- error_message nullable
- latency_ms
- created_at
```

必须记录失败调用，否则无法判断 fallback 频率和真实成本。

建议新增查询指标：

- 每个 Provider 成功率。
- fallback 比例。
- 平均纪要生成耗时。
- 每场会议平均 Token。
- 每场会议平均估算成本。
- 本地模型 Schema 首次通过率。
- GLM 月度累计成本。

## 13. 并发和资源控制

普通 M4 的本地推理应默认串行：

```text
LOCAL_LLM_MAX_CONCURRENCY=1
```

原因：

- 多并发会争抢统一内存。
- 长上下文 KV Cache 容易造成内存压力。
- 并发时单请求延迟可能显著上升。

建议将本地 LLM 调用放入进程内 semaphore。后续任务量增加时，再迁移到独立队列或本地推理服务。

当本地模型繁忙时，不应立即 fallback 到云端；应区分：

- 本地服务失败：允许 fallback。
- 本地服务排队：等待队列。
- 本地任务超时：根据任务类型决定 fallback。

## 14. 数据隐私

默认策略：

- 完整逐字稿优先只在本地处理。
- 云端 fallback 时记录“会议内容已发送到第三方 Provider”。
- 后续可增加会议级配置：

```text
cloud_fallback_allowed: bool
```

如果用户关闭云端 fallback，本地模型失败时应明确返回错误，不得静默上传会议内容。

## 15. 建议代码结构

```text
app/services/llm/
    router.py
    task_types.py
    result.py
    cost_guard.py
    usage.py
    providers/
        local_openai.py
        gemini.py
        glm.py

app/services/summarization/
    chunker.py
    chunk_summary.py
    final_merge.py
    quality.py
    orchestration.py
```

现有代码可分阶段迁移，不要求一次性移动所有文件。

## 16. 实施阶段

### Phase 1：本地 Provider 与任务级路由

范围：

1. 增加 `local_qwen` OpenAI 兼容 Provider。
2. 增加 `LLMTaskType`。
3. 支持按任务配置 Provider 顺序。
4. 纪要任务改为 `local_qwen -> gemini -> glm`。
5. 增加本地并发限制。
6. 保持现有单次整篇摘要流程，先验证兼容性。

验收：

- 本地服务可用时，纪要不调用 Gemini 或 GLM。
- 本地服务关闭时，能够按顺序 fallback。
- 现有测试全部通过。
- 新增 Provider 路由和 fallback 测试。

### Phase 2：分层摘要

范围：

1. 增加 Transcript chunker。
2. 本地模型逐块抽取结构化结果。
3. 合并局部结果生成最终 Summary。
4. 增加质量校验。
5. 保留原有单次摘要作为兼容路径和回滚开关。

验收：

- 60 分钟以上会议可稳定生成纪要。
- 单块失败可单独重试。
- 最终输出通过现有 Summary Schema。
- 本地内存占用在目标机器可接受范围内。

### Phase 3：成本审计与预算熔断

范围：

1. 增加 `llm_usage_records`。
2. 记录成功和失败调用。
3. 增加月度预算计算。
4. 超预算后阻止付费 fallback。
5. 管理端展示 Provider 用量和 fallback 原因。

验收：

- 能按月份统计 GLM/Gemini 使用量。
- 超过配置预算后不再调用受限 Provider。
- 用户能够看到纪要失败是因为预算熔断，而不是模型故障。

### Phase 4：本地 Embedding

范围：

1. 选择本地 embedding 模型。
2. 增加 embedding model/version 字段。
3. 提供重建索引任务。
4. 全量切换后停止使用 `embedding-3`。

该阶段应独立 PR 实现，不与摘要模型迁移混合。

## 17. 测试计划

### 单元测试

- task type 路由选择。
- 本地 Provider 成功。
- 本地失败后 Gemini fallback。
- Gemini 失败后 GLM fallback。
- JSON/Schema 失败修复重试。
- 总尝试次数限制。
- 月度预算熔断。
- 本地并发 semaphore。
- chunk 边界不截断 segment。
- chunk 合并去重。

### 集成测试

- Ollama/MLX 模拟 OpenAI API。
- 10 分钟、60 分钟和 120 分钟会议。
- 中英文混合会议。
- 包含多人、日期、金额和行动项的会议。
- 本地服务中途断开。
- Gemini 429。
- GLM 无额度或鉴权失败。

### 质量测试集

至少准备 20 场脱敏会议，人工标注：

- 关键议题。
- 最终决定。
- 行动项。
- 负责人。
- 截止日期。
- 风险和待确认事项。

比较：

- 当前 Gemini/GLM 路由。
- 本地 8B。
- 本地 14B。
- 分层摘要。
- 云端最终合并。

## 18. 回滚方案

所有改造必须保留旧路由开关：

```env
SUMMARY_PIPELINE_MODE=single_pass
```

新模式：

```env
SUMMARY_PIPELINE_MODE=hierarchical
```

出现本地部署问题时，可立即恢复：

```env
SUMMARY_FINAL_PROVIDERS=gemini,glm
SUMMARY_PIPELINE_MODE=single_pass
```

数据库 Schema 变更应向后兼容；旧 Summary 版本不得重写。

## 19. 推荐默认配置

### Mac mini M4 16GB

```env
LOCAL_LLM_MODEL=qwen3:8b
SUMMARY_CHUNK_PROVIDERS=local_qwen,gemini,glm
SUMMARY_FINAL_PROVIDERS=gemini,glm
QA_PROVIDERS=local_qwen,gemini,glm
LOCAL_LLM_MAX_CONCURRENCY=1
```

### Mac mini M4 24GB/32GB

```env
LOCAL_LLM_MODEL=qwen3:14b
SUMMARY_CHUNK_PROVIDERS=local_qwen,gemini,glm
SUMMARY_FINAL_PROVIDERS=local_qwen,gemini,glm
QA_PROVIDERS=local_qwen,gemini,glm
LOCAL_LLM_MAX_CONCURRENCY=1
```

## 20. 最终决策

推荐采用以下路线：

```text
短期：Ollama + Qwen 8B/14B + 任务级 Router
中期：分层摘要 + 成本审计 + 预算熔断
后期：MLX-LM 优化 + 本地 Embedding
```

对于美国地区的 Mac mini M4，本地 Qwen 应成为主要纪要模型；Gemini 和 GLM 保留为受控 fallback，而不是默认主链路。GLM 免费 Flash 可以继续使用，但不得假设其永久免费；`embedding-3` 的资源包和到期时间需要在智谱控制台单独核验。

## 21. 给实现 Agent 的执行要求

下一次实现 PR 只做 **Phase 1**：

- 本地 OpenAI 兼容 Provider。
- 任务级 Provider 路由。
- 本地并发限制。
- fallback 测试。

不要在同一个 PR 中实现：

- 分层摘要。
- 用量数据库。
- 预算控制。
- 本地 embedding。
- RAG 检索优化。

PR 描述必须列出：

1. 已实现内容。
2. 未实现内容。
3. 配置示例。
4. 测试结果。
5. 已知限制。
