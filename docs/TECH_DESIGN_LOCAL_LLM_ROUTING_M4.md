# Tech Design：GLM 主力模型、任务级路由与成本保护

> 状态：Draft for implementation  
> 目标分支：`dev`  
> 适用项目：AI Meeting Assistant  
> 部署位置：美国  
> 目标机器：Mac mini M4 16GB  
> 主要任务：中文会议纪要生成、结构化信息抽取、会议问答与纪要编辑

## 1. 背景

当前项目的默认 LLM 路由为：

```text
Gemini -> GLM
```

现状：

- Gemini 免费模型可用性不稳定，经常触发 fallback。
- GLM 在中文会议纪要、结构化输出和纪要范文模仿方面的实际效果较好。
- 项目部署在美国地区的 Mac mini M4 16GB 上，本地运行 8B 级模型可行，但不适合作为复杂长会议最终纪要的生产主力。
- 当前所有任务共用统一 Provider 顺序，无法针对纪要生成、问答、查询改写和纪要编辑分别选择模型。
- 项目缺少 Token 用量审计、资源包到期提醒和付费模型预算熔断。

用户当前智谱资源包如下，均于 **2026-10-10 08:40:18** 到期：

| 资源包 | 剩余额度 | 适用范围 | 本项目价值 |
|---|---:|---|---|
| 搜索模型 | 100 次 | search-std/search-pro 等 | 当前基本不用 |
| 通用基础模型推理 | 1,996,701 tokens | 所有按 Token 计费的基础模型 | 备用额度 |
| GLM-4.6V | 6,000,000 tokens | 多模态视觉推理 | 当前基本不用 |
| GLM-4.5-Air | 12,000,000 tokens | GLM-4.5-Air 推理 | 纪要主力资源包 |

## 2. 核心结论

本设计采用 **云端 GLM 主力、本地模型非主链路** 的方案。

推荐默认路由：

```text
纪要生成：GLM-4.5-Air -> GLM 免费 Flash -> Gemini
会议问答：GLM 免费 Flash -> Gemini -> GLM-4.5-Air
纪要编辑：GLM-4.5-Air -> GLM 免费 Flash -> Gemini
查询改写/意图分类：GLM 免费 Flash -> Gemini
```

原因：

1. GLM-4.5-Air 已有 1200 万 Token 专属赠送额度，到期后不结转，应该优先用于高价值任务。
2. GLM 免费 Flash 适合短输入、短输出和高频轻量任务。
3. Gemini 保留为跨供应商灾备，避免单一 Provider 故障。
4. Mac mini M4 16GB 本地 Qwen 8B 只保留开发、实验和未来轻量任务位置，不承担正式最终纪要。

## 3. 设计目标

- 将 GLM-4.5-Air 设为最终纪要生成和复杂纪要编辑的主模型。
- 将 GLM 免费 Flash 设为问答、查询改写和意图分类的主模型。
- 保持 Gemini 作为跨供应商 fallback。
- 保持现有纪要范文机制、Summary Schema、版本管理和引用溯源不变。
- 支持任务级 Provider 路由，而不是所有调用共用 `LLM_PROVIDERS`。
- 记录每次调用的模型、任务类型、Token、延迟、失败和 fallback 原因。
- 跟踪赠送额度使用进度，并在到期前给出预警。
- 防止资源包用完后静默进入高额付费。
- 保留未来本地模型接入能力，但不将其作为本阶段生产目标。

## 4. 非目标

本阶段不包含：

- 替换 ASR Provider。
- 重构整个 RAG 检索链路。
- 部署本地 Qwen 作为正式纪要主模型。
- 引入独立 GPU 服务或 Kubernetes。
- 自动充值或自动购买资源包。
- 将不同 embedding 模型生成的向量混用。
- 使用 GLM-4.6V 处理纯文本会议纪要。

## 5. 现有纪要范文机制兼容性

现有纪要范文机制在 Provider 调用前完成：

```text
加载 SummaryExample
-> 拼接风格规则和 few-shot 范文
-> LLM Router
-> Provider 生成 JSON
-> SummaryContent 校验
```

该机制与 Provider 无关，因此 GLM-4.5-Air、GLM Flash 和 Gemini 都可以直接复用。

必须保留：

- 范文只注入 single-pass 或 reduce 最终阶段。
- map 阶段不重复注入完整范文。
- `SummaryContent` Pydantic 校验。
- `_meta.model`。
- `_meta.style_examples`。
- `source_segment_seq` 引用约束。
- 结构化输出失败后的修复重试和 Provider fallback。

建议新增质量检查：

- 范文中的独有人名、数字、项目名不得在本次逐字稿无依据出现。
- TODO 的负责人、日期和来源 seq 必须可追溯。
- 参与人必须来自转录或已绑定说话人。

## 6. Provider 角色划分

### 6.1 最终会议纪要

推荐路由：

```text
glm_air -> glm_flash -> gemini
```

职责：

- `glm_air`：主模型，优先消耗 GLM-4.5-Air 1200 万 Token 专属赠送包。
- `glm_flash`：免费备用，处理 Air 暂时不可用、429、超时或结构化失败。
- `gemini`：跨供应商灾备，防止 GLM 整体故障。

### 6.2 长会议 map 阶段

推荐路由：

```text
glm_flash -> glm_air -> gemini
```

map 阶段偏事实抽取，优先使用免费 Flash，降低 Air 额度消耗。

建议 map 输出独立的事实抽取 Schema，而不是直接输出完整最终纪要：

```json
{
  "topics": [],
  "decisions": [],
  "todos": [],
  "risks": [],
  "open_questions": [],
  "source_segment_seqs": []
}
```

最终 reduce 阶段再注入范文并使用 `glm_air` 生成正式 `SummaryContent`。

### 6.3 纪要编辑

推荐路由：

```text
glm_air -> glm_flash -> gemini
```

原因：纪要编辑要求保持未修改字段、范文文风和引用，属于高价值结构化任务。

### 6.4 会议问答

推荐路由：

```text
glm_flash -> gemini -> glm_air
```

问答只输入检索后的少量片段，免费 Flash 通常足够。仅复杂问题或连续结构化失败时使用 Air。

### 6.5 查询改写与意图分类

推荐路由：

```text
glm_flash -> gemini
```

这些任务输入输出都很短，不应优先消耗 Air 专属额度。

### 6.6 Embedding

第一阶段继续使用单一 `embedding-3`，避免同时修改摘要和 RAG 向量空间。

必须单独记录：

- embedding 请求次数。
- 输入 Token 或字符数。
- 实际模型名。
- 是否由通用 1,996,701 Token 资源包抵扣。

不能假设 GLM 文本模型免费就代表 `embedding-3` 免费。

## 7. 资源包使用策略

### 7.1 GLM-4.5-Air 专属包

额度：

```text
12,000,000 tokens
到期：2026-10-10 08:40:18
```

用途优先级：

1. 最终纪要生成。
2. 复杂纪要编辑。
3. 用户主动重新生成纪要。
4. 复杂 QA fallback。

不建议用于：

- 意图分类。
- 查询改写。
- 简单会议问答。
- map 阶段的每个分块，除非 Flash 质量不达标。

粗略容量估算：

| 单场累计 Token | 可覆盖会议数 |
|---:|---:|
| 33,000 | 约 363 场 |
| 55,000 | 约 218 场 |
| 100,000 | 约 120 场 |

实际容量必须以应用记录的 usage 为准。

### 7.2 通用基础模型包

额度：

```text
1,996,701 tokens
到期：2026-10-10 08:40:18
```

用途：

- Air 专属包不适用的其他付费基础模型。
- FlashX 或其他低价模型的备用测试。
- 可能的 embedding 消耗，但必须先在控制台确认适用范围。

该资源包不应被当作无限 fallback。

### 7.3 GLM-4.6V 专属包

额度：

```text
6,000,000 tokens
```

当前纯文本会议项目不使用。

未来只有在支持以下能力时再启用：

- 白板照片理解。
- PPT/图表截图分析。
- PDF 页面图像分析。
- 会议视频画面理解。

### 7.4 搜索调用包

100 次搜索额度当前不进入默认链路。

除非未来明确实现联网事实核验，否则会议纪要不得主动调用搜索模型，以免把会议内部内容与外部搜索混杂。

## 8. 任务级 Router 设计

### 8.1 任务类型

```python
class LLMTaskType(str, Enum):
    SUMMARY_MAP = "summary_map"
    SUMMARY_FINAL = "summary_final"
    SUMMARY_EDIT = "summary_edit"
    QA_ANSWER = "qa_answer"
    QUERY_REWRITE = "query_rewrite"
    INTENT_CLASSIFY = "intent_classify"
```

### 8.2 Provider 实例

同一个 GLM API Key 下，将不同模型视为不同路由节点：

```text
glm_air
glm_flash
glm_paid_fallback
gemini
mock
```

不要只保留单一 `glm` Provider 名称，否则无法区分任务路由、预算和额度统计。

### 8.3 推荐环境变量

```env
# 任务级路由
SUMMARY_MAP_PROVIDERS=glm_flash,glm_air,gemini
SUMMARY_FINAL_PROVIDERS=glm_air,glm_flash,gemini
SUMMARY_EDIT_PROVIDERS=glm_air,glm_flash,gemini
QA_PROVIDERS=glm_flash,gemini,glm_air
QUERY_REWRITE_PROVIDERS=glm_flash,gemini
INTENT_CLASSIFY_PROVIDERS=glm_flash,gemini

# GLM 中国区现有账号
GLM_BASE_URL=https://open.bigmodel.cn/api/paas/v4
GLM_API_KEY=
GLM_AIR_MODEL=glm-4.5-air
GLM_FLASH_MODEL=glm-4.7-flash

# Gemini 灾备
GEMINI_API_KEY=
GEMINI_MODEL=

# 资源包信息，仅用于应用侧预警，不代表服务商真实余额
GLM_AIR_GRANT_TOTAL_TOKENS=12000000
GLM_GENERAL_GRANT_TOTAL_TOKENS=1996701
GLM_GRANT_EXPIRES_AT=2026-10-10T08:40:18+08:00

# 调用控制
LLM_REQUEST_TIMEOUT_SECONDS=180
LLM_MAX_PROVIDER_ATTEMPTS=3
LLM_RETRY_MAX_PER_PROVIDER=1
```

现有 `LLM_PROVIDERS` 暂时保留，作为尚未迁移任务的默认路由。

## 9. Provider 接口

保持统一结构化接口：

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

返回值：

```python
class LLMResult(BaseModel):
    content: BaseModel
    provider: str
    model: str
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    latency_ms: int
    fallback_index: int
    finish_reason: str | None
```

如果 Provider 不返回 usage，允许字段为空，但必须记录该事实，不能伪造 Token 数。

## 10. 失败与 fallback 策略

### 10.1 可重试错误

同一 Provider 仅对以下错误重试：

- 网络超时。
- HTTP 429。
- HTTP 502/503/504。
- 服务端临时不可用。

采用指数退避，并尊重 `Retry-After`。

### 10.2 结构化输出错误

以下情况最多进行一次修复重试，然后切换 Provider：

- JSON 解析失败。
- Pydantic Schema 校验失败。
- 输出为空。
- 输出被截断。
- 引用 seq 非法。

### 10.3 总尝试次数

```text
同一 Provider 最多 2 次
整个任务最多尝试 3 个 Provider
```

禁止在 GLM Air、Flash 和 Gemini 之间无限循环。

### 10.4 质量门槛

即使 Schema 通过，也必须检查：

- 参与人是否来自转录或绑定名单。
- TODO 的 source seq 是否存在。
- 负责人和日期是否有依据。
- 范文中的独有事实是否被错误复制。
- 最终纪要是否为空模板。
- topics、decisions、todos 是否与逐字稿基本一致。

质量检查失败可触发 fallback，但必须记录 `quality_gate_failed`。

## 11. 用量与额度审计

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
- total_tokens nullable
- success
- fallback_index
- error_type nullable
- error_message nullable
- latency_ms
- created_at
```

必须记录成功和失败调用。

建议指标：

- 每个 Provider 成功率。
- 每个模型 fallback 比例。
- 每场会议平均 Token。
- summary_map、summary_final 分别消耗多少 Token。
- GLM-4.5-Air 累计 Token。
- 预计剩余可生成会议数。
- 预计资源包耗尽日期。
- 资源包到期前剩余未使用比例。

应用侧估算公式：

```text
estimated_remaining = configured_grant_total - tracked_total_tokens
```

该值仅用于预警，控制台资源包余额仍是最终真值。

## 12. 额度预警与付费保护

### 12.1 到期预警

对 2026-10-10 到期的资源包设置：

- 到期前 30 天提醒。
- 到期前 14 天提醒。
- 到期前 7 天提醒。
- 到期前 1 天提醒。

提醒中展示：

- 应用侧累计消耗。
- 控制台人工录入剩余值（如有）。
- 最近 7 天平均每日消耗。
- 预计到期时剩余额度。

### 12.2 消耗阈值

```text
使用 70%：提示观察
使用 85%：提示评估路由
使用 95%：仅高价值任务允许使用 Air
使用 100%：按配置决定是否允许付费继续
```

### 12.3 防止静默付费

建议配置：

```env
GLM_ALLOW_PAID_AFTER_GRANT=false
GLM_AIR_SOFT_LIMIT_TOKENS=11400000
GLM_GENERAL_SOFT_LIMIT_TOKENS=1890000
```

当应用侧累计量达到软上限：

- `glm_air` 从低价值任务路由中移除。
- 最终纪要可以切到 `glm_flash -> gemini`。
- 管理端显示“资源包接近耗尽”。

由于服务商可能按自己的优先级抵扣多个资源包，应用侧不能保证完全阻止计费；必须同时在智谱控制台配置余额预警或停机保护。

## 13. 长会议摘要策略

保留现有 map-reduce 架构，并优化模型分配：

```text
逐字稿
-> 分块
-> GLM Flash 做事实抽取 map
-> GLM Air 做最终 reduce + 范文定型
-> SummaryContent 校验
-> 质量检查
-> 保存 Summary 版本
```

建议初始切块：

```text
SUMMARY_CHUNK_TARGET_TOKENS=5000
SUMMARY_CHUNK_MAX_TOKENS=7000
SUMMARY_CHUNK_OVERLAP_SEGMENTS=2
```

最终 reduce 输入为局部结构化结果，不再次输入完整逐字稿，以降低重复 Token 消耗。

## 14. 美国部署注意事项

当前使用中国区端点：

```text
https://open.bigmodel.cn/api/paas/v4
```

由于服务器位于美国，必须监控：

- DNS/连接失败率。
- 首 Token 延迟。
- 总响应时间。
- 429、5xx 比例。
- 连续请求成功率。

暂不自动迁移至 Z.AI 国际站，因为：

- 中国区与国际区账号、API Key 和赠送额度不一定互通。
- 当前资源包明确属于现有中国区账号。
- 迁移国际站可能导致无法使用现有赠送额度。

可在独立测试环境评估国际站，但不得在未确认资源包互通前替换生产端点。

## 15. 本地 M4 的定位

Mac mini M4 16GB 不作为正式最终纪要主模型。

保留本地能力用于：

- 开发环境 mock 替代。
- 查询改写实验。
- 意图分类实验。
- 短文本格式修复。
- 云端完全不可用时的人工启用应急模式。

本阶段不要求部署 Ollama 或 MLX-LM。

未来如要启用，本地模型必须作为可选 Provider：

```text
local_qwen
```

不得改变当前 GLM 主路由的默认行为。

## 16. 实施阶段

### Phase 1：GLM 多模型实例与任务级路由

范围：

1. 增加 `LLMTaskType`。
2. 将 `glm_air` 和 `glm_flash` 作为两个独立路由节点。
3. 支持不同任务配置不同 Provider 顺序。
4. 最终纪要切换为 `glm_air -> glm_flash -> gemini`。
5. 问答与轻量任务切换为 `glm_flash -> gemini`。
6. 保持现有纪要范文机制不变。
7. 增加 Router 和 fallback 测试。

验收：

- 纪要最终生成首先调用 GLM-4.5-Air。
- QA 和查询改写默认不消耗 Air。
- Air 失败后能切换到 Flash。
- GLM 整体失败后能切换到 Gemini。
- `_meta.model` 正确记录实际 Provider/模型。
- 现有测试全部通过。

### Phase 2：Token 用量审计

范围：

1. 增加 `llm_usage_records`。
2. 记录成功和失败调用。
3. 区分 `summary_map`、`summary_final`、`qa_answer` 等任务。
4. 显示累计 Air Token 和每场会议平均消耗。
5. 增加资源包到期提醒。

验收：

- 可按模型和任务统计 Token。
- 可估算剩余会议数量。
- 可显示 fallback 原因和失败率。

### Phase 3：额度软限制与付费熔断

范围：

1. 增加 Air 和通用资源包软上限。
2. 接近耗尽时自动调整低价值任务路由。
3. 到期或耗尽后禁止静默付费。
4. 管理端显示明确状态。

验收：

- 达到软上限后，查询改写和 QA 不调用 Air。
- 达到硬限制后，按配置停止 Air 或切换免费模型。
- 用户能区分模型故障、额度熔断和资源包到期。

### Phase 4：长会议 map/reduce Token 优化

范围：

1. map 使用事实抽取 Schema。
2. map 默认使用 GLM Flash。
3. reduce 使用 GLM Air 和范文。
4. 避免 reduce 再传完整逐字稿。
5. 增加范文污染和事实一致性检查。

该阶段独立 PR 实现。

## 17. 测试计划

### 单元测试

- task type 路由选择。
- `glm_air` 与 `glm_flash` 使用不同模型名。
- Air 成功时不调用 Flash/Gemini。
- Air 429 后 Flash fallback。
- GLM 失败后 Gemini fallback。
- QA 默认不调用 Air。
- JSON 修复重试。
- Provider 总尝试次数限制。
- Token usage 写入。
- 资源包阈值路由调整。
- 到期预警计算。

### 集成测试

- 10 分钟、60 分钟和 120 分钟会议。
- 中英文混合会议。
- 多人、日期、金额和行动项。
- 范文注入开启和关闭。
- GLM Air 超时。
- GLM Flash 429。
- Gemini 灾备成功。
- usage 字段缺失的响应。

### 质量评估

至少准备 20 场脱敏历史会议，对比：

- 当前 `glm-4-flash`。
- `glm-4.5-air`。
- 新版免费 Flash。
- Gemini fallback。

指标：

- JSON 首次通过率。
- TODO 遗漏率。
- 负责人准确率。
- 截止日期准确率。
- 引用 seq 合法率。
- 范文文风一致性。
- 范文事实污染率。
- 平均延迟。
- 平均 Token。

## 18. 回滚方案

保留旧路由开关：

```env
LLM_ROUTING_MODE=legacy
LLM_PROVIDERS=gemini,glm
```

新模式：

```env
LLM_ROUTING_MODE=task_based
```

发生问题时可回滚为：

```env
LLM_ROUTING_MODE=legacy
GLM_MODEL=glm-4-flash
```

数据库新增 usage 表必须向后兼容；旧 Summary 版本不得重写。

## 19. 推荐默认配置

```env
LLM_ROUTING_MODE=task_based

SUMMARY_MAP_PROVIDERS=glm_flash,glm_air,gemini
SUMMARY_FINAL_PROVIDERS=glm_air,glm_flash,gemini
SUMMARY_EDIT_PROVIDERS=glm_air,glm_flash,gemini
QA_PROVIDERS=glm_flash,gemini,glm_air
QUERY_REWRITE_PROVIDERS=glm_flash,gemini
INTENT_CLASSIFY_PROVIDERS=glm_flash,gemini

GLM_BASE_URL=https://open.bigmodel.cn/api/paas/v4
GLM_AIR_MODEL=glm-4.5-air
GLM_FLASH_MODEL=glm-4.7-flash

GLM_AIR_GRANT_TOTAL_TOKENS=12000000
GLM_GENERAL_GRANT_TOTAL_TOKENS=1996701
GLM_GRANT_EXPIRES_AT=2026-10-10T08:40:18+08:00
GLM_ALLOW_PAID_AFTER_GRANT=false
GLM_AIR_SOFT_LIMIT_TOKENS=11400000
GLM_GENERAL_SOFT_LIMIT_TOKENS=1890000
```

模型 ID 在上线前必须通过智谱控制台和一次真实 API 调用确认；如果账号暂不支持 `glm-4.7-flash`，则使用控制台列出的当前免费 Flash 稳定型号，不在代码中静默替换。

## 20. 最终决策

当前阶段推荐：

```text
主力最终纪要：GLM-4.5-Air
长会议 map：GLM 免费 Flash
轻量 QA/改写/分类：GLM 免费 Flash
跨供应商灾备：Gemini
本地 M4 16GB：实验和未来轻量任务，不进生产主链路
```

应在 2026-10-10 前优先使用 GLM-4.5-Air 的 1200 万 Token 专属赠送额度，同时建立真实 usage 统计。到期前两周，根据真实每场 Token、质量和稳定性决定：

- 继续付费使用 Air；
- 切换免费 Flash 为主；
- 使用低价 GLM 付费型号；
- 或将 Gemini 调整为主模型。

## 21. 给实现 Agent 的执行要求

下一次实现 PR 只做 **Phase 1**：

- GLM Air/Flash 多模型实例。
- `LLMTaskType`。
- 任务级 Provider 路由。
- 配置和 fallback 测试。
- 保持现有纪要范文机制不变。

不要在同一个 PR 中实现：

- usage 数据库表。
- 额度熔断。
- map/reduce Schema 重构。
- 本地 Qwen。
- 本地 embedding。
- RAG 检索优化。

PR 描述必须列出：

1. 已实现内容。
2. 未实现内容。
3. 新增环境变量。
4. 测试结果。
5. 回滚方式。
