# 会议问答 RAG 优化技术设计 v1

> 状态：Draft
>
> 适用范围：AI Meeting Assistant 当前 `dev` 架构
>
> 目标：在不引入新基础设施的前提下，逐步提升会议问答的召回率、上下文完整性、多轮追问能力和可解释性。

## 1. 背景

当前会议问答已经具备以下基础能力：

- 使用 `TranscriptSegment` 作为检索单位；
- 使用 Embedding + pgvector 做语义召回；
- 通过 `meeting_id` 限定单场会议；
- 根据 `person_id` 对问题中出现的真实说话人姓名做结构化补召回；
- 回答返回真实 segment 引用、时间戳和说话人信息；
- 引用由程序反查数据库，避免模型伪造原文和时间戳。

当前实现仍存在以下主要限制：

1. 单个 segment 过碎，完整事实经常跨越多轮发言；
2. 多轮追问中的代词和省略表达无法直接用于检索；
3. 同时询问多个说话人时，统一 `LIMIT` 可能导致候选被某一人占满；
4. 纯向量召回无法稳定处理编号、金额、日期、版本号和英文缩写等精确实体；
5. 缺少统一的候选融合、检索元数据和离线评估机制；
6. 证据不足时，模型仍可能生成看似合理但缺少依据的回答。

## 2. 设计原则

1. **原文是事实来源**：会议纪要属于派生数据，会议事实问答优先以 transcript 为依据。
2. **先做低成本改造**：优先复用 FastAPI、PostgreSQL、pgvector 和现有 LLM Router。
3. **检索与上下文组装分离**：先召回 anchor，再扩展上下文，不在单个函数内混合全部逻辑。
4. **所有引用可验证**：模型只能引用本轮提供的 segment，程序负责最终校验。
5. **阶段化交付**：每个 Phase 独立 PR，避免一次性引入过多变量。
6. **先评估再换模型**：Embedding、Top K、窗口大小和 Reranker 必须通过真实会议测试集评估。

## 3. 总体架构

```text
用户输入
   ↓
意图识别 + 多轮问题改写
   ↓
候选召回
   ├── pgvector 语义召回
   ├── person_id 说话人召回
   └── PostgreSQL 关键词/精确实体召回
   ↓
候选融合与去重
   ↓
邻居 Segment 扩展
   ↓
可选 Reranker
   ↓
LLM 生成回答
   ↓
程序校验引用
   ↓
时间戳跳播 + 原文证据 + 置信度
```

## 4. 分阶段计划

## Phase 1：多轮问题改写、上下文扩展与多人召回

### 4.1 目标

在不新增数据库表和外部服务的前提下，解决当前最明显的问答准确率问题。

本阶段包含：

1. 多轮问题改写；
2. 检索命中后的邻居 segment 扩展；
3. 多说话人分别召回；
4. 基本检索元数据记录；
5. 对应单元测试和集成测试。

本阶段不包含：

- 关键词召回；
- RRF；
- Reranker；
- 新建 TranscriptChunk 表；
- Graph RAG；
- 跨会议检索；
- 更换向量数据库。

### 4.2 多轮问题改写

#### 问题

当前系统直接对本轮 `question` 生成 embedding。以下追问缺少独立语义：

```text
用户：叶玉娇提出了什么风险？
AI：……
用户：她后来给解决方案了吗？
```

第二句应改写为：

```text
叶玉娇后来是否针对她提出的风险给出了解决方案？
```

#### 推荐接口

```python
async def rewrite_query(
    question: str,
    recent_messages: list[ChatMessage],
    speaker_names: list[str],
) -> str:
    ...
```

#### 输入

- 当前用户问题；
- 最近 4～6 条聊天消息；
- 当前会议已经绑定的真实说话人姓名；
- 当前会议标题，可选。

#### 输出

只返回一个可独立检索的问题，例如：

```json
{
  "standalone_query": "叶玉娇后来是否针对她提出的风险给出了解决方案？"
}
```

#### 使用规则

- `standalone_query` 用于：
  - 生成 embedding；
  - 匹配真实说话人姓名；
  - 后续关键词提取。
- 原始 `question` 用于最终回答，避免回答语气偏离用户当前表达。
- 无历史上下文或问题已经完整时，允许原样返回。
- 改写失败时降级为原始问题，不阻塞问答。

### 4.3 多说话人分别召回

#### 问题

当前多个 `person_ids` 共用一个 `LIMIT`：

```python
WHERE person_id IN (...)
ORDER BY distance
LIMIT qa_speaker_top_k
```

用户询问“张三和李四分别怎么看这个方案”时，候选可能全部来自同一个人。

#### 推荐实现

```python
async def retrieve_speaker_anchors(
    session: AsyncSession,
    meeting_id: UUID,
    qvec: list[float],
    person_ids: list[UUID],
    per_person_k: int,
) -> list[TranscriptSegment]:
    ...
```

对每个 `person_id` 单独取 Top K，最后合并去重。

#### 配置建议

```text
QA_TOP_K=6
QA_SPEAKER_TOP_K_PER_PERSON=4
QA_MAX_SPEAKER_PERSONS=4
```

当问题匹配人数超过上限时，保留在问题中首次出现的前 N 人，并在检索元数据中记录截断。

### 4.4 邻居 Segment 扩展

#### 问题

完整事实经常跨多个 segment：

```text
[100] 王总：上线时间先延后。
[101] 李工：延到什么时候？
[102] 王总：下周三，周一先交测试版。
```

只命中 `[102]` 时，模型难以判断“下周三”对应正式上线还是测试版。

#### 推荐流程

```text
向量/说话人召回 anchors
        ↓
按 segment id 去重
        ↓
每个 anchor 扩展 seq ± window
        ↓
再次去重
        ↓
按 seq 排序
        ↓
限制最终上下文数量
```

#### 推荐接口

```python
async def expand_neighbor_segments(
    session: AsyncSession,
    meeting_id: UUID,
    anchors: list[TranscriptSegment],
    window: int,
    max_segments: int,
) -> list[TranscriptSegment]:
    ...
```

#### 配置建议

```text
QA_NEIGHBOR_WINDOW=2
QA_MAX_CONTEXT_SEGMENTS=24
```

#### 约束

- 不跨会议；
- 通过 `meeting_id + seq` 查询；
- 最终上下文按 `seq` 排序；
- 引用仍允许落在 expanded context 内的任何 segment；
- anchor 与 expanded segment 都写入检索元数据，便于调试。

### 4.5 检索函数拆分

建议将当前检索流程拆成明确阶段：

```python
standalone_query = await rewrite_query(...)
qvec = await embed_query(standalone_query)

vector_anchors = await retrieve_vector_anchors(...)
speaker_anchors = await retrieve_speaker_anchors(...)
anchors = merge_and_dedupe(vector_anchors, speaker_anchors)

context_segments = await expand_neighbor_segments(...)
answer = await generate_grounded_answer(...)
```

推荐目录结构：

```text
app/services/retrieval/
    query_rewriter.py
    vector_search.py
    speaker_search.py
    context_expander.py
    models.py

app/services/qa.py
    仅保留问答编排、消息落库和引用校验
```

如果本阶段不希望调整目录，也至少在 `qa.py` 内拆成独立函数，避免 `_retrieve()` 同时承担召回、融合和上下文组装。

### 4.6 检索元数据

建议为 assistant 消息记录最小调试信息。优先使用已有 JSON metadata 字段；如果当前模型没有该字段，可先通过日志记录，后续再迁移到数据库。

建议结构：

```json
{
  "original_question": "她后来同意了吗？",
  "standalone_query": "叶玉娇后来是否同意延期到下周三？",
  "matched_person_ids": ["..."],
  "retrieval_strategies": ["vector", "speaker"],
  "anchor_segment_ids": ["..."],
  "context_segment_ids": ["..."],
  "truncated": false,
  "latency_ms": {
    "rewrite": 120,
    "embedding": 80,
    "retrieval": 45,
    "generation": 900
  }
}
```

### 4.7 Phase 1 测试要求

至少覆盖：

1. 无历史消息时，问题原样或等价返回；
2. 代词追问能改写为包含真实姓名和前文主题的独立问题；
3. 改写服务失败时降级到原问题；
4. 同时命中两个说话人时，两人均有候选；
5. 同一说话人的多个 speaker label 不产生重复 person；
6. anchor 扩展后包含前后邻居；
7. 多个 anchor 的重叠邻居能正确去重；
8. 最终 context 按 seq 排序；
9. context 数量不会超过上限；
10. citation 只能引用本轮 context 中的 segment；
11. 原有按姓名提问测试继续通过；
12. 原有普通向量问答测试继续通过。

### 4.8 Phase 1 验收标准

- “她/他/这个方案/后来呢”等多轮追问可以结合最近对话完成检索；
- 同时询问两名说话人时，两人的发言都能进入候选；
- 跨 2～3 个连续 segment 的事实可以完整进入上下文；
- 不引入新的外部服务；
- 现有 API 响应结构保持兼容；
- 全量后端测试通过；
- 前端类型检查和构建通过。

## Phase 2：关键词和精确实体召回

### 5.1 目标

解决向量检索不擅长的精确字段：

- 项目编号；
- API 名称；
- 日期；
- 金额；
- 版本号；
- 英文缩写；
- 文件名称；
- 专有名词。

### 5.2 第一版实现

暂不引入 Elasticsearch。基于 PostgreSQL 实现轻量召回：

1. 从 `standalone_query` 中提取精确 token；
2. 对 transcript text 使用 `ILIKE` 或 `pg_trgm`；
3. 将关键词候选并入 vector 和 speaker candidates；
4. 统一去重后再扩展邻居。

可优先支持：

```python
patterns = {
    "version": r"\bv?\d+(?:\.\d+)+\b",
    "code": r"\b[A-Z]{2,10}-?\d+\b",
    "number": r"\d+(?:\.\d+)?",
    "date": r"\d{1,2}\s*[月/-]\s*\d{1,2}",
}
```

中文普通词的分词检索后续再评估 `pg_jieba`、ParadeDB 或 Elasticsearch，不在第一版关键词召回中强制引入。

### 5.3 候选融合

三路召回后不建议简单 append。推荐使用 RRF：

```text
score = Σ 1 / (rrf_k + rank)
```

建议：

```text
RRF_K=60
```

候选数据结构：

```python
class RetrievalCandidate(BaseModel):
    segment_id: UUID
    sources: set[str]
    vector_rank: int | None = None
    speaker_rank: int | None = None
    keyword_rank: int | None = None
    fused_score: float = 0
```

### 5.4 Phase 2 验收标准

- `API-203`、`v2.1.4`、`29.5 万`、明确日期等查询能稳定命中原文；
- vector、speaker、keyword 候选统一去重；
- 可通过日志或 metadata 看到每个候选的召回来源；
- 无新增独立搜索集群。

## Phase 3：可靠性、评估与可选 Reranker

### 6.1 证据不足时拒答

建议回答 Schema 增加：

```json
{
  "answer": "会议原文中没有找到明确结论。",
  "cited_segment_seqs": [],
  "confidence": "low",
  "insufficient_evidence": true
}
```

程序侧规则：

- `insufficient_evidence=false` 时，引用不能为空；
- 所有引用必须属于本轮 context；
- 明确日期、金额、负责人等事实必须有对应引用；
- 没有证据时禁止根据常识补全。

### 6.2 离线评估集

从真实会议整理至少 50～100 个问题：

```json
{
  "meeting_id": "...",
  "question": "王经理最后确定的上线日期是什么？",
  "expected_segment_ids": ["..."],
  "expected_answer": "下周三",
  "category": "date",
  "speaker": "王经理"
}
```

覆盖类型：

- 指定说话人；
- 日期；
- 金额；
- 行动项；
- 决策；
- 多轮代词；
- 多人对比；
- 跨 segment 信息；
- 否定信息；
- 无答案问题。

建议指标：

- Recall@5；
- Recall@10；
- MRR；
- 引用准确率；
- 答案事实正确率；
- 无答案拒答率；
- P50/P95 延迟。

### 6.3 Reranker 引入条件

只有满足以下条件后再评估 Reranker：

- 候选经常超过 20；
- vector + speaker + keyword 三路召回均已上线；
- 离线测试显示 Top K 噪声明显；
- 可以接受额外推理延迟和成本。

推荐流程：

```text
召回 30～50 个候选
        ↓
Reranker 选 Top 8～12 anchors
        ↓
邻居扩展
        ↓
生成回答
```

## 7. 会议纪要与原文问答的路由

后续聊天入口建议区分数据源：

```python
class ChatIntent(BaseModel):
    intent: Literal[
        "transcript_query",
        "summary_query",
        "summary_edit",
        "summary_verify",
        "summary_undo",
    ]
    requires_retrieval: bool
    standalone_query: str | None = None
```

行为建议：

| 用户意图 | 数据源 |
|---|---|
| 会议中最后决定了什么 | Transcript RAG |
| 当前纪要有哪些行动项 | 最新 Summary |
| 把第二条负责人改成李明 | 最新 Summary，不查原文 |
| 根据原文检查负责人是否正确 | Summary + Transcript RAG |
| 撤销刚才的修改 | Summary 版本历史 |

此路由不属于 Phase 1 的强制范围，可在纪要对话编辑功能扩展时单独实现。

## 8. Embedding 生命周期

当前首次问答时懒生成 embedding 可以保留作为兜底，但生产环境建议逐步迁移到会议处理管道：

```text
transcribing
→ segments 入库
→ embedding
→ summarizing
→ done
```

至少记录：

```text
embedding_provider
embedding_model
embedding_dimension
embedding_updated_at
```

不要只依赖向量维度判断模型是否变化，因为不同模型可能维度相同。

这部分建议作为独立 PR，不与 Phase 1 同时实现。

## 9. Claude 执行说明

下一次实现时，请先阅读：

1. `docs/PRD_AI_Meeting_Assistant_v2.md`
2. `docs/TECH_DESIGN_MEETING_RAG_V1.md`

第一个 RAG 优化 PR **仅实现 Phase 1**，不要实现 Phase 2 和 Phase 3。

建议 PR 标题：

```text
会议问答增强：多轮问题改写、邻居上下文扩展与多人召回
```

PR 描述必须列出：

- 已实现内容；
- 明确未实现内容；
- 配置项变化；
- 数据库迁移情况；
- 新增测试；
- 全量测试结果；
- 已知限制。

## 10. Phase 1 完成定义

Phase 1 只有在以下条件全部满足时才算完成：

- 多轮问题改写已接入实际检索链路；
- 改写后的问题同时用于 embedding 和说话人姓名匹配；
- 多说话人按 person 独立召回；
- anchor 和 context expansion 已分离；
- 邻居扩展支持去重、排序和总量限制；
- 回答引用只能来自本轮 context；
- 关键检索元数据可被日志或数据库追踪；
- 新增测试覆盖本文 Phase 1 测试要求；
- 全量后端测试通过；
- 前端类型检查和构建通过；
- 未引入 Phase 2、Phase 3 范围内的额外复杂度。
