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

1. 多轮问题改写（与意图分类合并为一次调用）；
2. 检索命中后的邻居 segment 扩展（anchor 优先预算）；
3. 多说话人分别召回（含每人保底 speaker anchor）；
4. 最小证据不足拒答；
5. embedding 模型身份校验（唯一一列迁移）；
6. 基本检索元数据记录（默认脱敏）；
7. 对应单元测试和集成测试。

本阶段不包含：

- 关键词召回；
- RRF；
- Reranker；
- 新建 TranscriptChunk 表；
- Graph RAG；
- 跨会议检索；
- 更换向量数据库；
- `app/services/retrieval/` 目录 package 化（留 Phase 2）。

> **本文的权威修订见下方 §4.1.1**（PR #16 评审后与人工确认）。当 §4.1.1 与
> §4.2–§4.8 的细节冲突时，**以 §4.1.1 为准**。

### 4.1.1 Phase 1 定稿修订（评审已确认，优先于 §4.2–§4.8 的细节）

> 来源：PR #16《RAG v1 技术设计 review 修订提案》讨论定案。与下文冲突时以本节为准。

**A. 意图识别与查询改写合并为一次 LLM 调用**

- 保留现有二值意图 `query` / `edit`（**不引入 `other`、不重命名 `edit`**）；仅在意图分类的结构化输出中新增 `standalone_query`。
- 无历史消息：直接用原始问题作 `standalone_query`，不额外调用模型。
- 有历史消息：意图分类 + 改写**合并为一次** `INTENT_CLASSIFY` 调用，输出 `{intent, standalone_query}`。
- 结构化输出失败：降级为原始问题 + 默认 `query`，不阻塞问答。
- 不新增独立 `QUERY_REWRITE` 串行调用点。
- 时序：`handle_chat` 先 commit 用户消息再分类，加载"最近 N 条"须在插入前取历史，或按 `message_id` 排除当前消息。

**B. 改写上下文（按需补引用证据）**

- 默认输入：最近 N 条**用户**消息 + 上一轮 `standalone_query` + 当前会议说话人名单。
- **指代门控**：本轮命中「他 / 她 / 它 / 这个 / 那个 / 该方案 / 后来 / 那件事 / 上面说的」等指代标记时，才附带上一轮**合法 `cited_segment_ids` 的少量原文**（`QA_REWRITE_CITED_MAX_SEGMENTS` 默认 3 + 字符预算）；无指代不附带，省 token。
- cited 原文作事实语境；assistant 自由文本仅作指代参考，**不作**人名 / 日期 / 金额 / 结论的事实来源（写进 prompt 硬约束）。

**C. 说话人匹配用「原问题 ∪ 改写问题」并集**

- `matched = match_people(original_question) | match_people(standalone_query)`——改写可能遗漏原问题里的第二个人名。
- 改写臆造、且在原问题 / 历史用户消息 / 合法引用原文中均不存在的人名，**不得**作为强 speaker 召回条件，记为可疑并忽略。

**D. anchor 优先的上下文预算（替换 §4.4 的"按 seq 排序后简单截断"）**

- 区分"优先级选择"与"最终展示顺序"：
  1. 按融合排名（Phase 1 = 合并去重后的列表序）先纳入每个 anchor **自身**；
  2. 按距离 ±1、±2… **逐圈**为各 anchor 补邻居（不是一个 anchor 一次吃满窗口）；
  3. 达 `QA_MAX_CONTEXT_SEGMENTS` 即停；
  4. **仅在选择完成后**按 `seq` 排序喂给 LLM。
- `max_segments < anchor 数`时保留最高排名 anchors 并记录淘汰数；anchor 优先级恒高于邻居；按 seq 排序只影响展示、不影响取舍。

**E. vector / speaker anchor 保底配额（策略 a）**

- 每个明确命中的 person（受 `QA_MAX_SPEAKER_PERSONS` 限制，按在问题中首次出现序）先取 `QA_MIN_SPEAKER_ANCHORS_PER_PERSON`（默认 1）条 speaker anchor 设为**受保护**，最先进入 anchor 集合。
- 其余按「vector（距离序）+ 剩余 speaker」去重后追加其后，构成 D 的 `anchors_by_rank`。
- 保证点名问题（"王建国说了什么"）的 speaker anchor 不被 vector 挤掉，语义问题仍保留最强 vector anchor。**不引入 RRF**（跨源融合留 Phase 2）。

**F. 最小证据不足拒答（提前到 Phase 1）**

- QA prompt 硬约束：只依据本轮 context 作答；无明确证据答"会议原文中没有找到明确结论"；**禁止**据常识 / 会议标题 / assistant 历史 / 范文补全；日期 / 金额 / 负责人 / 版本号 / 决策须有直接引用。
- `QaAnswer` 加可选 `insufficient_evidence: bool = False`（向后兼容，前端忽略未知字段）。程序规则：`insufficient_evidence=false` 的事实性回答须有 ≥1 条**本轮 context 内**的合法引用，否则回退拒答；非法引用不接受。**不硬拒**合法的"未找到"。

**G. embedding 模型身份校验（提前到 Phase 1）**

- `meetings` 增一列 `embedding_model_key`（格式如 `glm/embedding-3:1024`）——**Phase 1 唯一迁移**。
- `ensure_embeddings` 的 stale 判断 = 任一向量为空 **OR** 维度不符 **OR 模型身份不符**；身份变化对该会议整场重嵌。旧数据 `embedding_model_key` 为空视为 stale，首次访问重嵌并回填。
- 存 `Meeting` 而非 `TranscriptSegment`：整场 all-or-nothing 重嵌下一列即可，避免逐 segment 重复存储。

**H. 检索 metadata 与隐私**

- Phase 1 走**结构化日志**（不加 schema）。默认字段：`question_hash`（sha256）、`standalone_query_changed`、各来源候选数、`anchor_count / kept_anchor_count / evicted_anchor_count`、`context_segment_count`、`truncated`、分段 `latency_ms`。
- 明文（原始问题 / standalone_query）仅在 `QA_RETRIEVAL_DEBUG_TEXT=true` 时记录；**绝不**记 API Key / 整段逐字稿。

**I. 配置项**

| 配置 | 默认 | 说明 |
|---|---|---|
| `QA_TOP_K` | 6 | 向量 anchor 数（不变）|
| `QA_SPEAKER_TOP_K_PER_PERSON` | 4 | 原 `qa_speaker_top_k` 改名（语义"每人 K"）；**旧名保留一版兼容** |
| `QA_MAX_SPEAKER_PERSONS` | 4 | 匹配人数上限，超出按首次出现截断并记录 |
| `QA_MIN_SPEAKER_ANCHORS_PER_PERSON` | 1 | 每个命中 person 的保底 speaker anchor |
| `QA_NEIGHBOR_WINDOW` | 2 | 邻居 ± 窗口 |
| `QA_MAX_CONTEXT_SEGMENTS` | 24 | 最终 context 上限 |
| `QA_REWRITE_CITED_MAX_SEGMENTS` | 3 | 指代改写附带的上轮引用原文数上限 |
| `QA_RETRIEVAL_DEBUG_TEXT` | false | 是否记录问题明文 |

**J. 文件组织**：Phase 1 **不**做 `app/services/retrieval/` package 化；在 `qa.py` 内拆函数 + 新增 `query_intent.py`（合并意图 + 改写）。package 化留 Phase 2 三路候选 + RRF 落地时再评估。

**K. Phase 1 测试增量**（在 §4.7 基础上补充）：无历史不触发额外改写调用；多轮一次调用同出 `intent` 与 `standalone_query`；改写失败降级原问题；assistant 错误历史不被固化为事实；原问题人名被改写遗漏仍参与 speaker 召回；改写臆造人名不成强 speaker 条件；高排名 anchor 位于会议末尾不因按 seq 截断丢失；anchor 优先于邻居；点名问题保底 speaker anchor 保留；无证据可拒答且不伪造引用；同维不同模型触发整场重嵌；默认日志不含问题明文。

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

#### 热路径与时序约束

- **首轮直接跳过改写调用**：`recent_messages` 为空时无需调用 LLM，直接使用原始
  `question`。判断"问题是否完整"本身若还要过一次模型，等于给每轮问答都加一次串行
  往返；用"有无历史"作为硬门控，把改写成本限制在真正的多轮追问上。
- **历史加载时序**：`handle_chat` 先 commit 用户消息再作答，因此加载"最近 N 条"时会把
  当前这句也带进去。实现时应在插入前取历史，或按 message_id 排除当前消息。
- **与意图分类的关系（评审定稿：Phase 1 即合并，不分开实现）**：意图分类与查询改写
  **在 Phase 1 就合并为一次 LLM 调用**同时产出 `intent` 与 `standalone_query`，不新增独立
  `QUERY_REWRITE` 串行调用点。详见 §4.1.1-A。（原"Phase 1 可先各自独立实现"的说法已废弃。）

### 4.3 多说话人分别召回

#### 现状

当前 `qa.py::_retrieve` **已经按人独立召回**：对每个命中的 `person_id` 单独执行一次
`ORDER BY distance LIMIT qa_speaker_top_k`，最后合并去重。因此"多人同问时候选被单人
占满"的问题在 `dev` 上已不存在，本节不是修 bug。

```python
# 当前实现（已按人分配配额）
for pid in person_ids:
    boosted = ... base.where(person_id == pid).order_by(distance).limit(qa_speaker_top_k)
```

#### 本阶段增量

1. **人数上限与截断**：当问题匹配到的说话人过多时，按在问题中首次出现的顺序保留前
   N 人，并在检索元数据中记录 `truncated=true`；
2. **配置改名**：把 `qa_speaker_top_k` 语义显式化为"每人 K"，避免误读为全局配额；
3. 顺带把这段召回抽成独立函数，便于 Phase 2 并入关键词候选。

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

对每个 `person_id` 单独取 Top K，最后合并去重（保持现有行为，仅补充人数上限与截断记录）。

#### 配置建议

```text
QA_TOP_K=6
QA_SPEAKER_TOP_K_PER_PERSON=4   # 现 qa_speaker_top_k 改名，语义为"每人 K"
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

> **注（评审定稿）**：下面的"按 seq 排序 → 限制数量"流程若实现为"排序后简单截断"，会丢掉
> 会议后半段的高排名 anchor。**Phase 1 采用 §4.1.1-D 的 anchor 优先预算**（先按排名纳入 anchor
> 自身，再逐圈补邻居，最后才按 seq 排序仅用于展示），本节流程仅作背景。

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

> 注：Phase 1 的 `merge_and_dedupe` 只做简单合并去重，会在 Phase 2 被 `RetrievalCandidate` +
> RRF（见 §5.3）替换。因此本阶段不必在合并/打分逻辑上过度投入，把 anchor 集合去重、
> 保序输出即可。

### 4.6 检索元数据

建议为 assistant 消息记录最小调试信息。注意 `ChatMessage` 当前**没有通用 metadata/JSONB
字段**（仅有 `cited_segment_ids`），因此 Phase 1 先通过结构化日志记录，后续需要持久化时再
加字段迁移，不在本阶段引入 schema 变更。

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
    "amount": r"\d+(?:\.\d+)?\s*(?:万|亿|元|块|美元|USD|RMB)",  # 带单位才算金额
    "date": r"\d{1,2}\s*[月/-]\s*\d{1,2}",
}
```

> **不要用裸 `number` 模式** `\d+(?:\.\d+)?`：它会匹配几乎所有数字（seq、时长秒数、页码
> 等噪声），再配 `ILIKE '%token%'` 会产出大量低价值候选，反而稀释 RRF。只在带单位/明确
> 上下文（金额、编号、版本）时才提取数字。

> **性能约束**：`ILIKE '%token%'` 带前导通配符用不上 B-tree 索引，会全表扫 transcript text。
> 上关键词召回前**必须为 `transcript_segments.text` 建 `pg_trgm` GIN 索引**（`CREATE
> EXTENSION pg_trgm` + `gin (text gin_trgm_ops)`），这是一条要写进迁移的硬性前置。

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

### 8.1 当前实现的潜在错检索风险（建议提前处理）

`qa.py::ensure_embeddings` 目前**只靠维度判断模型是否变化**：

```python
stale = any(s.embedding is None or len(s.embedding) != expected_dim for s in segments)
```

一旦换成**同维度的另一个 embedding 模型**（例如 `embedding-3` 与某个同为 1024 维的模型），
问题向量与库内向量落在互不兼容的空间，检索会"看起来在跑、结果全错"，且**没有任何报错**。
考虑到 Phase 1 重度依赖向量召回质量，这不是纯未来工作，而是懒加载路径上的一个静默缺陷。

因此建议**把最小版的 `embedding_model` 标识记录提前到 Phase 1 或紧随其后**：在 segment（或
meeting）上记下生成向量的 `embedding_model`，`ensure_embeddings` 同时比对**模型标识**而不仅是
维度，模型变化即触发全量重嵌。完整的四字段元数据与管道化生成仍可作为独立 PR。

这部分（除上述最小标识外）建议作为独立 PR，不与 Phase 1 同时实现。

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
