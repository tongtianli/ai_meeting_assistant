# 会议问答 RAG 优化技术设计 v1：Review 修订提案

> 状态：Discussion Draft
>
> 目标：对 `docs/TECH_DESIGN_MEETING_RAG_V1.md` 提出合并前修订项，供 Claude 与人工评审讨论。
>
> 原则：不改变原设计的总体方向，只修正可能导致重复重构、静默错检索、上下文截断失真和回答幻觉的关键点。

## 1. 总体结论

原设计总体方向合理：

- 原文优先于纪要；
- 复用 PostgreSQL、pgvector 和现有 LLM Router；
- anchor 召回与 context expansion 分离；
- 关键词召回先使用 PostgreSQL，不提前引入 Elasticsearch；
- Reranker 仅在真实评测证明有必要后引入；
- Phase 1～3 分阶段交付。

建议保留总体架构，但在 Phase 1 开始实现前调整以下四项为强制要求：

1. 意图识别与多轮 standalone query 改写合并为一次 LLM 调用；
2. context budget 必须保证高排名 anchor 不会因最终按 seq 排序和截断而丢失；
3. 最小证据不足拒答约束提前到 Phase 1；
4. embedding 模型标识校验提前到 Phase 1，不能只比较向量维度。

其余建议作为 Phase 1 的实现细化或 Phase 2 的边界约束。

---

## 2. Phase 1 必改：意图识别与查询改写合并

### 2.1 问题

原设计允许 Phase 1 暂时分别调用“意图分类”和“查询改写”，后续再合并。这会让多轮问答热路径变成：

```text
意图分类 LLM
→ 查询改写 LLM
→ embedding
→ 检索
→ 最终回答 LLM
```

这会增加一次串行网络往返、Token 消耗和失败点，并可能造成两次模型调用对用户意图理解不一致。

### 2.2 修订要求

Phase 1 直接使用一次调用同时产出：

```python
class ChatIntent(BaseModel):
    intent: Literal["query", "summary_edit", "other"]
    standalone_query: str | None = None
```

示例：

```json
{
  "intent": "query",
  "standalone_query": "叶玉娇后来是否针对她提出的延期风险给出了解决方案？"
}
```

规则：

- 无历史消息时，直接使用原始问题作为 `standalone_query`，不为“问题是否完整”额外调用模型；
- 有历史消息时，意图分类与 standalone query 改写合并为一次调用；
- 编辑类意图不进入 Transcript RAG；
- 结构化输出失败时，降级为原始问题并按现有意图规则处理；
- 不新增独立 `QUERY_REWRITE` 串行调用点。

### 2.3 历史上下文规则

历史 assistant 回答可能错误，不能作为事实来源。改写输入优先包含：

1. 最近用户问题；
2. 上一轮 standalone query（如有）；
3. 上一轮合法引用对应的原文片段（如有）；
4. 当前会议真实说话人名单；
5. assistant 回答仅用于理解指代，不得作为人名、日期、金额或结论的事实依据。

### 2.4 说话人匹配

不能只在改写后的问题中匹配姓名。应使用并集：

```python
matched_people = (
    match_people(original_question)
    | match_people(standalone_query)
)
```

原因：改写模型可能遗漏原问题中明确出现的第二个人名。

如果 standalone query 引入了原问题、历史用户消息和合法引用原文中均不存在的人名，该人名不得直接作为强 speaker 召回条件，应记录为可疑改写结果并忽略该新增姓名。

---

## 3. Phase 1 必改：anchor 优先的上下文预算算法

### 3.1 问题

原设计流程中先扩展全部邻居、按 seq 排序，再限制 context 数量。如果实现为排序后简单截断，会出现：

- 高排名但发生在会议后半段的 anchor 被截掉；
- 某个 anchor 的大窗口占满预算，其他 anchor 自身无法进入 context；
- 最终排序顺序错误地影响检索优先级。

### 3.2 修订算法

context 选择必须区分“优先级选择”和“最终展示顺序”。

推荐算法：

```python
selected = OrderedSet()

# 第一轮：按融合排名保留每个 anchor 自身
for anchor in anchors_by_rank:
    add_if_budget(selected, anchor)

# 后续按距离逐圈补邻居
for distance in range(1, window + 1):
    for anchor in anchors_by_rank:
        add_if_budget(selected, segment(anchor.seq - distance))
        add_if_budget(selected, segment(anchor.seq + distance))

# 仅在选择完成后按时间顺序提供给 LLM
return sorted(selected, key=lambda s: s.seq)
```

### 3.3 强制约束

- `max_segments` 小于 anchor 数量时，只保留排名最高的 anchors，并记录被淘汰数量；
- anchor 自身优先级始终高于邻居；
- 邻居按距离 `±1`、`±2` 逐圈填充，而不是一个 anchor 一次吃完整窗口；
- 最终按 seq 排序仅用于阅读上下文，不用于决定保留谁；
- metadata 记录：
  - 原始 anchor 数；
  - 保留 anchor 数；
  - 被预算淘汰 anchor 数；
  - context 是否截断。

### 3.4 必测场景

1. 高排名 anchor 位于会议末尾，不能因按 seq 截断而丢失；
2. 多个 anchor 重叠窗口正确去重；
3. context 达到上限时，anchor 优先于邻居；
4. anchor 数大于预算时，保留最高排名 anchor；
5. 最终输出仍按 seq 排序。

---

## 4. Phase 1 必改：最小证据不足拒答

### 4.1 原因

Phase 1 会扩大可供模型阅读的上下文：

- 多轮改写；
- 说话人补召回；
- 邻居扩展。

上下文更多不等于证据更充分。模型可能从“相关但没有回答问题”的内容中拼出看似合理的结论。

完整 confidence 体系和拒答评估可以留到 Phase 3，但最小拒答规则必须提前。

### 4.2 Phase 1 最小要求

回答 Prompt 明确：

- 只能依据本轮 context 回答；
- 没有明确证据时，回答“会议原文中没有找到明确结论”；
- 禁止根据常识、会议标题、历史 assistant 回答或范文补全事实；
- 日期、金额、负责人、版本号和最终决策必须有直接引用依据。

程序校验：

```text
有明确事实答案 → cited_segment_seqs 非空
所有引用 → 必须属于本轮 context
引用非法 → 不接受模型输出，走已有结构化修复/fallback
```

Phase 1 不强制增加完整 confidence 字段，但允许提前增加：

```json
{
  "answer": "会议原文中没有找到明确结论。",
  "cited_segment_seqs": [],
  "insufficient_evidence": true
}
```

如果为保持 API 兼容不加字段，则至少通过固定拒答文案和空引用表达证据不足。

---

## 5. Phase 1 必改：embedding 模型身份校验

### 5.1 当前风险

当前 `ensure_embeddings` 仅通过向量是否为空和维度是否一致判断是否需要重嵌。

两个 embedding 模型可能具有相同维度，但向量空间不兼容。此时：

- 查询向量由新模型生成；
-数据库存量向量由旧模型生成；
- pgvector 查询可以正常执行；
- 检索结果会静默失真；
- 系统不会报错。

### 5.2 修订要求

Phase 1 必须至少记录一个稳定的 embedding identity：

```text
embedding_model_key
```

推荐格式：

```text
glm/embedding-3:1024
```

也可以一次性记录：

```text
embedding_provider
embedding_model
embedding_dimension
embedding_updated_at
```

`ensure_embeddings` 的 stale 判断必须同时包含模型身份：

```python
stale = (
    any(s.embedding is None for s in segments)
    or stored_dimension != current_dimension
    or stored_model_key != current_model_key
)
```

模型身份变化时，对该会议全部 segment 重嵌，不能混用不同模型生成的向量。

### 5.3 数据存放建议

如果同一会议的全部 segment 始终使用同一 embedding 配置，优先将模型标识记录在 `Meeting`，避免每个 segment 重复存储。

若未来允许 segment 级异步重嵌，再考虑下沉到 segment。

---

## 6. Phase 1 实现边界调整

### 6.1 建议保留

Phase 1 包含：

- 合并后的意图识别 + standalone query；
- 原问题和改写问题的说话人匹配并集；
- vector anchors；
- speaker anchors；
- 简单合并去重；
- anchor 优先的邻居扩展；
- 最小证据不足拒答；
- embedding 模型身份校验；
- 基本检索 metadata；
- 单元测试和集成测试。

### 6.2 建议暂不包含

- 关键词召回；
- pg_trgm 迁移；
- RRF；
- Reranker；
- TranscriptChunk；
- Graph RAG；
- 跨会议检索；
- 完整离线评测平台；
- 复杂 confidence 评分；
- 大规模目录重构。

### 6.3 文件组织

Phase 1 不建议立即拆成多个 retrieval 子模块。优先采用：

```text
app/services/qa.py
app/services/query_intent.py
```

或者：

```text
app/services/retrieval.py
```

先把函数职责拆清楚。等 Phase 2 引入 vector/speaker/keyword 三路候选和 RRF 后，再决定是否升级为 `app/services/retrieval/` package。

这样可以避免同一个 PR 同时包含行为变化和大规模文件迁移，降低 review 难度。

---

## 7. 检索 metadata 与隐私

Phase 1 可以先使用结构化日志，但生产环境默认不得记录完整问题和改写文本。

默认建议：

```json
{
  "question_hash": "sha256:...",
  "standalone_query_changed": true,
  "matched_person_count": 2,
  "retrieval_strategy_counts": {
    "vector": 6,
    "speaker": 4
  },
  "anchor_count": 8,
  "kept_anchor_count": 6,
  "context_segment_count": 20,
  "truncated": true,
  "latency_ms": {
    "intent_rewrite": 120,
    "embedding": 80,
    "retrieval": 45,
    "generation": 900
  }
}
```

只有显式开启以下配置时才记录问题明文：

```env
QA_RETRIEVAL_DEBUG_TEXT=false
```

日志不得包含 API Key、完整逐字稿或大段原文。

---

## 8. Phase 2 补充约束

原设计的 PostgreSQL + `pg_trgm` 方向保留，但实现时补充以下约束。

### 8.1 关键词提取

- 不提取裸数字；
- 普通英文 token 设置最短长度；
- 版本号、编号、金额、日期可绕过普通最短长度规则；
- 限制每个查询最多关键词数量：

```env
QA_KEYWORD_MAX_TOKENS=8
```

### 8.2 SQL 安全与性能

- `ILIKE` 参数必须使用 SQLAlchemy 绑定参数；
- 对 `%`、`_` 和转义符做 LIKE escape；
- 关键词召回上线前必须创建 `pg_trgm` GIN 索引；
- 每个关键词独立限制候选数，避免高频词占满全局 LIMIT；
- 查询始终带 `meeting_id` 过滤。

### 8.3 RRF

`RRF_K=60` 仅作为初始配置，不声明为已验证最优值。

单会议 segment 规模较小时，`k=60` 可能使不同 rank 的分数差异过小。后续真实测试至少比较：

```text
10 / 30 / 60
```

---

## 9. 修订后的 Phase 1 流程

```text
当前问题 + 历史用户问题 + 上轮合法引用原文
        ↓
一次 LLM：intent + standalone_query
        ↓
original question 与 standalone query 匹配说话人并集
        ↓
standalone_query embedding
        ↓
vector anchors + per-person speaker anchors
        ↓
简单合并去重，保留 anchor 排名
        ↓
anchor 优先、邻居逐圈填充 context budget
        ↓
最终 context 按 seq 排序
        ↓
LLM grounded answer
        ↓
程序校验引用必须来自本轮 context
        ↓
证据不足则明确拒答
```

---

## 10. 修订后的 Phase 1 测试要求

至少覆盖：

1. 无历史消息时不增加独立查询改写调用；
2. 多轮时一次调用同时返回 intent 与 standalone query；
3. 改写失败时使用原问题继续检索；
4. 历史 assistant 错误答案不会被当作事实固化到改写问题；
5. 原问题中的人名即使被改写遗漏，仍参与 speaker 召回；
6. 改写凭空新增的人名不会直接成为强 speaker 条件；
7. 同时命中两名说话人时，两人的发言均有候选；
8. 同一 person 的多个 speaker label 不产生重复 person；
9. anchor 扩展包含前后邻居；
10. context 预算不足时，高排名 anchor 不会因按 seq 排序而被截掉；
11. anchor 优先于邻居；
12. 多个 anchor 的重叠邻居正确去重；
13. 最终 context 按 seq 排序；
14. citation 只能引用本轮 context；
15. 无明确证据时允许拒答，且不伪造引用；
16. embedding 模型相同且维度相同不重复重嵌；
17. embedding 模型变化但维度相同时仍触发全量重嵌；
18. 生产默认日志不含原始问题明文；
19. 原有按姓名问答测试继续通过；
20. 原有普通向量问答测试继续通过。

---

## 11. 修订后的 Phase 1 完成定义

Phase 1 只有在以下条件全部满足时才算完成：

- 多轮意图识别与 standalone query 在一次 LLM 调用中完成；
- 无历史消息时不产生额外改写调用；
- 改写后的问题用于 embedding；
- 原问题和改写问题共同用于说话人姓名匹配；
- 多说话人继续按 person 独立召回；
- anchor 和 context expansion 已分离；
- context budget 使用 anchor 优先算法；
- 邻居扩展支持去重、排序和总量限制；
- 回答引用只能来自本轮 context；
- 无明确证据时可以拒答，不要求模型强行给结论；
- embedding 模型身份与维度均参与 stale 判断；
- 关键检索 metadata 可追踪，且默认日志不记录敏感问题明文；
- 新增测试覆盖本文测试要求；
- 全量后端测试通过；
- 前端类型检查和构建通过；
- 未引入 Phase 2、Phase 3 的额外复杂度。

---

## 12. 待 Claude 讨论的问题

请 Claude 针对以下问题逐项回应：

1. 当前意图分类实现是否已经具备合并 standalone query 的基础，是否存在必须拆成两次调用的技术原因？
2. 当前 `_retrieve` 是否保存 vector distance/rank，足以支持 anchor 优先预算？如不足，最小改动是什么？
3. 当前回答 Schema 和前端是否允许空引用拒答？如果不允许，兼容性最小改法是什么？
4. embedding 模型标识更适合存放在 Meeting 还是 TranscriptSegment？请结合当前模型和迁移结构说明。
5. Phase 1 是否确实需要配置改名 `qa_speaker_top_k → qa_speaker_top_k_per_person`，是否应保留旧环境变量兼容一个版本？
6. 结构化日志当前使用什么 logger/字段格式，如何保证默认不记录问题明文？
7. 是否同意 Phase 1 暂不做大规模 `app/services/retrieval/` 目录迁移？

Claude 回应后，再将确认的修订合并进主设计 `TECH_DESIGN_MEETING_RAG_V1.md`。