# 会议问答离线评估：数据集整理与评估操作指南

> 对应 RAG 设计 `docs/TECH_DESIGN_MEETING_RAG_V1.md` §6.2。
> 目标：从真实会议整理 50~100 题，跑出检索质量报告，为 RRF k 调参与
> Reranker 引入决策（§6.3）提供数据。

## 前置条件

- 在**部署环境**（能连业务数据库的机器）的项目目录下操作，脚本读 `.env` 的 `DATABASE_URL`；
- 相关会议已处理完成（status=done，有转录）。

## 第 1 步：挑会议

```bash
uv run python scripts/dump_transcript.py --list
```

输出近期 30 场会议：`<meeting_id>  <时间>  segments=<段数>  <标题>`。
建议挑 3~5 场内容有实质讨论的会议（段数太少的会议出不了几道题）。

## 第 2 步：导出带 seq 的转录

```bash
uv run python scripts/dump_transcript.py <meeting_id> > transcript.txt
```

输出格式（行首 `[N]` 就是 seq，出题时的证据引用全靠它）：

```
# meeting_id: c9c46e63-...
# title: 项目周会
[0] [00:00:00] 张三: 大家好，我们开始今天的项目周会。
[1] [00:00:12] 李四: 先同步一下上周的进展...
```

> 提示：出题前先在页面上把主要说话人绑定真名，导出会显示真名，
> 出"指定说话人"类问题更自然。

## 第 3 步：与大模型对话出题

把 `transcript.txt` 全文 + `eval/qa_dataset.example.json` 的内容一起粘给大模型，
提示词参考：

> 以下是一场会议的转录（行首方括号内是 segment 编号 seq）和一个 JSON 格式样例。
> 请照样例格式为这场会议出 15~20 道问答评估题，要求：
> 1. `meeting_id` 用转录文件头部标注的那个；
> 2. `expected_segment_seqs` 必须引用转录行首真实出现的 seq（能回答该问题的
>    一或多段），严禁编造；
> 3. `expected_answer` 填原文中的简短答案；
> 4. `category` 覆盖多种类型：speaker（指定说话人）/ date（日期）/
>    amount（金额）/ action_item（行动项）/ decision（决策）/
>    comparison（多人对比）/ cross_segment（跨段信息）/ negative（否定信息）；
> 5. 另出 2~3 道会议里**没有**答案的题，`category` 填 `no_answer`，
>    不填 `expected_segment_seqs` 和 `expected_answer`；
> 6. 输出一个合法 JSON 数组，不要输出其他内容。

## 第 4 步：人工审核

逐条过一遍生成的题：

- 删掉问题含糊、答案有争议的；
- 抽查几条 `expected_segment_seqs`，对照转录确认 seq 指向的段落确实能回答问题；
- 多场会议的题**合并成一个 JSON 数组**，存为 `eval/qa_dataset.json`
  （该文件不入库控，`eval/` 下只有 example 提交）。

> 不用担心 seq 手滑：评估时不存在的 seq 会被显式报告并剔除
> （报告 `skipped` 字段），不会静默污染指标。

## 第 5 步：跑评估

```bash
# 检索档（默认，零 LLM 消耗）：Recall/MRR/延迟 + RRF k 三组对比
uv run python scripts/qa_eval.py --dataset eval/qa_dataset.json --rrf-k 10,30,60

# 回答档（真实调用当前 LLM provider，注意额度）：另出引用准确率/拒答率/置信度分布
uv run python scripts/qa_eval.py --dataset eval/qa_dataset.json --with-answers
```

## 第 6 步：读报告

关键字段：

| 字段 | 含义 | 怎么用 |
|---|---|---|
| `retrieval.recall@5 / recall@10` | 期望证据进入 top-K anchors 的比例 | 三组 rrf_k 对比，选最高的 k |
| `retrieval.mrr` | 第一条命中证据的平均倒数排名 | 越接近 1 越好 |
| `retrieval.context_hit_rate` | 期望证据最终进入 context 的比例 | 低说明预算/窗口需要调 |
| `retrieval.avg_candidates` / `candidates_gt_20_rate` | 候选规模 | §6.3：候选经常 >20 才考虑 Reranker |
| `answers.citation_precision` | 模型引用命中期望证据的比例 | 仅 --with-answers |
| `answers.no_answer_refusal_rate` | 无答案题的正确拒答率 | 仅 --with-answers |
| `by_category` | 按题型细分的 recall@5 | 找短板题型（如 amount 低 → 关键词召回问题）|
| `skipped` | 被跳过/剔除的题与原因 | 修数据集 |

## 决策参考

- **RRF k**：取三组对比中 recall@5 / mrr 最优者，写进 `.env` 的 `QA_RRF_K`；
- **Reranker**（§6.3）：仅当 `candidates_gt_20_rate` 较高**且** top-K 噪声明显
  （recall@5 明显低于 recall@10）时才值得评估引入；
- 某类型 recall 显著偏低时，优先怀疑对应召回路（amount/date/编号 → 关键词；
  speaker → 绑定与保底；语义类 → 向量/embedding 模型）。
