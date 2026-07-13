# AI Meeting Assistant 产品需求文档（PRD v2）

> 版本：v2（2026-07）
> 本版本在 v1 基础上，吸收架构评审结论修订而成。核心变化：
> 1. 处理管道补充「转码归一化」「对齐合并」「结构化 JSON」三个隐藏步骤
> 2. 数据模型全面预留多用户（user_id）、声纹（embedding 留存 + 版本号）、可回滚绑定
> 3. MVP 新增「Speaker 重命名」轻量功能
> 4. 明确技术选型（Python 全栈、云 ASR API 优先、pgvector）
> 5. 新增非功能需求：状态机、重试、鉴权、上传、合规、成本
> 6. 标注 MVP 边界，防止提前实现二期功能

---

## 1. 产品定位

### 产品名称（暂定）
AI Meeting Assistant

### 产品目标
打造一个智能会议助手，将会议录音自动转化为结构化会议纪要，并支持基于会议内容的 AI 对话、内容追溯、跨会议知识积累。

核心价值：
1. 降低人工整理会议纪要成本
2. 提高会议内容可搜索性和可追溯性
3. 建立个人/团队长期会议知识库
4. 通过 AI 理解会议上下文，辅助后续工作

最终目标：从「会议记录工具」升级为「个人/团队长期 AI 工作记忆系统」。

### 设计原则（全项目最高约束）
1. 所有 AI 输出必须保留来源引用（可溯源到 segment 与时间戳）
2. 会议文本必须保存时间戳
3. Speaker（会议内临时标签）与 Person（全局身份）分离设计
4. LLM 负责理解，程序负责结构化（LLM 输出 JSON，程序渲染 Word）
5. 原始数据不可丢失（音频为不可变资产，删除仅由用户显式触发）

---

## 2. 核心用户流程（v2 修订版管道）

```
上传录音（预签名 URL 直传对象存储，支持分片）
    ↓
转码归一化（统一转为 16kHz 单声道 wav 中间格式）
    ↓
转写 + 说话人分离 + 对齐（一个管道阶段，产出统一 segment）
    ↓
segments 入库 ──────→ 【随时可用】查看/下载完整转录、点击定位音频
    ↓
（可选）用户重命名 Speaker（speaker_001 → "Tim"，写 SpeakerBinding）
    ↓
LLM 总结 → 结构化 JSON（schema 校验 + 失败重试）→ Summary 入库
    ↓
Word 导出（用户点击时由程序用模板实时渲染，不占管道）
```

管道说明：
- 每个阶段为独立异步任务，阶段间通过数据库/对象存储传递中间产物
- 任一阶段失败可从该阶段单独重试，不从头重跑
- Meeting 状态机：`uploaded → transcoding → transcribing → summarizing → done / failed`，前端可实时查看进度
- 「查看原文」不依赖后续阶段：segments 入库即可用，LLM 失败时用户仍有完整转录可用

支持上传格式：mp3 / wav / m4a / mp4（转码阶段归一化，未来小程序 aac 等格式仅需扩展转码入口）

---

## 3. 核心功能需求

### Feature 1：语音转文字（含对齐合并）

将会议录音转换为带时间戳、带说话人标签的结构化文本。

**实现要点（v2 明确）：**
- ASR 与 diarization 输出的是两条独立时间轴，必须有显式的对齐合并步骤
- MVP 推荐直接使用云 ASR API（自带说话人分离）或 WhisperX（转写+对齐+diarization 一体）
- ASR 引擎必须抽象为可替换的 provider 接口（云 API ↔ 自建模型可切换）
- 支持热词/术语表注入机制（应对专业术语识别）

**segment 数据结构：**
```json
{
  "seq": 42,
  "speaker_label": "speaker_001",
  "person_id": null,
  "start_time": 120.5,
  "end_time": 130.2,
  "text": "我们计划下周完成接口开发"
}
```

**能力要求：**
- 查看完整转录
- 下载原始文字
- 点击 segment 跳转音频播放位置（音频访问用短时签名 URL 按需换发，存储需支持 range 请求）

### Feature 2：说话人识别

**第一阶段（MVP）：会议内说话人分离 + 手动重命名**
- 区分 Speaker A / B / C，标记每句话归属
- 【MVP 新增】用户可将 speaker_001 手动重命名为真实姓名（如 "Tim"）
  - 仅写 SpeakerBinding 表（confirmed_by = human），不做声纹
  - 纪要与 TODO 归属立即使用真名
- 【MVP 必做的暗桩】diarization 阶段顺手产出的 speaker embedding 必须入库留存（VoiceSample 表，person_id 暂空，带模型版本号）——为二期声纹避免全量重跑历史音频

**第二阶段：跨会议声纹记忆**
- 用户确认 Speaker A = Tim 后保存声音特征；后续会议自动匹配
- 匹配结果分三档：高置信自动绑定 / 中置信待人工确认 / 低于阈值创建新 Person 或标记未知
- 支持一人多声音样本、人工确认、未知用户、重新绑定（绑定为追加式事件，可回滚、可审计）
- 合规（见 §9.4）：声纹属敏感个人信息，需单独同意与物理级联删除

### Feature 3：AI 生成会议纪要

**输入：** 完整转录 segments（长会议采用分段 map-reduce 策略，注意跨段上下文的 TODO/决策不丢失）

**输出：** 结构化 JSON（经 schema 校验，失败自动重试，降级策略为至少产出纯文本纪要），包含：

```
会议主题 / 会议时间 / 参会人员
## 会议总结
## 讨论事项（1. 2. 3.）
## 决策事项
## TODO（事项 | 负责人 | 截止时间，每条绑定 source_segment_id）
```

**溯源要求：** LLM 引用 segment 编号而非自由复述原文，程序反查真实文本与时间戳，杜绝幻觉引用。

### Feature 4：会议纪要风格统一

根据历史会议纪要学习固定写作风格（标题格式、语言习惯、TODO 格式、术语，如统一「推进 XX 事项」）。

- 第一阶段：Few-shot Prompt（历史纪要样本注入）
- 第二阶段：RAG（历史纪要 → embedding → pgvector 检索 → 生成参考）
- Summary 记录 style_profile_id，风格配置可追溯

### Feature 5：AI 会议问答（二期）

通过聊天理解会议内容，支持三类问题：
- 内容查询：「支付接口什么时候完成？」
- 原文追溯：「TODO 第二条是谁提出的？」→ 返回负责人 + 原文 + 时间戳（如 31:20）
- 决策查询：「为什么最后选择方案 B？」

实现：segments embedding 入 pgvector → 检索相关 segment → LLM 回答，回答强制携带 cited_segment_ids，前端据此渲染原文对照与音频跳转。

> 现状：内容查询与原文引用已实现（pgvector 检索 + LLM 引用 segment，前端引用跳播）；「修改纪要」见 §7.1。

### Feature 6：Word 自动生成

- 输入：Summary 的结构化 JSON（content_json）
- 输出：Word 文件（公司固定模板、表格填充、标题格式）
- 实现：docxtpl 填模板，纯程序步骤，与 LLM 完全解耦；用户点击导出时实时渲染
- Summary 版本化：二期通过聊天修改纪要时生成新版本而非覆盖（复用 summarize 的 max(version)+1；改纪要须同步重建 ActionItem 以保证 Word 导出一致）

### Feature 7：前端应用

**第一阶段：Web**
- 首页：上传会议、查看历史会议、处理进度状态
- 会议详情页：AI 纪要、原始转录、音频播放器（segment 点击跳转）、Speaker 重命名、Word 导出
- 前端工程要求：业务逻辑（API client、状态管理）与视图层分离，为二期跨端复用铺路（可评估 Taro 等跨端框架）

**第二阶段：微信小程序**
- 新增 AI 聊天页：修改纪要、查询细节、追溯原文
- 平台前置约束已落实在一期后端设计中（见 §9.2 / §9.3）

---

## 4. 系统架构

```
客户端（Web / 二期小程序）
    ↓ HTTPS + Bearer Token
API 服务（FastAPI：上传凭证、任务投递、状态查询、结果查询、聊天）
    ↓
任务队列（MVP：BackgroundTasks/线程池；量大后换 Celery + Redis，阶段划分不变）
    ↓
异步处理管道 Worker：转码归一化 → 转写+分离+对齐 → LLM 总结（结构化 JSON）
    ↓
存储层：对象存储（音频，不可变） + PostgreSQL + pgvector（结构化数据 + 向量）
外部依赖：云 ASR API（provider 可替换）、LLM API（统一 service 抽象）
```

分层职责：
- **API 服务**：轻量，绝不在请求线程跑模型；上传采用「申请凭证 → 分片直传对象存储 → 通知确认」
- **管道**：阶段独立、状态写回 Meeting、失败按阶段重试；「任务状态变更」发布为内部事件，通知渠道（轮询 / SSE / 二期微信订阅消息）作为订阅者实现
- **LLM service**：屏蔽多厂商差异，统一 prompt 管理、JSON schema 校验、重试、成本记录
- **Word 导出**：纯程序模块，读 Summary JSON 填模板

---

## 5. 数据模型

> 所有业务表自带 user_id（MVP 可写死默认用户）、created_at / updated_at。

**User / AuthIdentity（身份与登录方式分离）**
```
User:          id, name, created_at
AuthIdentity:  id, user_id, type(email|wechat_unionid|phone), identifier, credential
```
微信侧按 UnionID 设计，避免多端多账号。

**Meeting**
```
id, user_id, title, status(uploaded|transcoding|transcribing|summarizing|done|failed),
duration, audio_url, error_message, created_at
```

**TranscriptSegment（逐行存储，不存大 JSON）**
```
id, meeting_id, seq, start_time, end_time,
speaker_label,          -- 会议内标签 "speaker_001"
person_id (nullable),   -- 全局身份，绑定后物化
text, embedding(vector) -- 二期问答用，一期可空
```

**SpeakerBinding（Speaker→Person 映射，追加式、可审计）**
```
id, meeting_id, speaker_label, person_id,
confirmed_by(auto|human), confidence, superseded_by(nullable), created_at
```

**Person / VoiceSample**
```
Person:      id, user_id, name, consent_record, created_at
VoiceSample: id, person_id(nullable), source_meeting_id, embedding(vector),
             model_name, model_version, sample_start, sample_end, created_at
```
删除 Person 必须物理级联删除全部 VoiceSample 与 embedding。

**Summary（版本化）**
```
id, meeting_id, version, style_profile_id, content_json, created_at
```

**ActionItem**
```
id, meeting_id, task, owner_person_id(nullable), owner_text,
deadline, source_segment_id
```

**ChatMessage（二期）**
```
id, meeting_id, role, content, cited_segment_ids, created_at
```

---

## 6. MVP 范围（第一版，2 周）

### 包含
- 上传录音（预签名 URL 直传 + 分片）
- 转码归一化
- 自动转文字 + Speaker 分离 + 对齐（云 ASR API 优先）
- diarization embedding 留存入库（VoiceSample，暗桩）
- Speaker 手动重命名
- AI 生成会议纪要（结构化 JSON + schema 校验重试）
- Word 导出（docxtpl）
- 下载原始转录、音频定位播放
- Meeting 状态机 + 阶段级重试
- Bearer Token 鉴权（单默认用户）

### 明确不包含（不要提前实现）
- 跨会议声纹自动识别与匹配
- 微信小程序
- AI 聊天问答 / RAG 知识库
- 多用户注册登录、团队与权限
- 自建 GPU 部署 ASR 模型

### 两周计划
**第 1 周（打通管道）：**
1. 项目骨架 + 数据库 schema（含全部暗桩字段）+ 上传接口 + 对象存储
2. ASR provider 接入：转写 + diarization + 对齐，segments 入库，embedding 留存
3. 会议详情接口：状态查询、完整转录、原文下载
- 建议：头两天先跑通一条最简端到端流水线（摘要哪怕一句话 prompt），再逐段加深

**第 2 周（产出价值）：**
4. LLM 摘要：map-reduce → 结构化 JSON（校验 + 重试）→ Summary 入库
5. docxtpl 模板 + Word 导出接口
6. Web 前端：上传页、列表页、详情页（纪要 + 转录 + 播放器 + Speaker 重命名）
7. 端到端联调 + 长音频（2 小时）压测

---

## 7. 第二阶段规划

> 标注现状：✅ 已实现 / 🟡 部分 / ⬜ 待建。步骤以当前代码为基线，尽量复用既有件。

### 7.1 AI 聊天
- **查询会议内容（✅ 已实现）**：RAG 问答，pgvector 检索相关 segment → LLM 回答。
  见 `app/services/qa.py`、`app/api/routes/chat.py`。
- **原文引用（✅ 已实现）**：LLM 引用 segment 编号（seq），程序反查真实 segment
  落 `chat_messages.cited_segment_ids`，前端引用 chip 点击跳播。
- **修改纪要 → Summary 新版本（⬜ 待建，本期重点）**：在同一聊天框内用自然语言
  指令改纪要（如「把决策第二条改成…」「合并前两个议题」），生成新版本而非覆盖。
  实现步骤：
  1. 意图路由：聊天 POST 先用一次轻量 LLM 分类 `query | edit`（复用 `build_router`，
     结构化输出 `{intent}`）。`query` 走现有 `answer_question`；`edit` 走改纪要分支。
     （备选：前端加「编辑纪要」显式开关，避免误分类——若分类不稳再切换。）
  2. 改纪要分支：取该会议最新 Summary 的 `content_json` + 用户指令，
     （可选）附相关 segment 做依据 → LLM 产出完整新 `content_json`，
     经 `SummaryContent` schema 校验（复用 `app/schemas/summary.py`）。
  3. 落库新版本：复用 `summarize.py` 的 `max(version)+1` 模式写入 Summary；
     `_meta` 记来源（如 `{"origin":"chat","edit_instruction":…}`）。
  4. 重建 ActionItem：像 `summarize_meeting` 一样 delete+重插本会议 ActionItem
     （seq→segment 反查 owner/source），否则 Word 导出 TODO 表与新纪要不一致。
  5. 聊天回执：assistant 消息说明改了什么并标注新版本号；
     （可选）`chat_messages` 加 `summary_version` 列记录该轮产出的版本（需迁移，审计用）。
  6. 前端：`QaPanel` 增 `onSummaryUpdated` 回调 → `MeetingDetailPage.refresh` 重拉
     `getSummary`，使纪要 tab 即时反映新版本（现有 done 后不再轮询，须显式刷新）。
- **版本历史 / 回滚（⬜ 增强，可选）**：`GET /summary` 目前只返回最新版。
  如需历史与回滚，加 `GET /meetings/{id}/summary/versions` 与 `?version=N`，
  回滚 = 复制旧版内容为新版本（仍不覆盖）。

### 7.2 声纹记忆（🟡 部分）
- 已实现：SeedASR 声纹匹配命中自动绑定（`auto_bind_voiceprints`）、Person 声纹登记、
  删除 Person 云端清理提示。
- 待建：三档置信度流（高置信自动绑 / 中置信「待人工确认」队列 + 确认 UI /
  低置信新建 Person 或标记未知）；绑定回滚 UI（SpeakerBinding 已是追加式可回滚，
  补前端）；合规同意流（`Person.consent_record` 已有字段，补录入/展示）。

### 7.3 RAG 知识库（⬜ 待建）
- 跨会议搜索：现有 segment embedding 检索限定单会议；放开 meeting_id 约束、
  按 user 检索全部会议，答案标注来源会议+时间戳。
- 历史决策查询、纪要风格 RAG：纪要范例库（summary_examples）已按 few-shot 注入；
  二期可将历史纪要 embedding 化做检索式风格参照。

### 7.4 微信小程序（⬜ 待建）
- 微信登录：AuthIdentity 已预留 `type=wechat_unionid`，补 UnionID 换 JWT 流。
- 订阅消息通知：管道状态变更事件已是内部信号，补微信订阅消息订阅者。
- 录音格式：转码入口（`transcode.py`）扩 aac 等小程序格式。

### 7.5 任务队列升级（⬜ 待建）
- BackgroundTasks → Celery + Redis：管道阶段划分不变（`run_pipeline` 各阶段幂等），
  换执行器即可；换后移除启动时「in-flight 标 failed」的兜底（`app/main.py` lifespan）。

## 8. 第三阶段规划（企业级）

多用户、团队成员管理、权限控制、企业知识库、飞书/Slack/Teams 集成。

---

## 9. 非功能需求

### 9.1 性能与可靠性
- 处理为全异步，前端展示阶段进度；1 小时音频端到端处理目标 ≤ 15 分钟（云 API 路线）
- 阶段级重试，失败任务告警；LLM 输出 schema 校验失败自动重试，最终降级为纯文本纪要
- 实际用量预算按理论值 × 1.2-1.5（重试与调试损耗）

### 9.2 鉴权与上传（为小程序前置）
- 全站 Bearer Token（JWT），不用 cookie session
- 上传：预签名 URL 分片直传对象存储，服务端确认合并
- 音频播放：按需换发短时签名 URL，存储支持 range 请求

### 9.3 部署与域名
- HTTPS + 已备案域名（含对象存储下载域名，需可加入小程序合法域名白名单）
- 密钥与配置走环境变量，不入仓库

### 9.4 安全与合规
- 录音为高敏感数据：存储加密、删除即物理删除、明确数据保留策略（一期即落实）
- 音频不可变：无自动清理策略，删除仅由用户显式触发；长期用低频/归档存储控制成本
- 声纹（二期）：属《个人信息保护法》敏感个人信息，需单独同意（Person.consent_record）、支持物理级联删除；注意「参会人」与「系统用户」非同一批人的授权流程设计

### 9.5 成本模型（决策参考）
- 变动成本约 ¥2-6 / 音频小时（云 ASR ¥1-3 + LLM ¥0.1-3 + 存储忽略不计）
- MVP 现金成本 < ¥1200/月（100 音频小时/月量级）；千小时级约 ¥3000-8000/月
- 自建 GPU 盈亏平衡点约数百至一千音频小时/月，届时通过 ASR provider 接口切换
- 全周期成本排序：人力维护 > ASR > LLM > 基础设施 > 存储；优先用托管服务换运维人力

---

## 10. 技术选型

| 层 | 选型 | 说明 |
|---|---|---|
| 后端 | Python + FastAPI | 全栈统一 Python，ASR 生态所在 |
| 任务 | MVP: BackgroundTasks → 二期 Celery+Redis | 阶段划分先行，执行器可替换 |
| ASR | 云 ASR API（带说话人分离）优先；WhisperX 备选 | provider 接口可替换 |
| 声纹 | pyannote / SpeechBrain（二期） | embedding 一期即留存 |
| LLM | Claude / GPT / DeepSeek 等，经统一 service 抽象 | 成本记录、schema 校验 |
| 数据库 | PostgreSQL + pgvector | 一库同时管结构化与向量，Milvus 规模化后再评估 |
| 存储 | 对象存储（OSS/S3 类） | 预签名直传、range、生命周期归档 |
| 文档 | docxtpl / python-docx | JSON → Word |
| 前端 | React（逻辑与视图分离）或 Taro | 二期小程序复用 |

---

## 11. 风险与缓解

| 风险 | 影响 | 缓解 |
|---|---|---|
| ASR 质量是全链路天花板（术语、中英夹杂、抢话） | 下游全部继承误差 | 尽早用真实会议录音验证；热词注入；provider 可切换 |
| Diarization 错误传播 | 污染 TODO 归属与二期声纹库 | 人工确认不省略；自动绑定阈值保守；绑定可回滚 |
| LLM 幻觉与引用失真 | 信任崩塌 | 引用 segment id 而非复述；程序反查；前端原文对照 |
| 处理时延与 GPU 成本 | 体验差 / 成本失控 | MVP 走云 API；异步 + 进度展示 |
| 隐私合规 | 法律风险 | §9.4 一期落实，声纹合规二期前设计完成 |
| 2 周排期 | 集成爆炸 | 第 1 周头两天先通最脏的端到端流水线 |
