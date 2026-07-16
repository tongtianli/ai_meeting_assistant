# 声纹记忆增强技术设计 v1（讨论稿）

> 状态：Discussion Draft——本 PR 仅供方案讨论，不含业务代码，PRD 修改待定稿后随实现 PR。
>
> 目标交互（产品诉求）：
> 1. 用户在之前的会议中对某说话人**重命名**后，后续会议出现该说话人时**系统自动标注**；
> 2. 会议中出现**未知说话人**时，系统给出一段或几段录音**询问用户这个人是谁**（可跳过）。

## 1. 合规立场变更

PRD §9.4 现行要求「声纹属敏感个人信息，需**单独同意**（Person.consent_record）」，
产品决定**删除单独同意的强制流程**：

- 本产品当前为**单用户个人工具**，录音与声纹均为使用者自有会议数据，
  强制 consent 流程（`persons` 路由注册 voiceprint 时必填 consent_note）
  只增加交互摩擦，无实际保护对象；
- **保留**的合规底线（与同意流程无关、属数据卫生）：
  - 删除 Person 物理级联删除全部 VoiceSample 与 embedding（现有行为不变）；
  - 云端声纹（SeedASR）删除 Person 时同步清理云端样本的提示/动作保留；
- `Person.consent_record` 字段**保留不删**（历史数据 + 未来多租户重启用），
  仅去掉路由层的强制校验；
- 注：若产品未来走向多租户/对外服务（§8 企业级），声纹按 PIPL 仍属敏感个人
  信息，届时需重新评估参会人（非系统用户）的授权设计——本次变更以
  单用户自用为前提。

**PRD 修改点（随实现 PR 执行）**：§9.4 声纹条目改写为上述立场；Feature 2 /
§7.2 的「合规同意流（补录入/展示）」从待建清单移除。

## 2. 现状盘点（代码事实）

已有的暗桩与半成品：

| 组件 | 现状 |
|---|---|
| `VoiceSample` 表 | ✅ 每场会议留存 diarization speaker embedding（model_name/version、sample_start/end）；**缺 `speaker_label` 列**（见 §5 缺口） |
| `SpeakerEmbeddingSample`（ASR 抽象） | ✅ 带 `speaker_label`，但管道入库时丢弃了该字段 |
| 本地 FunASR | ✅ 产出每说话人 spk_embedding（cam++ 系）|
| SeedASR 云端声纹 | ✅ 转写时带 `voice_print_list` 匹配 + `auto_bind_voiceprints` 命中自动绑定（阈值可配）；`Person.voiceprint_id` 已有；**注册是人工步骤**（控制台），注册 API（RegisterVoicePrint，16kHz WAV mono ≥10s、base64 ≤2MB）已调研未接 |
| `SpeakerBinding` | ✅ 追加式、可回滚、`confirmed_by(auto/human)` + `confidence`——三档置信度的存储底座现成 |
| `bind_speaker`（重命名） | ✅ 建/找 Person + 绑定 + 物化 segment.person_id；**未触碰 VoiceSample 与云端注册** |
| 音频跳播 | ✅ 短时签名 URL + segment start_time——确认卡的"听一段"直接复用 |

## 3. 总体设计：声纹记忆双后端

```
                     重命名（bind_speaker）＝隐式声纹登记
                                   ↓
              ┌────────────────────┴───────────────────┐
        本地 embedding 后端                        云端声纹后端（SeedASR）
   物化 VoiceSample.person_id                 切干净样本 → RegisterVoicePrint
   （provider 无关，任何 ASR 可用）            → 回填 Person.voiceprint_id
              └────────────────────┬───────────────────┘
                                   ↓
                     新会议 transcribe 完成后「识人」
              ┌────────────────────┴───────────────────┐
   本地：新 speaker embedding vs 已归属           云端：转写响应自带
   VoiceSample（pgvector cosine，                voiceprint 命中
   仅同 model_name/version 可比）                （现 auto_bind 逻辑）
              └────────────────────┬───────────────────┘
                                   ↓
                          三档置信度统一收口
        高 ≥ T_auto   → 自动绑定（confirmed_by=auto，详情页可撤销）
        中 [T_ask,T_auto) → 进「待确认」，确认卡带音频片段询问
        低 < T_ask    → 未知说话人，确认卡询问「这个人是谁？」（可跳过）
```

两后端并存的理由：用户当前主用 SeedASR（云端声纹效果好），但额度耗尽后回
本地 FunASR——本地 embedding 后端 provider 无关，是长期解；云端后端是现用
路径的补全（把人工注册变成重命名即自动注册）。

## 4. 能力 A：重命名即登记 + 跨会议自动标注

### 4.1 重命名钩子（隐式登记）

`bind_speaker(meeting, speaker_label, name)` 在现有动作之外：

1. **本地**：把该会议、该 speaker_label 的 VoiceSample.person_id 物化为绑定的
   Person（需 §5 的 speaker_label 列）——此人声纹样本自此可被后续会议比对；
2. **云端**（配置了 SeedASR 且 Person 尚无 voiceprint_id 时）：从该 speaker 的
   segments 中选取干净样本（拼接至 ≥10s，16kHz WAV mono，避开重叠发言），
   调 RegisterVoicePrint 注册，回填 `Person.voiceprint_id`；失败仅告警不阻断
   重命名（登记是增强，不是重命名的前置条件）。

改绑/撤销时的处置（见 §9 讨论问题 3）：样本归属跟随最新绑定，旧归属样本
标记失效或物理清理。

### 4.2 新会议自动标注

管道 `_stage_transcribe` 落库后（现 `auto_bind_voiceprints` 的位置扩展）：

- 云端命中：保持现有逻辑（命中即绑，阈值 `voiceprint_auto_bind_min_confidence`）；
- 本地比对（新增 `match_local_voiceprints`）：本会议每个未绑 speaker 的
  embedding，对**同 model_name/model_version** 的已归属 VoiceSample 做
  pgvector cosine top-k，得分聚合（如按 Person 取最高分）后走三档：
  - `VOICEPRINT_AUTO_BIND_THRESHOLD`（建议初始 0.8，实测调）→ 自动绑定，
    `SpeakerBinding(confirmed_by=auto, confidence=score)`；
  - `VOICEPRINT_ASK_THRESHOLD`（建议 0.6）→ 候选带进确认卡（"是张三吗？"）；
  - 以下 → 未知，确认卡开放式询问。
- 自动绑定在前端有可见标识（如「自动识别」徽标），一键撤销走现有
  SpeakerBinding 追加式回滚。

## 5. 数据与迁移

1. **`voice_samples` 加 `speaker_label` 列**（关键缺口）：新数据由管道直写
   （`SpeakerEmbeddingSample.speaker_label` 本来就有）；存量数据用
   `sample_start/end` 与 segments 时间重叠做 best-effort 回填，重叠歧义的留空。
2. **确认卡状态**：跳过（dismissed）需要落地，否则每次进详情页都再问。
   方案 a：`speaker_bindings` 允许 person_id 为空 + `confirmed_by=dismissed`
   （复用追加式审计）；方案 b：Meeting 上 JSONB `dismissed_speaker_labels`。
   倾向 a（可回滚、可审计、无新表）——见讨论问题 4。
3. 阈值配置：`VOICEPRINT_AUTO_BIND_THRESHOLD` / `VOICEPRINT_ASK_THRESHOLD`
   （本地后端）；云端沿用现有 `voiceprint_auto_bind_min_confidence`。

## 6. 能力 B：未知说话人确认卡（可跳过）

- **触发**：会议 done 后详情页顶部（或说话人区）出现确认卡，列出每个
  待确认/未知 speaker；不打断浏览，可整卡忽略。
- **音频片段选取**（"给出一段或几段录音"）：每个 speaker 取发言时长最长的
  1~3 段（过滤 <3s 的碎句），前端用现有音频跳播直接播放对应区间；
  同时展示该段转录文本。
- **LLM 的角色（可选增强）**：用现有 QA 管线为该说话人生成一句提示
  （"此人主要负责 XX、提到过 YY"），帮用户回忆是谁——纯展示增强，
  不参与识人判定，走 flash 不耗 Air。
- **操作**：
  - 选择已有 Person（下拉/搜索）→ 走 `bind_speaker` 全流程（含隐式登记）；
  - 输入新名字 → 建 Person + 绑定 + 登记；
  - 「跳过」→ 记 dismissed，本会议不再询问（详情页仍可随时手动重命名）；
  - 中置信候选卡片直接给出「是张三吗？[是] [不是，选择其他] [跳过]」。

## 7. 阶段划分（每阶段独立 PR）

- **Phase 1（本地后端 + 自动标注闭环）**：voice_samples 迁移 + 重命名物化 +
  `match_local_voiceprints` 三档 + 高置信自动绑（带撤销）。provider 无关，
  mock 即可端到端测试。
- **Phase 2（确认卡 UI）**：待确认/未知列表 API + 前端确认卡（音频片段、
  三种操作、dismissed 落地）+（可选）LLM 发言提示。
- **Phase 3（云端补全）**：RegisterVoicePrint API 接入（重命名即自动注册）、
  删除 Person 云端同步清理自动化；consent 强制校验移除 + PRD §9.4 改写
  随本阶段落地。
- 顺带清欠：§7.2 原待建项中「绑定回滚 UI」并入 Phase 2。

## 8. 风险

| 风险 | 缓解 |
|---|---|
| 跨模型 embedding 不可比（换 ASR/声纹模型后历史样本失效） | 比对强制同 model_name/version；模型升级时旧样本自然淘汰（不删、不比） |
| diarization 错误传播（错误样本污染声纹库） | 只从**人工确认**的绑定登记样本；auto 绑定的样本不再二次入库（防漂移放大） |
| 自动绑错人（高置信误判） | 阈值保守起步 + 前端「自动识别」徽标 + 一键撤销（SpeakerBinding 追加式回滚现成） |
| 样本质量差（短发言/重叠音） | 登记时选最长干净段、过滤碎句；不足 10s（云端）则不注册仅本地物化 |
| mock/本地 embedding 维度不一（8 维 mock vs cam++ 192 维） | 比对按 model_name 分桶天然隔离；测试用 mock 桶 |

## 9. 待讨论问题

1. **合规改写范围**：§1 的立场（删强制 consent、保留级联删除与字段）是否符合
   预期？还是希望连 `consent_record` 字段也一并清理？
2. **后端优先级**：按 §7 先做本地（provider 无关、长期解）再补云端注册，
   还是你当前 SeedASR 用得多、希望先把「重命名即云端注册」做了？
3. **改绑/撤销时的样本处置**：跟随最新绑定重新归属，还是保守起见只解除
   归属（样本回到无主池）？
4. **dismissed 的落地方式**：倾向复用 `speaker_bindings`（confirmed_by=dismissed，
   person_id 允许空，需一条迁移放宽非空约束）；接受吗？
5. **自动绑定的 UX**：高置信直接绑（详情页徽标+可撤销）是否可接受，
   还是所有自动识别都先进确认卡、点一下才生效（更保守但多一步）？
6. **LLM 发言提示**（§6 可选增强）：Phase 2 就带上，还是先纯音频+文本？
7. **阈值初值**：本地 cosine 0.8 / 0.6 起步、实测后调——有无既有经验值想指定？
