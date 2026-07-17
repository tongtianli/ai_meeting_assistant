# 声纹记忆增强技术设计 v1

> 状态：已定稿（PR #22 讨论结论已回写，§9 记录七项决议）
>
> 目标交互（产品诉求）：
> 1. 用户在之前的会议中对某说话人**重命名**后，后续会议出现该说话人时**系统自动标注**
>    （首版为「系统认为这是张三」的一键确认，验证准确率后再开启直接自动绑定）；
> 2. 会议中出现**未知说话人**时，系统给出一段或几段录音**询问用户这个人是谁**（可跳过）；
>    所有说话人均已识别时**完全静默**。

## 1. 合规立场变更（决议 1）

- **当前单用户版本不在产品内强制执行单独同意流程**；保留同意记录字段、
  物理级联删除和云端清理能力。未来进入多用户、团队或商业部署场景时，
  重新评估告知、授权及敏感信息处理要求。
- 实现要点：
  - `Person.consent_record` 字段保留且允许为空（历史数据不动）；
  - 注册声纹不再要求 `consent_note` 必填（去掉 `persons` 路由的强制校验）；
  - 删除 Person 继续物理级联删除本地 VoiceSample；
  - SeedASR 云端声纹继续提供删除时同步清理；
  - 预留部署级开关 `VOICEPRINT_REQUIRE_CONSENT=false`（非本期硬要求）。
- **PRD 修改点（随 Phase 1 实现 PR 执行）**：§9.4 声纹条目按上述措辞改写；
  Feature 2 / §7.2 的「合规同意流」从待建清单移除。

## 2. 现状盘点（代码事实）

| 组件 | 现状 |
|---|---|
| `VoiceSample` 表 | ✅ 每场会议留存 diarization speaker embedding（model_name/version、sample_start/end）；**缺 `speaker_label` 列**（ASR 抽象层有、入库时被丢弃）——Phase 1 前置迁移 |
| 本地 FunASR | ✅ 产出每说话人 spk_embedding（cam++ 系）|
| SeedASR 云端声纹 | ✅ 转写时 `voice_print_list` 匹配 + `auto_bind_voiceprints` 命中自动绑定；`Person.voiceprint_id` 已有；**注册是人工步骤**（RegisterVoicePrint API 已调研未接：16kHz WAV mono ≥10s、base64 ≤2MB）|
| `SpeakerBinding` | ✅ 追加式、可回滚、`confirmed_by(auto/human)` + `confidence` |
| `bind_speaker`（重命名） | ✅ 建/找 Person + 绑定 + 物化 segment.person_id；未触碰 VoiceSample 与云端注册 |
| 音频跳播 | ✅ 短时签名 URL + segment start_time——确认卡的「听一段」直接复用 |

## 3. 总体设计：声纹记忆双后端（决议 2：本地优先）

```
                 重命名 / 确认卡确认（bind_speaker）＝隐式声纹登记
                                   ↓
              ┌────────────────────┴───────────────────┐
        本地 embedding 后端（Phase 1）          云端声纹后端（SeedASR，Phase 3）
   按 binding 精确物化 VoiceSample 归属        切干净样本 → RegisterVoicePrint
   （provider 无关，任何 ASR 可用）            → 回填 Person.voiceprint_id
              └────────────────────┬───────────────────┘
                                   ↓
                     新会议 transcribe 完成后「识人」
              ┌────────────────────┴───────────────────┐
   本地：新 speaker embedding vs 长期参考         云端：转写响应自带
   样本（pgvector cosine，仅同                  voiceprint 命中
   model_name/version 可比）                    （现 auto_bind 逻辑）
              └────────────────────┬───────────────────┘
                                   ↓
                        统一匹配结果（跨后端同构）
        VoiceprintMatchResult(backend, person_id, score,
                              model_key, evidence_sample_ids)
                                   ↓
                          三档置信度统一收口
   高（score≥T_auto 且满足 margin/质量门槛）
        → 首版：确认卡显示「系统认为这是张三」一键确认/否认（自动绑默认关）
        → 校准后：VOICEPRINT_AUTO_BIND_ENABLED=true 时直接自动绑（徽标+可撤销）
   中 [T_ask, T_auto) → 确认卡候选询问
   低 < T_ask         → 未知说话人，确认卡开放式询问（可跳过）
```

- 本地后端优先的原因：`voice_samples.speaker_label`、按 binding 的样本物化、
  模型版本隔离、三档匹配、审计与撤销是**两类后端共同依赖的基础能力**；
  本地路径 provider 无关，覆盖 SeedASR 额度耗尽后回退 FunASR 的场景。
- SeedASR RegisterVoicePrint 可并行做小型技术验证，正式业务集成放
  Phase 3，不阻塞 Phase 1。

## 4. 能力 A：重命名即登记 + 跨会议识人

### 4.1 样本归属规则（决议 3：按 binding 精确追踪）

- `VoiceSample` 增加归属来源列 `assigned_by_binding_id`（指向产生本次归属的
  SpeakerBinding，可空）——样本来源可审计、可安全回滚；
- **人工绑定/改绑**：只处理**当前会议、当前 speaker_label** 的样本，精确转给
  新 Person（写 person_id + assigned_by_binding_id）；不迁移 Person 名下其他样本；
- **撤销绑定**：对应样本回到无主状态（person_id/assigned_by_binding_id 置空），
  **不物理删除**；
- **`confirmed_by=auto` 的绑定**：仅用于本会议标注，其样本**不自动成为长期
  参考样本**；用户确认自动结果后（升级为 human 确认），样本才升级为长期参考
  ——防止错误自动识别在声纹库中自我放大（防漂移）；
- 物理删除仅发生在删除音频或删除 Person 时。

"长期参考样本" 定义：`person_id 非空 且 assigned_by_binding_id 指向
confirmed_by=human 的有效绑定`——匹配时只比对长期参考样本。

### 4.2 新会议识人

管道 `_stage_transcribe` 落库后（现 `auto_bind_voiceprints` 位置扩展）：

- 云端命中：保持现有逻辑（Phase 3 前不变）；
- 本地比对（`match_local_voiceprints`）：本会议每个未绑 speaker 的 embedding，
  对**同 model_name/model_version** 的长期参考样本做 pgvector cosine top-k，
  按 Person 聚合后产出 `VoiceprintMatchResult`；
- **自动绑定门槛（决议 5/7）**——须同时满足，缺一进确认卡：
  1. `VOICEPRINT_AUTO_BIND_ENABLED=true`（**默认 false**，校准前不开）；
  2. top1 score ≥ `VOICEPRINT_AUTO_BIND_THRESHOLD`；
  3. top1 − top2 ≥ `VOICEPRINT_AUTO_BIND_MIN_MARGIN`（防两人声音接近时误绑）；
  4. 该 Person 长期参考样本数 ≥ `VOICEPRINT_MIN_REFERENCE_SAMPLES`；
  5. 样本质量达标（时长 ≥ `VOICEPRINT_MIN_SAMPLE_SECONDS`）；
  6. model_name/model_version 一致（分桶天然保证）。
- 自动绑定生效时：`SpeakerBinding(confirmed_by=auto, confidence=score)`，
  前端「自动识别」徽标 + 一键改名/撤销；其样本不入长期参考库（见 4.1）。

## 5. 数据与迁移

1. `voice_samples` 加 `speaker_label`（新数据管道直写；存量按 sample_start/end
   与 segments 时间重叠 best-effort 回填，歧义留空）；
2. `voice_samples` 加 `assigned_by_binding_id`（可空，FK → speaker_bindings，
   ON DELETE SET NULL）；
3. **`speaker_identity_dismissals` 独立轻量表**（决议 4，不复用/不放宽
   SpeakerBinding——其语义保持"真实人物绑定"）：
   ```text
   speaker_identity_dismissals
   - id
   - meeting_id
   - speaker_label
   - dismissed_at
   UNIQUE(meeting_id, speaker_label)
   ```
   pending 状态**不持久化**，动态计算：
   - 有有效 SpeakerBinding → 已识别；
   - 无绑定但有 dismissal → 本会议已跳过；
   - 两者皆无 → 未知，展示确认卡。
   跳过时插入 dismissal；之后手动绑定时**删除**对应 dismissal；再撤销绑定时
   dismissal 已清除，确认卡自然重新出现。
4. 配置（决议 7：0.8/0.6 为**未经校准的开发占位值，不构成可靠生产默认值**；
   完成真实数据校准前自动绑定默认关闭；阈值按 model_name+version 分桶校准）：
   ```env
   VOICEPRINT_AUTO_BIND_ENABLED=false   # 校准前恒 false
   VOICEPRINT_AUTO_BIND_THRESHOLD=0.8   # 占位
   VOICEPRINT_ASK_THRESHOLD=0.6         # 占位；≥0.6 进候选确认，<0.6 开放式询问
   VOICEPRINT_AUTO_BIND_MIN_MARGIN=0.1  # 占位
   VOICEPRINT_MIN_SAMPLE_SECONDS=5      # 占位
   VOICEPRINT_MIN_REFERENCE_SAMPLES=1   # 占位
   ```

## 6. 能力 B：未知说话人确认卡（决议 5/6）

- **触发**：会议 done 后详情页；仅当存在未知或待确认说话人时出现，
  **全部已识别则完全静默**；
- **内容（不加入 LLM 发言提示，不做职责/职位/意图推理）**：
  - 每个 speaker 取发言时长最长的 1~3 段（过滤 <3s 碎句）音频，
    点击用现有跳播直接听；
  - 对应原文文本与时间位置；
  - 高置信候选：「系统认为这是张三 [确认] [不是，选择其他] [跳过]」；
  - 操作：选择已有 Person（下拉/搜索）/ 输入新名字 / 跳过。
- 确认/输入新名字 → 走 `bind_speaker` 全流程（含 4.1 的隐式登记）；
  跳过 → 写 dismissal。
- 该能力 = 声纹匹配结果 + 音频证据 + 用户确认，**不需要额外 LLM 调用**。

## 7. 阶段划分（每阶段独立 PR）

- **Phase 1（本地后端基础，本期）**：
  - 迁移：voice_samples 加 speaker_label + assigned_by_binding_id（含存量回填）、
    speaker_identity_dismissals 表；
  - 重命名钩子：按 binding 精确物化样本归属（含改绑/撤销规则 4.1）；
  - `match_local_voiceprints` + `VoiceprintMatchResult` + 全部门槛配置
    （自动绑定默认关闭，匹配结果先落日志与内部接口）；
  - consent 强制校验移除 + PRD §9.4 改写；
  - 测试：归属物化/改绑/撤销、同模型分桶、margin/质量门槛、
    dismissal 动态状态、mock 端到端。
- **Phase 2（确认卡 UI）**：待确认/未知列表 API（含高置信候选与证据片段）+
  前端确认卡（音频、三种操作、dismissed）+ 绑定回滚 UI（清 §7.2 旧欠账）。
- **Phase 3（云端补全 + 自动绑定开启评估）**：SeedASR RegisterVoicePrint
  业务集成（重命名即云端注册）、删除 Person 云端清理自动化；
  用真实数据校准阈值后评估开启 `VOICEPRINT_AUTO_BIND_ENABLED`。

## 8. 风险

| 风险 | 缓解 |
|---|---|
| 跨模型 embedding 不可比 | 比对强制同 model_name/version 分桶；模型升级旧样本自然淘汰 |
| diarization 错误传播污染声纹库 | 只有 human 确认的绑定产生长期参考样本；auto 样本不二次入库（防漂移） |
| 自动绑错人 | 首版自动绑定默认关闭、先收集准确率；开启后有 margin+质量多重门槛、徽标+一键撤销 |
| 样本质量差（短发言/重叠音） | 最长干净段优先、碎句过滤、最低时长门槛；云端注册不足 10s 不注册 |
| mock/本地维度不一（8 维 vs cam++ 192 维） | model_name 分桶天然隔离；测试走 mock 桶 |

## 9. 决议记录（PR #22，2026-07-16）

1. 去强制 consent，保留字段与删除/清理能力；措辞避免"单用户因此不需要授权"，
   写明多用户/商业部署时重新评估；
2. 本地后端优先，SeedASR 云端注册 Phase 3 后补（可并行技术验证）；
   预留统一 `VoiceprintMatchResult` 结构；
3. 样本按对应 binding 精确改绑，撤销回无主池不删除；auto 绑定样本不成为
   长期参考，human 确认后才升级；加 `assigned_by_binding_id` 溯源；
4. dismissed 用独立轻量表（UNIQUE(meeting_id, speaker_label)），
   不复用 SpeakerBinding；pending 动态计算不持久化；
5. 首版先确认卡收集真实准确率，自动绑定默认关闭；开启条件含
   top1 阈值 + top1-top2 margin + 参考样本数 + 样本质量 + 同模型；
6. 未知/待确认才询问、全识别静默；确认卡不加入 LLM 身份提示；
7. 0.8/0.6 仅为未校准占位值；阈值按模型分桶校准，校准前自动绑定不开启。
