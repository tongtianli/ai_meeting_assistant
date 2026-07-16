from pathlib import Path
from typing import Literal

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = (
        "postgresql+asyncpg://postgres:postgres@localhost:5432/meeting_assistant"
    )
    # 音频与中间产物根目录（MVP 本地磁盘；二期换对象存储时仅换存储实现）
    data_dir: Path = Path("./data")
    # ASR provider 选择（接口可替换，PRD §10）
    asr_provider: str = "mock"
    # 转码前处理滤镜链（远场/小音量会议：去低频隆隆声 → 降噪 → 响度归一化，
    # 避免 VAD 把音量低的语音段当静音丢弃）；置空字符串可禁用
    transcode_filters: str = "highpass=f=80,afftdn,loudnorm=I=-16:TP=-1.5:LRA=11"
    # FunASR VAD 语音/噪音判定阈值，越低越不容易丢弃小音量语音（官方默认 0.6）。
    # 经验：paraformer（漏识别倾向）用 0.5；Fun-ASR-Nano（幻觉倾向）用 0.6+
    funasr_speech_noise_thres: float = 0.5
    # VAD 单段最大时长（毫秒）：限制长段中混入噪音区间的面积，
    # 降低 LLM-based 模型的幻觉面（官方默认 60000）
    funasr_vad_max_segment_ms: int = 30000
    # FunASR 主识别模型：paraformer-zh（快）或 FunAudioLLM/Fun-ASR-Nano-2512
    # （LLM-based，识别力更强、CPU 更慢；VAD/标点/声纹模型不随之变化）
    funasr_model: str = "paraformer-zh"

    # 通义听悟（云 ASR，会议场景；需先上传 OSS 供听悟拉取）
    aliyun_access_key_id: str = ""
    aliyun_access_key_secret: str = ""
    tingwu_app_key: str = ""
    tingwu_region: str = "cn-beijing"
    oss_endpoint: str = ""  # 如 https://oss-cn-beijing.aliyuncs.com
    oss_bucket: str = ""
    oss_prefix: str = "meeting-audio/"
    tingwu_poll_interval_seconds: float = 10.0
    tingwu_timeout_seconds: float = 3600.0

    # 火山引擎 Seed ASR 2.0（大模型录音文件识别；base64 直传，无需对象存储）
    volc_app_key: str = ""  # 控制台 AppID
    volc_access_key: str = ""  # 控制台 AccessToken
    volc_resource_id: str = "volc.bigasr.auc"  # 模型资源号，按控制台显示调整
    volc_base_url: str = "https://openspeech.bytedance.com/api/v3/auc/bigmodel"
    # 直传体积上限：网关约 16MB 请求体（实测），base64 膨胀 4/3 → 原始音频 ≤11MB；
    # 超过则按时长自适应码率压成 mp3 再上传
    seedasr_max_upload_mb: int = 11
    # 附加请求参数（JSON 对象字符串），如声纹匹配：
    # SEEDASR_EXTRA_REQUEST={"voice_print_list": ["vp-xxx"]}
    seedasr_extra_request: str = ""
    seedasr_poll_interval_seconds: float = 5.0
    seedasr_timeout_seconds: float = 3600.0
    # 本服务的公网入口（如 Cloudflare 隧道域名）。设置后，超过直传上限的
    # 音频改走 URL 模式：云端凭短时签名 URL 拉取文件，不再压低码率
    public_base_url: str = ""
    # URL 模式签名的有效期：需覆盖任务排队 + 云端拉取的全过程
    seedasr_url_ttl_seconds: int = 7200
    # 声纹自动绑定的最低置信度（0 = 信任云端阈值，命中即绑定；
    # 返回带 score 时可调高做二次过滤，如 0.6）
    voiceprint_auto_bind_min_confidence: float = 0.0

    # 鉴权（PRD §9.2：全站 Bearer Token JWT，MVP 单默认用户）
    # HS256 密钥需 ≥32 字节；生产环境必须通过环境变量覆盖
    auth_secret: str = "dev-only-secret-change-me-0123456789abcdef"
    auth_password: str = "dev-password"
    token_ttl_hours: int = 24
    # 音频播放短时签名 URL 有效期（秒）
    audio_url_ttl_seconds: int = 600

    # LLM Router（PRD §4：统一 service 抽象；按序尝试，失败降级到下一个）
    # legacy：所有任务共用 LLM_PROVIDERS；task_based：按任务类型独立路由
    # （Tech Design M4：出问题时回滚 LLM_ROUTING_MODE=legacy 即可）。
    # Literal 严格校验：拼写错误在启动阶段直接失败，不静默落入 legacy
    llm_routing_mode: Literal["task_based", "legacy"] = "task_based"
    llm_providers: str = "gemini,glm"  # legacy 模式顺序；测试/无 key 联调可用 "mock"
    # 任务级路由（task_based 模式，Tech Design M4 §6/§19）：
    # 高价值任务（最终纪要/纪要编辑）优先消耗 GLM-4.5-Air 赠送额度，
    # 轻量任务（map/QA/改写/分类）走免费 Flash，Gemini 保留跨供应商灾备
    summary_map_providers: str = "glm_flash,glm_air,gemini"
    summary_final_providers: str = "glm_air,glm_flash,gemini"
    summary_edit_providers: str = "glm_air,glm_flash,gemini"
    qa_providers: str = "glm_flash,gemini,glm_air"
    query_rewrite_providers: str = "glm_flash,gemini"
    intent_classify_providers: str = "glm_flash,gemini"
    # 质量裁判（Phase 4 hybrid）：走免费 Flash，默认不消耗 Air 赠送额度
    quality_check_providers: str = "glm_flash,gemini"
    # 纪要范例库 few-shot（公司文风第二层）：注入 prompt 的范例数量与总字符预算。
    # 超预算的范例自动跳过；调大可强化文风模仿，代价是每次摘要的 token 消耗
    summary_examples_max_count: int = 3
    summary_examples_max_chars: int = 6000
    # 全局术语表注入 ASR 的热词数量上限（对齐 SeedASR/听悟的 100 上限；
    # FunASR 不做数量限制，故在管道侧统一截断，见 pipeline._stage_transcribe）
    glossary_max_terms: int = 100
    # 长会议 map/reduce 分块（Tech Design M4 §13，Phase 4）：按字符估算 token
    # 切块（不引入分词依赖），达 target 收口、单行超 max 独占块；相邻块重叠
    # overlap_segments 行以防跨块 TODO/决策丢失（重复由 reduce 去重兜底）
    summary_chunk_target_tokens: int = 5000
    summary_chunk_max_tokens: int = 7000
    summary_chunk_overlap_segments: int = 2
    summary_chars_per_token: float = 2.0  # 中文约 2 字符/token 的粗估
    # 纪要质量检查（Tech Design M4 §12/Phase 4）：
    # off=紧急回滚跳过；rules=默认，纯确定性启发式；hybrid=规则+条件式 LLM 裁判
    summary_quality_check_mode: Literal["off", "rules", "hybrid"] = "rules"
    gemini_api_key: str = ""
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai"
    gemini_model: str = "gemini-3.5-flash"
    glm_api_key: str = ""
    glm_base_url: str = "https://open.bigmodel.cn/api/paas/v4"
    glm_model: str = "glm-4-flash"  # legacy 模式的单一 glm 节点模型
    # task_based 模式下同一 GLM key 的两个独立路由节点（额度/统计可区分）。
    # 模型 ID 上线前须经控制台+真实调用确认，不在代码中静默替换：
    # flash 默认沿用已在生产验证的 glm-4-flash，确认账号支持新版免费
    # Flash（如 glm-4.7-flash）后再通过环境变量切换
    glm_air_model: str = "glm-4.5-air"
    glm_flash_model: str = "glm-4-flash"

    # 资源包信息（Tech Design M4 §12：仅用于应用侧用量预警，控制台余额是最终真值）
    glm_air_grant_total_tokens: int = 12_000_000
    glm_general_grant_total_tokens: int = 1_996_701
    glm_grant_expires_at: str = "2026-10-10T08:40:18+08:00"

    # 额度软限制与付费熔断（Tech Design M4 §12.3，Phase 3）。应用侧按
    # llm_usage_records 的账号级累计估算，不能完全阻止服务商计费——
    # 必须同时在智谱控制台配置余额预警/停机保护
    glm_allow_paid_after_grant: bool = False  # 到期/耗尽后是否允许继续调用 Air（付费）
    glm_air_soft_limit_tokens: int = 11_400_000  # 达到后 Air 仅供高价值任务（95%）
    glm_general_soft_limit_tokens: int = 1_890_000  # 通用包仅预警（embedding 不可降级混用）
    glm_quota_cache_ttl_seconds: int = 60  # 配额快照缓存；0 = 每次调用都查库

    # AI 问答 RAG（PRD Feature 5）。embedding 不能像 LLM 那样降级混用
    # （不同模型向量空间不通），故单一 provider、无 fallback。
    embedding_provider: str = "glm"  # glm | gemini | mock（key 缺失时报错不降级）
    glm_embedding_model: str = "embedding-3"
    gemini_embedding_model: str = "text-embedding-004"
    embedding_batch_size: int = 32
    qa_top_k: int = 6  # 向量召回 anchor 数
    # 问题命中说话人真名时，额外并入的片段数（说话人感知混合检索，
    # 每位命中的说话人独立取 top-k，多人同问不互相挤占）。
    # 已由 qa_speaker_top_k_per_person 取代（语义相同、名字更明确），
    # 本名保留一版做兼容 fallback
    qa_speaker_top_k: int = 6
    # ---- RAG Phase 1（TECH_DESIGN_MEETING_RAG_V1 §4.1.1）----
    qa_speaker_top_k_per_person: int | None = None  # None → 沿用 qa_speaker_top_k
    qa_max_speaker_persons: int = 4  # 匹配说话人数上限，超出按首次出现序截断
    qa_min_speaker_anchors_per_person: int = 1  # 每个命中 person 的保底 speaker anchor
    qa_neighbor_window: int = 2  # anchor 邻居 ± 窗口
    qa_max_context_segments: int = 24  # 最终 context 上限（anchor 优先预算）
    qa_rewrite_recent_messages: int = 6  # 改写输入的最近用户消息条数
    qa_rewrite_cited_max_segments: int = 3  # 指代改写附带的上轮引用原文数上限
    qa_rewrite_cited_max_chars: int = 1200  # 附带引用原文的字符预算
    qa_retrieval_debug_text: bool = False  # true 才在检索日志记录问题明文
    # ---- RAG Phase 2：关键词召回 + RRF 融合（§5，评审 §8 约束）----
    qa_keyword_max_tokens: int = 8  # 每次查询最多提取的精确 token 数
    qa_keyword_top_k_per_token: int = 4  # 每个 token 独立候选限额（防高频词占满）
    qa_keyword_min_token_len: int = 3  # 普通英文 token 最短长度（版本/编号/金额/日期豁免）
    qa_rrf_k: int = 60  # RRF 常数；初始值非验证最优，真实评估建议对比 10/30/60

    @property
    def qa_per_person_k(self) -> int:
        return (
            self.qa_speaker_top_k_per_person
            if self.qa_speaker_top_k_per_person is not None
            else self.qa_speaker_top_k
        )

    @model_validator(mode="after")
    def _validate_chunking(self) -> "Settings":
        """分块参数在启动阶段即校验，避免环境变量错误让摘要管线除零/异常分块。"""
        if self.summary_chars_per_token <= 0:
            raise ValueError("SUMMARY_CHARS_PER_TOKEN 必须 > 0")
        if self.summary_chunk_target_tokens <= 0:
            raise ValueError("SUMMARY_CHUNK_TARGET_TOKENS 必须 > 0")
        if self.summary_chunk_max_tokens < self.summary_chunk_target_tokens:
            raise ValueError(
                "SUMMARY_CHUNK_MAX_TOKENS 必须 >= SUMMARY_CHUNK_TARGET_TOKENS"
            )
        if self.summary_chunk_overlap_segments < 0:
            raise ValueError("SUMMARY_CHUNK_OVERLAP_SEGMENTS 必须 >= 0")
        return self

    @model_validator(mode="after")
    def _validate_qa_retrieval(self) -> "Settings":
        """RAG 检索参数启动校验（Phase 1 §4.1.1）：错误配置直接拒绝启动，
        不让"每人保底"与上下文预算语义被静默破坏。"""
        non_negative = {
            "QA_TOP_K": self.qa_top_k,
            "QA_SPEAKER_TOP_K": self.qa_speaker_top_k,
            "QA_MAX_SPEAKER_PERSONS": self.qa_max_speaker_persons,
            "QA_MIN_SPEAKER_ANCHORS_PER_PERSON": self.qa_min_speaker_anchors_per_person,
            "QA_NEIGHBOR_WINDOW": self.qa_neighbor_window,
            "QA_REWRITE_RECENT_MESSAGES": self.qa_rewrite_recent_messages,
            "QA_REWRITE_CITED_MAX_SEGMENTS": self.qa_rewrite_cited_max_segments,
            "QA_REWRITE_CITED_MAX_CHARS": self.qa_rewrite_cited_max_chars,
        }
        if self.qa_speaker_top_k_per_person is not None:
            non_negative["QA_SPEAKER_TOP_K_PER_PERSON"] = (
                self.qa_speaker_top_k_per_person
            )
        for name, value in non_negative.items():
            if value < 0:
                raise ValueError(f"{name} 必须 >= 0")
        if self.qa_max_context_segments <= 0:
            raise ValueError("QA_MAX_CONTEXT_SEGMENTS 必须 > 0")
        if self.qa_min_speaker_anchors_per_person > self.qa_per_person_k:
            raise ValueError(
                "QA_MIN_SPEAKER_ANCHORS_PER_PERSON 不得大于每人召回数 "
                "QA_SPEAKER_TOP_K_PER_PERSON（否则保底数无法满足）"
            )
        if (
            self.qa_max_speaker_persons * self.qa_min_speaker_anchors_per_person
            > self.qa_max_context_segments
        ):
            raise ValueError(
                "QA_MAX_SPEAKER_PERSONS × QA_MIN_SPEAKER_ANCHORS_PER_PERSON "
                "不得超过 QA_MAX_CONTEXT_SEGMENTS（否则满员点名时无法保证每人保底）"
            )
        for name, value in {
            "QA_KEYWORD_MAX_TOKENS": self.qa_keyword_max_tokens,
            "QA_KEYWORD_TOP_K_PER_TOKEN": self.qa_keyword_top_k_per_token,
            "QA_KEYWORD_MIN_TOKEN_LEN": self.qa_keyword_min_token_len,
        }.items():
            if value < 0:
                raise ValueError(f"{name} 必须 >= 0")
        if self.qa_rrf_k <= 0:
            raise ValueError("QA_RRF_K 必须 > 0")
        return self


settings = Settings()
