from pathlib import Path

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

    # 鉴权（PRD §9.2：全站 Bearer Token JWT，MVP 单默认用户）
    # HS256 密钥需 ≥32 字节；生产环境必须通过环境变量覆盖
    auth_secret: str = "dev-only-secret-change-me-0123456789abcdef"
    auth_password: str = "dev-password"
    token_ttl_hours: int = 24
    # 音频播放短时签名 URL 有效期（秒）
    audio_url_ttl_seconds: int = 600

    # LLM Router（PRD §4：统一 service 抽象；按序尝试，失败降级到下一个）
    llm_providers: str = "gemini,glm"  # 逗号分隔优先级；测试/无 key 联调可用 "mock"
    gemini_api_key: str = ""
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai"
    gemini_model: str = "gemini-3.5-flash"
    glm_api_key: str = ""
    glm_base_url: str = "https://open.bigmodel.cn/api/paas/v4"
    glm_model: str = "glm-4-flash"


settings = Settings()
