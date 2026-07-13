"""确定性 Mock LLM：无 key 联调与测试用，输出与 Mock ASR 剧本自洽的合法纪要 JSON。"""
import json

from app.services.llm.base import LLMProvider, LLMResponse

_SUMMARY_JSON = {
    "title": "项目周会",
    "participants": ["speaker_001", "speaker_002"],
    "summary": "本次周会同步了后端与前端进展，后端接口完成八成，前端列表页与详情页联调完毕，并确认了上传接口的开发计划。",
    "topics": [
        {
            "title": "后端接口进展",
            "owner": "speaker_001",
            "items": [
                "接口开发已完成约八成，剩余部分同步推进。",
                "上传接口列入下周开发计划，按期落地执行。",
            ],
        },
        {
            "title": "前端联调与数据库迁移",
            "owner": "speaker_002",
            "items": [
                "列表页和详情页联调完毕。",
                "数据库迁移方案确认无异议，按方案执行。",
            ],
        },
    ],
    "decisions": ["下周完成上传接口开发"],
    "todos": [
        {
            "task": "完成上传接口的开发",
            "owner": "speaker_001",
            "deadline": "周五",
            # 引用 seq=4（"那这项任务记给我"，speaker_001 认领）——与 mock ASR 剧本自洽
            "source_segment_seq": 4,
        }
    ],
}


class MockLLMProvider(LLMProvider):
    name = "mock"
    model = "mock-llm"

    async def complete(
        self,
        system: str,
        user: str,
        json_mode: bool = True,
        temperature: float = 0.2,
    ) -> LLMResponse:
        text = (
            json.dumps(_SUMMARY_JSON, ensure_ascii=False)
            if json_mode
            else _SUMMARY_JSON["summary"]
        )
        return LLMResponse(
            text=text,
            provider=self.name,
            model=self.model,
            prompt_tokens=len(user) // 4,
            completion_tokens=len(text) // 4,
        )
