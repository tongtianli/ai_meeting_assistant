"""确定性 Mock LLM：无 key 联调与测试用，输出与 Mock ASR 剧本自洽的合法纪要 JSON。"""
import json
import re

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
        if "事实抽取器" in system:  # 长会议 map 分支（SYSTEM_MAP_EXTRACT 稳定标记）
            # 从本块 user 里取 [seq]，产出块特异 MapFacts——相邻块 partial 不同，
            # 可验证 reduce 的合并/去重；todo 溯源到本块内某真实 seq
            seqs = [int(m) for m in re.findall(r"\[(\d+)\]", user)]
            facts = {
                "topics": [
                    {"title": "本部分要点", "owner": None, "items": ["阶段性事实要点。"]}
                ],
                "decisions": [],
                "todos": [
                    {
                        "task": "完成上传接口的开发",
                        "owner": "speaker_001",
                        "deadline": "周五",
                        "source_segment_seq": seqs[0] if seqs else 0,
                    }
                ],
                "risks": [],
                "open_questions": [],
                "source_segment_seqs": seqs,
            }
            text = json.dumps(facts, ensure_ascii=False)
        elif "质量审查员" in system:  # 质量裁判分支（SYSTEM_QUALITY_JUDGE 稳定标记）
            # 确定性：把所有可疑项判为误报（confirmed 空）——hybrid 测试据此
            # 断言裁判过滤掉规则误报
            text = json.dumps({"confirmed": []}, ensure_ascii=False)
        elif "意图分类器" in system:  # 意图+改写分支（SYSTEM_INTENT 稳定标记）
            # 只对"当前问题"做关键词判定（历史消息里的动词不算数）
            m = re.search(r"当前问题：(.+)\s*$", user, re.S)
            question = (m.group(1) if m else user).strip()
            is_edit = any(k in question for k in ("改", "删", "换", "合并", "加上"))
            # 确定性指代消解：多轮（带历史块）且问题含 他/她 时，用名单第一个名字替换
            standalone = question
            if "最近的用户提问" in user and re.search(r"[他她]", question):
                nm = re.search(r"说话人名单：([^\n、]+)", user)
                if nm:
                    standalone = re.sub(r"[他她]", nm.group(1).strip(), question)
            text = json.dumps(
                {"intent": "edit" if is_edit else "query", "standalone_query": standalone},
                ensure_ascii=False,
            )
        elif "纪要编辑器" in system:  # 改纪要分支（SYSTEM_SUMMARY_EDIT 稳定标记）
            edited = dict(_SUMMARY_JSON)
            edited["decisions"] = ["【已修改】下周完成上传接口开发并同步验收标准"]
            text = json.dumps(edited, ensure_ascii=False)
        elif "会议问答助手" in system:  # QA 分支（SYSTEM_QA 稳定标记）
            # 引用检索到片段中 seq 最小的一条，保证落在真实 segment 上
            seqs = [int(m) for m in re.findall(r"\[(\d+)\]", user)]
            cited = [min(seqs)] if seqs else []
            payload = {
                "answer": "根据会议记录，相关内容见引用片段。",
                "cited_segment_seqs": cited,
                "confidence": "high" if cited else "low",
            }
            text = json.dumps(payload, ensure_ascii=False)
        elif json_mode:
            text = json.dumps(_SUMMARY_JSON, ensure_ascii=False)
        else:
            text = _SUMMARY_JSON["summary"]
        return LLMResponse(
            text=text,
            provider=self.name,
            model=self.model,
            prompt_tokens=len(user) // 4,
            completion_tokens=len(text) // 4,
        )
