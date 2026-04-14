# 当前状态

> 更新时间：2026-04-14（session 收尾）

## 阶段：P7 完成，诚实面对 Writer 系统性缺陷，明天修复

评分轨迹：55/90 (起始) → 63 (Wave 1) → 68 (P5) → 71 (P2+P6) → **~72 (P7)**，🟢 **强推级**

## 今日完成

- **P0**: Agent IO 可观测化（AgentTrace + inspect_agent_io.py + trace_demo）
- **P1**: 真 API E2E 集成测试（2 个 integration tests）
- **P3**: Bootstrap 95% CI（Evolution Δ=+0.30 [+0.10, +0.47]）
- **P4**: NLI Calibration → 发现 precision=0%（诚实披露）
- **P5**: Attribution Method Pivot（NLI → LLM-judge）
- **P6**: Attribution Calibration（multi-class 62.5%, recall=100%）
- **P2**: Pipeline 集成 Citation Verification + async 重构
- **P7**: 重复 smoke test n=3 → **Writer 60.5% 错配，CI [56.2%, 66.7%]**

## 明天主任务：P8 修复 Writer Attribution 问题

**为什么必须做**：用户质疑"这个实验的价值就是判定 writer agent 不可靠？" — 确实是。
Writer 60% 错配是**硬伤不是亮点**，光发现不修等于 spin。

**修复策略**（按 ROI 排序，先做策略 1）：

1. **Structured citation prompt**（~1h）
   - Writer prompt 强制每条声称 quote abstract 具体表述
   - 当前 writer.py 让 Writer 自由写，改为"只能用 abstract 明确提到的内容"

2. **Reader-Writer contract**（~2h）
   - Writer 只能用 Reader note.core_contribution 里的 claim
   - Prompt 明确禁止使用 note 中未出现的方法名

3. **Self-check via Attribution**（~3h）
   - Writer 生成后自己跑 attribution，mismatch > threshold 就 revise

**验证方法**：复用 `experiments/p2_repeated_smoke_test.py` 跑 n=3，对比 CI。

**成功标准**：mismatch rate 60% → 20-30%

## 核心数据（面试可讲）

### 自进化（3 次重复, Multi-LLM Critic）
- Overall 7.79 ± 0.07, **Evolution Δ = +0.30, 95% CI [+0.10, +0.47]**
- Round 0 (6 queries) → Round 1 (38 queries, 0 重复) — 真 diverge

### PDF 全文消融
- Abstract 7.2 → Full-text 7.9, **Δ+0.7, depth +0.9**
- 提取率 96.7%

### SurGE 外部对标
- Coverage 25.0% vs 最强基线 6.3%（4x，有 methodology caveat）

### Hybrid 引用验证（v6, n=3）
- **Writer 60.5% attribution 错配, CI [56.2%, 66.7%]** ← 硬伤待修复
- NLI 矛盾 precision=0%（已 deprecate）
- Attribution multi-class agreement 62.5%

## 技术栈（已实现）

Python 3.11+ | OpenAI SDK (中转站) | Multi-LLM: GPT-4o + Claude | FAISS + BM25 | 
Semantic Scholar + arXiv | fastembed BGE-small | sentence-transformers (optional) | 
pypdf | Pydantic v2 | structlog | pytest (96 unit + 20 integration)

## Next Session 启动检查

1. 读此文件（status.md）
2. 读 `memory/tasks.md`
3. 读 `/Users/lxl/.claude/projects/-Users-lxl--openclaw-code-research-agent/memory/project_apr14_writer_fix_plan.md`
4. 读 `/Users/lxl/.claude/projects/-Users-lxl--openclaw-code-research-agent/memory/feedback_avoid_spinning.md`
5. 检查 git log：最新 commit 应是 `e07b6f8 feat: P7 P2 Smoke Test n=3 + Bootstrap CI`
6. 跑 `conda run -n base python -m pytest -m "not integration" -q` 确认 96 tests 通过
7. 开始 P8 策略 1（writer.py prompt）
