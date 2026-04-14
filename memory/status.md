# 当前状态

> 更新时间：2026-04-14（session 收尾）

## 项目定位（明确）

**ReSearch v2 = 学术综述生成系统**。用户给主题，输出研究者能真用的综述。

**对标**：STORM (stanford-oval/storm, 13k stars)。

**不是**：研究平台、技术 showcase、方法论 testbed（之前 spin 过，被用户拆穿）。

## 当前状态：核心能力不可用

**Writer Agent 60.5% 引用错配**（P7 n=3 CI [56.2%, 66.7%] 确认）。

这意味着**生成的综述引用一半以上是错的，研究者不能直接用**。这是项目当前的致命硬伤。

## 今日完成的技术基础（支撑明天修复）

- Hybrid citation verifier（Attribution LLM-judge）
- Pipeline 集成 + async 重构
- n=3 repeated smoke test 框架（`experiments/p2_repeated_smoke_test.py`）
- Bootstrap CI（`src/research/evaluation/statistics.py`）
- Agent IO 追踪（用于验证 Agent 真实行为）

这些**基础设施**让明天修复 Writer 后**能立刻量化验证**效果。

## 明天主任务：P8 修复 Writer（生死任务）

**不是 feature work**，是**不修这个项目就不可用**。

**方案**：策略 1 + 策略 2 + 策略 3（详见 `/Users/lxl/.claude/projects/-Users-lxl--openclaw-code-research-agent/memory/project_apr14_writer_fix_plan.md`）

1. **Reader Claim Contract**（~2h）— Writer 只能用 Reader note 的 claim，禁止编造方法名
2. **Structured Citation Format**（~1h）— 每条引用显式引用 note 内容
3. **Attribution Self-Check Loop**（~3h）— Writer 自检 mismatch > 25% 就 revise

**验证标准**（硬指标，不是 spin）：
- ✅ Mismatch rate CI 上界 < 30%（从 [56.2%, 66.7%]）
- ✅ 至少 1 个 section 做到 100% matching
- ✅ 抽样 5 条 matching 引用给人读，觉得"能用"

**修不到的处理**：诚实记录，考虑换 Writer 模型（gpt-4o-mini → gpt-4o），或缩小范围。

## 其他技术能力（已验证稳定）

- **自进化机制**: Round 0 (6 queries) → Round 1 (38 queries, 0 重复)，3 次重复 Δ=+0.30 [+0.10, +0.47]
- **Multi-LLM Critic**: GPT-4o + Claude 交叉评估 + spread 分歧度
- **PDF 全文**: 提取率 96.7%, overall Δ+0.7
- **SurGE 对标**: Coverage 25% vs 最强基线 6.3%（有 methodology caveat）
- **RAG Ablation**: Hybrid best F1=0.740

## Next Session 启动 checklist

1. **先读**：
   - 本文件（status.md）
   - memory/tasks.md
   - `/Users/lxl/.claude/projects/-Users-lxl--openclaw-code-research-agent/memory/project_apr14_writer_fix_plan.md`
   - `/Users/lxl/.claude/projects/-Users-lxl--openclaw-code-research-agent/memory/feedback_avoid_spinning.md`（硬纪律）
2. **核查**：
   - `git log -1` 最新 commit 应是 "保存 session 收尾状态"
   - `conda run -n base python -m pytest -m "not integration" -q` 应 96 passed
3. **开工**：直接改 `src/research/agents/writer.py` 做策略 1
4. **验证**：跑 `experiments/p2_repeated_smoke_test.py` 看 mismatch CI 对比

## 硬纪律（切勿违反）

- ❌ 不要把 Writer 60% 错配再包装成任何亮点
- ❌ 不要扩大"research platform"叙事
- ❌ 不要在解决问题前先优化 README 叙事
- ✅ 先改代码 → 跑实验 → 看真实 IO → 再看能不能讲
