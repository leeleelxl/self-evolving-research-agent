# 任务看板

> 更新时间：2026-04-14（session 收尾）

## ✅ P0 核心系统（全部完成）

| # | 任务 | 状态 |
|---|------|:----:|
| 1-15 | 5 Agent + KB + PDF + 错误处理 + Prompt | ✅ |

## ✅ P1 实验验证（全部完成）

| # | 任务 | 状态 |
|---|------|:----:|
| 16-20 | 端到端 + 自进化 + RAG/KB/PDF 消融 + README | ✅ |

## ✅ P2 架构升级 + 诚实度（全部完成）

| # | 任务 | 状态 | 成果 |
|---|------|:----:|------|
| Wave 1 | 诚实度修复（Limitations + Opt-in + 数据溯源） | ✅ | 诚实定位 5→7 |
| P0 | Agent IO 可观测化（AgentTrace + inspect 工具） | ✅ | 发现 38 queries 0 重复 |
| P1 | 真 API E2E 集成测试 | ✅ | 消除 anti-overselling -1 |
| P3 | Bootstrap 95% CI | ✅ | Evolution Δ +0.30 [+0.10, +0.47] |
| P4 | NLI Calibration | ✅ | precision=0% 诚实披露 |
| P5 | Attribution Method Pivot | ✅ | NLI → LLM-judge |
| P6 | Attribution Calibration | ✅ | multi-class 62.5% |
| **P2** | Pipeline 集成 Citation Verification | ✅ | + async 重构修 bug |
| **P7** | n=3 重复 smoke test + CI | ✅ | Writer 60.5% [56.2, 66.7] 确认 |

## 🔥 P8 明天做 — 修复 Writer Attribution

**优先级**: 最高（硬伤不是亮点，光诊断不修 = spin）

| 步骤 | 内容 | 成本 |
|------|------|:----:|
| 8.1 | Writer prompt 加 structured citation 约束 | 1h |
| 8.2 | Reader-Writer contract（禁止 Writer 编造方法名）| 2h |
| 8.3 | 跑 p2_repeated_smoke_test.py n=3 验证修复效果 | 1h (实验) |
| 8.4 | 如果 mismatch 仍 > 40%，做 Self-check via Attribution | 3h |
| 8.5 | README 更新 "修复前 60% → 修复后 X%" 对比 | 0.5h |

**成功标准**: mismatch rate 60% → 20-30%
**面试叙事**: 从 "发现问题" → "诊断 + 修复 + 验证" 完整闭环

## 🟡 P9 可选优化（时间允许）

| # | 任务 | 价值 |
|---|------|------|
| P9.1 | Streamlit demo + README GIF | 面试第一印象 +1 |
| P9.2 | 内容场景 vertical demo（给小红书/快手叙事） | 核心 3 家平均 +0.5 |
| P9.3 | 自进化扩 n=3 → n=10 | 更强的 CI |

## 🟢 P2 完成后可以 stop 并转面试准备

当前 72/90 强推级已够。P8 修完 Writer 后 ~74/90。继续刷分边际递减。
