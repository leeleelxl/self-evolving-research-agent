# 任务看板

> 更新时间：2026-04-14（session 收尾）

## 项目定位（明确）

**学术综述生成系统**（对标 STORM）。不是研究平台，不是 showcase。

## ✅ 已完成（技术基础，不是亮点）

| # | 任务 | 状态 | 备注 |
|---|------|:----:|------|
| Wave 1 | Limitations + Opt-in + 数据溯源 | ✅ | 诚实度修复 |
| P0 | Agent IO 可观测化 | ✅ | 验证工具 |
| P1 | E2E 集成测试 | ✅ | 工程质量 |
| P3 | Bootstrap CI | ✅ | 统计工具 |
| P4 | NLI Calibration → precision=0% | ✅ | 否定发现 |
| P5 | NLI → Attribution pivot | ✅ | 代码重构 |
| P6 | Attribution Calibration | ✅ | multi-class 62.5% |
| P2 | Pipeline 集成 + async | ✅ | 发现 Writer 错配 |
| P7 | n=3 CI 确认 | ✅ | Writer 60.5% [56.2, 66.7] |

## 🔴 P8 明天做 — 修复 Writer（生死任务）

**Why this is critical**: 项目是做综述生成。Writer 60% 错配 = 综述不可用 = 项目无法使用。

| 步骤 | 内容 | 成本 |
|------|------|:----:|
| 8.1 | Reader Claim Contract：改 writer.py prompt 禁止编造方法名 | 2h |
| 8.2 | Structured Citation Format：每条引用显式引用 note 内容 | 1h |
| 8.3 | 跑 p2_repeated_smoke_test.py n=3 看 mismatch CI | 1h (实验) |
| 8.4 | 如 8.1+8.2 不够，加 Attribution Self-Check Loop | 3h |
| 8.5 | 如都不够，换 Writer 模型（gpt-4o-mini → gpt-4o） | 1h + API $ |

**硬成功标准**：
- Mismatch rate CI 上界 < 30%
- 至少 1 个 section 100% matching
- 抽样 5 条 matching 引用真研究者觉得能用

**硬失败处理**：
- 诚实记录"这路径走不通"
- 不要 spin 为亮点
- 考虑换模型或缩小 domain

## 🟡 P9 延后（修好 Writer 后再考虑）

| # | 任务 | 前提 |
|---|------|------|
| P9.1 | 对标 STORM 的 perspective-guided writing | Writer 可用 |
| P9.2 | outline-driven RAG（STORM 的另一个核心） | Writer 可用 |
| P9.3 | Streamlit demo | Writer 可用 |
| P9.4 | 自进化 n 扩 | 低优 |

## 纪律提醒

**避免的反模式**（今天被用户拆穿过）：
- ❌ 修完 Writer 前不要做其他 feature
- ❌ 不要再写 README 把 60% 错配包装成亮点
- ❌ 不要把项目重新定位成"研究平台" 绕开 Writer 问题

**工作流**：先改代码 → 跑真实验 → 看 IO → 再决定叙事。
