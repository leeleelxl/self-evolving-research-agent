# ReSearch v2 — 自进化多 Agent 学术研究系统

自动化学术文献检索、精读、综述生成的多 Agent 系统。核心差异化：Critic 驱动的自进化机制 + 完整消融实验。

## 技术栈

Python 3.11+ | Claude API (Anthropic SDK) | FAISS + BM25 | Semantic Scholar API | arXiv API

## 关键命令

```bash
uv run pytest                    # 运行测试
uv run python -m research        # 运行主 pipeline
uv run python -m research.eval   # 运行评估
```

## 硬性规则

1. **回答使用中文**
2. **保持诚实**：不夸大能力，不编造性能数据
3. **所有实验必须有量化数据**：用 experiments/ 下的脚本复现
4. **提交前必须测试通过**
5. **README 面向面试官**：突出"解决了什么问题"和"技术亮点"
6. **控制输出量**：大文件用 offset/limit 定位

## Session 启动必读

| 文件 | 内容 |
|------|------|
| `memory/status.md` | 当前进度 + 下一步行动 |
| `memory/tasks.md` | 活跃任务看板 |

## 按需读取

| 触发条件 | 文件 |
|---------|------|
| 架构设计相关 | `docs/architecture.md` |
| 技术选型决策 | `docs/tech_decisions.md` |
| 经验教训 | `memory/retro.md` |
| 项目调研背景 | `docs/research_report.md` |

## 信息权威表 (SSOT)

| 信息类型 | 权威文件 | 说明 |
|---------|---------|------|
| 当前进度 | `memory/status.md` | 每个 milestone 更新 |
| 任务列表 | `memory/tasks.md` | 完成即标记 |
| 技术决策 | `docs/tech_decisions.md` | 记录 why 不只 what |
| 踩坑记录 | `memory/retro.md` | 可复用的经验 |
| 架构设计 | `docs/architecture.md` | 图 + 文字 |
