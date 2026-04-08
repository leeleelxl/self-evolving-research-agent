---
globs: "**/*.py"
---

# Python 代码规则

- 使用 Python 3.11+ 语法
- 类型注解必须写（函数签名 + 返回值）
- 使用 async/await 处理 IO 密集操作（API 调用、文件读写）
- 用 pydantic 做数据校验
- 日志用 structlog（结构化日志，方便后续分析）
- 测试用 pytest + pytest-asyncio
