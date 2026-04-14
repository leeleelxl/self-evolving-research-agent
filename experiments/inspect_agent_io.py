"""
Agent IO 审查工具 — 读取 pipeline 运行结果，逐 Agent 打印实际 IO

用法:
  python experiments/inspect_agent_io.py <results.json>
  python experiments/inspect_agent_io.py <results.json> --agent Planner
  python experiments/inspect_agent_io.py <results.json> --iteration 1
  python experiments/inspect_agent_io.py <results.json> --diff-queries

这是 Agent 项目的核心审查工具。做 Agent 项目时：
- 聚合数字（分数、计数）可能说谎
- 真正判断 Agent 是否按预期行动，必须看实际 IO 文本

关键用途:
- 验证 Planner 自进化是真 diverge 还是换皮
- 验证 Critic improvement_suggestions 是具体建议还是套话
- 验证 Reader core_contribution 是真提炼还是 paraphrase abstract
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load_result(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def get_traces(data: dict) -> list[dict]:
    """从实验 JSON 找 agent_traces，支持多种嵌套结构"""
    # 直接在顶层
    if "agent_traces" in data:
        return data["agent_traces"]
    # 嵌套在 result 字段里
    if "result" in data and isinstance(data["result"], dict) and "agent_traces" in data["result"]:
        return data["result"]["agent_traces"]
    # 嵌套在 pipeline 字段里
    if "pipeline" in data and isinstance(data["pipeline"], dict) and "agent_traces" in data["pipeline"]:
        return data["pipeline"]["agent_traces"]
    return []


def print_planner(trace: dict) -> None:
    print(f"\n{'='*70}\n[Planner] Iteration {trace['iteration']}")
    print(f"Input: {trace['input_summary']}")
    out = trace["output"]
    print(f"\nSub-questions ({len(out.get('sub_questions', []))}):")
    for i, sq in enumerate(out.get("sub_questions", []), 1):
        print(f"  {i}. {sq}")
    queries = out.get("search_strategy", {}).get("queries", [])
    print(f"\nQueries ({len(queries)}):")
    for i, q in enumerate(queries, 1):
        year_range = ""
        if q.get("year_min") or q.get("year_max"):
            year_range = f" [{q.get('year_min','?')}-{q.get('year_max','?')}]"
        print(f"  {i}. {q['query']}{year_range}")
    focus = out.get("search_strategy", {}).get("focus_areas", [])
    if focus:
        print(f"\nFocus areas: {', '.join(focus)}")
    exclude = out.get("search_strategy", {}).get("exclude_terms", [])
    if exclude:
        print(f"Exclude terms: {', '.join(exclude)}")


def print_retriever(trace: dict) -> None:
    print(f"\n{'='*70}\n[Retriever] Iteration {trace['iteration']}")
    print(f"Input: {trace['input_summary']}")
    out = trace["output"]
    papers = out.get("papers", [])
    print(f"Total retrieved: {out.get('total', len(papers))}")
    print(f"\nTop 10 by citations:")
    sorted_papers = sorted(papers, key=lambda p: p.get("citations", 0), reverse=True)[:10]
    for p in sorted_papers:
        pdf = "📄" if p.get("has_pdf_url") else "  "
        print(f"  {pdf} [{p.get('citations', 0):>4}cit {p.get('year', '?')}] {p['title'][:80]}")


def print_reader(trace: dict) -> None:
    print(f"\n{'='*70}\n[Reader] Iteration {trace['iteration']}")
    print(f"Input: {trace['input_summary']}")
    out = trace["output"]
    notes = out.get("notes", [])
    print(f"Kept: {out.get('kept', len(notes))}, filtered: {out.get('filtered_out', 0)}")
    print(f"\nAll notes (relevance_score desc):")
    sorted_notes = sorted(notes, key=lambda n: n.get("relevance_score", 0), reverse=True)
    for i, n in enumerate(sorted_notes, 1):
        score = n.get("relevance_score", 0)
        print(f"\n  [{i}] ({score:.2f}) {n['title'][:80]}")
        print(f"      Contribution: {n.get('core_contribution', '')[:200]}")
        if n.get("methodology"):
            print(f"      Method: {n['methodology'][:150]}")
        if n.get("key_findings"):
            findings = n["key_findings"][:2]
            for f in findings:
                print(f"      • {f[:150]}")
        print(f"      Relevance: {n.get('relevance_reason', '')[:150]}")


def print_writer(trace: dict) -> None:
    print(f"\n{'='*70}\n[Writer] Iteration {trace['iteration']}")
    print(f"Input: {trace['input_summary']}")
    out = trace["output"]
    print(f"\nTitle: {out.get('title', 'N/A')}")
    print(f"Abstract: {out.get('abstract', '')[:300]}...")
    sections = out.get("sections", [])
    print(f"\nSections ({len(sections)}):")
    for i, sec in enumerate(sections, 1):
        n_cited = len(sec.get("cited_papers", []))
        content_preview = sec.get("content", "")[:150].replace("\n", " ")
        print(f"  {i}. [{n_cited} refs] {sec['section_title']}")
        print(f"     {content_preview}...")
    refs = out.get("references", [])
    print(f"\nTotal references: {len(refs)}")


def print_critic(trace: dict) -> None:
    print(f"\n{'='*70}\n[Critic] Iteration {trace['iteration']}")
    print(f"Input: {trace['input_summary']}")
    out = trace["output"]
    scores = out.get("scores", {})
    print(f"\nScores: coverage={scores.get('coverage')} depth={scores.get('depth')} "
          f"coherence={scores.get('coherence')} accuracy={scores.get('accuracy')}")
    if scores.get("cross_model_spread"):
        print(f"Cross-model spread: {scores['cross_model_spread']}")
    print(f"Is satisfactory: {out.get('is_satisfactory')}")

    print(f"\nMissing aspects ({len(out.get('missing_aspects', []))}):")
    for i, asp in enumerate(out.get("missing_aspects", []), 1):
        print(f"  {i}. {asp}")

    print(f"\nImprovement suggestions ({len(out.get('improvement_suggestions', []))}):")
    for i, s in enumerate(out.get("improvement_suggestions", []), 1):
        print(f"  {i}. {s}")

    print(f"\nNew queries proposed ({len(out.get('new_queries', []))}):")
    for i, q in enumerate(out.get("new_queries", []), 1):
        print(f"  {i}. {q}")


PRINTERS = {
    "Planner": print_planner,
    "Retriever": print_retriever,
    "Reader": print_reader,
    "Writer": print_writer,
    "Critic": print_critic,
}


def diff_queries(traces: list[dict]) -> None:
    """专用视图: 对比每轮 Planner 生成的 queries，看是否真 diverge"""
    planner_traces = [t for t in traces if t["agent_name"] == "Planner"]
    if len(planner_traces) < 2:
        print("需要至少 2 轮 Planner trace 才能对比 diverge")
        return

    print(f"\n{'='*70}")
    print(f"Planner Queries Diverge Analysis ({len(planner_traces)} iterations)")
    print(f"{'='*70}\n")

    prev_queries: set[str] = set()
    for t in planner_traces:
        iter_num = t["iteration"]
        queries = [q["query"] for q in t["output"]["search_strategy"]["queries"]]
        current_set = set(queries)
        new_queries = current_set - prev_queries
        repeated = current_set & prev_queries

        print(f"Iteration {iter_num}: {len(queries)} queries total")
        print(f"  新增: {len(new_queries)} | 重复上轮: {len(repeated)}")
        print(f"  新增的 queries:")
        for q in sorted(new_queries):
            print(f"    + {q}")
        if repeated:
            print(f"  和上轮重复的 queries:")
            for q in sorted(repeated):
                print(f"    = {q}")
        print()
        prev_queries = current_set


def main() -> None:
    parser = argparse.ArgumentParser(description="Agent IO 审查工具")
    parser.add_argument("json_path", type=Path, help="实验 JSON 结果文件路径")
    parser.add_argument("--agent", choices=list(PRINTERS.keys()), help="只看某个 Agent")
    parser.add_argument("--iteration", type=int, help="只看某轮 iteration")
    parser.add_argument("--diff-queries", action="store_true",
                        help="Planner queries diverge 对比视图")
    args = parser.parse_args()

    if not args.json_path.exists():
        print(f"Error: {args.json_path} not found", file=sys.stderr)
        sys.exit(1)

    data = load_result(args.json_path)
    traces = get_traces(data)

    if not traces:
        print(f"⚠️  {args.json_path} 中没有 agent_traces 字段。")
        print("  可能是旧版本实验数据（P0 前跑的）。请用新实验脚本重跑。")
        sys.exit(1)

    print(f"Loaded {len(traces)} agent traces from {args.json_path}")
    print(f"Question: {data.get('question', 'N/A')}")
    print(f"Timestamp: {data.get('timestamp', 'N/A')}")

    if args.diff_queries:
        diff_queries(traces)
        return

    for t in traces:
        if args.agent and t["agent_name"] != args.agent:
            continue
        if args.iteration is not None and t["iteration"] != args.iteration:
            continue
        PRINTERS[t["agent_name"]](t)


if __name__ == "__main__":
    main()
