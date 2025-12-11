#!/usr/bin/env python3
import json
from json import JSONDecodeError
from pathlib import Path

# 源文件和输出文件路径
src = Path("/mnt/data/welkinni/table_det/report_inspect_summaries.json")
pretty_out = src.with_suffix(".pretty.json")
summary_out = src.with_suffix(".summary.txt")

def load_data():
    raw = src.read_text(encoding="utf-8")
    try:
        return json.loads(raw)
    except JSONDecodeError:
        # 退回按行解析（常见于 NDJSON / JSON Lines）
        return [json.loads(line) for line in raw.splitlines() if line.strip()]


def main():
    data = load_data()

    # 写出格式化后的 JSON
    pretty_out.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    # 如果是列表，生成更易读的概要文本
    if isinstance(data, list):
        lines = []
        for i, item in enumerate(data, 1):
            if isinstance(item, dict):
                # 优先使用现有字段，否则回退 reportId / 行号
                title = (
                    item.get("title")
                    or item.get("name")
                    or item.get("id")
                    or item.get("reportId")
                    or f"item_{i}"
                )

                # 尝试用 summary/description，否则从检查项拼一个简述
                desc = item.get("summary") or item.get("description")
                if desc is None:
                    inspect_summaries = item.get("inspectSummaries")
                    if isinstance(inspect_summaries, list) and inspect_summaries:
                        first = inspect_summaries[0]
                        subject = first.get("subject")
                        inspect_items = first.get("inspectItems")
                        head_items = []
                        if isinstance(inspect_items, list):
                            for it in inspect_items[:3]:
                                if isinstance(it, dict):
                                    name = it.get("name")
                                    result = it.get("result")
                                    head_items.append(f"{name}={result}")
                        parts = []
                        if subject:
                            parts.append(f"subject={subject}")
                        if head_items:
                            parts.append("items: " + "; ".join(head_items))
                        desc = " | ".join(parts) if parts else None

                status = item.get("status")
                lines.append(f"[{i}] title={title} | status={status} | desc={desc}")
            else:
                lines.append(f"[{i}] {item!r}")
        summary_out.write_text("\n".join(lines), encoding="utf-8")
        print(f"写出了概要: {summary_out}")
    else:
        print("顶层不是列表，未生成概要文本。")

    print(f"格式化后的 JSON: {pretty_out}")

if __name__ == "__main__":
    main()