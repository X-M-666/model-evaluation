# -*- coding: utf-8 -*-
"""清洗历史评测记录中的敏感字段（如 API Key），使旧报告不再保留明文 Key。

用法（在 webapp/ 目录下）：
    python -m scripts.scrub_history            # 实际重写
    python -m scripts.scrub_history --dry-run  # 仅预览将处理的文件
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.security import redact_sensitive  # noqa: E402
from backend.storage import BASE_DIR  # noqa: E402


def _scan_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*.json") if p.is_file()) if root.exists() else []


def main() -> int:
    parser = argparse.ArgumentParser(description="清洗历史评测记录中的敏感字段")
    parser.add_argument("--dry-run", action="store_true", help="仅预览，不实际重写文件")
    args = parser.parse_args()

    targets = _scan_files(BASE_DIR)
    if not targets:
        print(f"未发现历史记录（目录: {BASE_DIR}）")
        return 0

    changed = 0
    for p in targets:
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        safe = redact_sensitive(data)
        if safe == data:
            continue
        changed += 1
        print(f"{'[dry-run]' if args.dry_run else '[rewrite]'} {p}")
        if not args.dry_run:
            p.write_text(json.dumps(safe, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"共处理 {changed}/{len(targets)} 个文件（{'仅预览' if args.dry_run else '已重写'}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
