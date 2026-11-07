"""本地协议工具，不依赖 Bark 配置，不发出交易订单。"""

import argparse
import json
import sqlite3
from pathlib import Path

from . import ResearchProtocol, ResearchRegistry


def main() -> None:
    parser = argparse.ArgumentParser(description="冻结收益研究协议与上线判据")
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate", help="校验协议，列出仅允许探索的原因")
    validate.add_argument("source", type=Path)
    freeze = commands.add_parser("freeze", help="追加冻结版本，输出协议标识")
    freeze.add_argument("source", type=Path)
    freeze.add_argument("--registry", type=Path, required=True)
    history = commands.add_parser("history", help="读取研究的全部协议版本")
    history.add_argument("study_id")
    history.add_argument("--registry", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "history":
            versions = ResearchRegistry(args.registry).protocols(args.study_id)
            output = [version.model_dump(mode="json") for version in versions]
        else:
            protocol = ResearchProtocol.model_validate_json(args.source.read_text(encoding="utf-8"))
            if args.command == "freeze":
                output = ResearchRegistry(args.registry).freeze(protocol).model_dump(mode="json")
            else:
                reasons = protocol.exploration_reasons()
                output = {
                    "status": "exploration_only" if reasons else "ready_for_confirmation",
                    "reasons": reasons,
                    "protocol": protocol.model_dump(mode="json"),
                }
        print(json.dumps(output, ensure_ascii=False, indent=2))
    except (ValueError, OSError, sqlite3.DatabaseError) as error:
        parser.exit(1, f"协议操作失败：{error}\n")


if __name__ == "__main__":
    main()
