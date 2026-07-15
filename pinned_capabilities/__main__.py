"""Small administrative CLI; experiment runners are added gate by gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import ProtocolConfig
from .manifest import freeze_manifest


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m pinned_capabilities")
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze = subparsers.add_parser("freeze", help="freeze the default protocol manifest")
    freeze.add_argument("path", type=Path)
    freeze.add_argument("--repo", type=Path, default=Path.cwd())
    show = subparsers.add_parser("show-config", help="print the default protocol configuration")
    args = parser.parse_args()
    config = ProtocolConfig()
    if args.command == "freeze":
        manifest = freeze_manifest(config, args.path, repo=args.repo)
        print(json.dumps(manifest, indent=2, sort_keys=True))
    else:
        print(json.dumps(config.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
