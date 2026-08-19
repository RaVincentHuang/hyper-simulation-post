"""Question-answering boundary for invoking and inspecting HyperMatch.

The algorithms themselves live in :mod:`hyper_simulation.component` as a
library.  This module owns only compatibility-flag parsing and immutable
contract inspection.  Programmatic question-answering code imports solver interfaces
directly from :mod:`hyper_simulation.component.hyper_simulation`.

Model weights are optional and are not imported by this module.  The CLI only
prints the immutable selected contract.
"""

from __future__ import annotations

import argparse
import json
from typing import Sequence

from hyper_simulation.component.config import config_for_mode, config_summary


def build_parser() -> argparse.ArgumentParser:
    """Build the QA-owned parser for the supported compatibility flags."""

    parser = argparse.ArgumentParser(
        description="Select the HyperMatch library contract used by QA."
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--fancy",
        dest="fancy",
        action="store_true",
        help="select the fancy HCCalc and D-match profile",
    )
    mode.add_argument(
        "--no-fancy",
        dest="fancy",
        action="store_false",
        help="select the standard profile (default)",
    )
    parser.set_defaults(fancy=False)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Print the immutable component contract requested by the QA layer."""

    args = build_parser().parse_args(argv)
    selected = config_for_mode(fancy=bool(args.fancy))
    print(json.dumps(config_summary(selected), indent=2, sort_keys=True))
    return 0


__all__ = ["build_parser", "main"]


if __name__ == "__main__":  # pragma: no cover - covered by subprocess smoke tests.
    raise SystemExit(main())
