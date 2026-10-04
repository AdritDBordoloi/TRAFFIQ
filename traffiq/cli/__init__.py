"""
TRAFFIQ Command Line Interface (CLI) submodule.
Provides unified launcher, argument parsing, and execution dispatchers
for the 4 operational modes: diagnostics, detection, tracking, enforcement.
"""

from traffiq.cli.app import (
    build_parser,
    main,
    parse_args,
    run_cli,
    run_detection,
    run_diagnostics,
    run_enforcement,
    run_tracking,
)

__all__ = [
    "main",
    "build_parser",
    "parse_args",
    "run_cli",
    "run_diagnostics",
    "run_detection",
    "run_tracking",
    "run_enforcement",
]
