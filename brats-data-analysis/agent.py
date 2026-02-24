#!/usr/bin/env python3
"""
agent.py — BraTS 2024 MRI Analysis Agent

A single CLI entry point that dispatches to all analysis tools.

Usage:
  python agent.py --data_dir ~/workspace/brats/training_data1_v2 --tool explore
  python agent.py --data_dir ... --tool visualize --case BraTS-GLI-00005-100
  python agent.py --data_dir ... --tool grid --n 9 --modality t1c
  python agent.py --data_dir ... --tool stats
  python agent.py --data_dir ... --tool intensity --sample 50
  python agent.py --data_dir ... --tool longitudinal
  python agent.py --data_dir ... --tool qc
"""

import sys
import argparse
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

console = Console()

TOOLS = ("explore", "visualize", "grid", "stats", "intensity", "longitudinal", "qc")
MODALITIES = ("t1n", "t1c", "t2f", "t2w")

TOOL_DESCRIPTIONS = {
    "explore":      "Scan all cases, build dataset_summary.csv, print stats table",
    "visualize":    "Multi-planar 4-modality viz with seg overlay for one case",
    "grid":         "N-case axial-slice overview grid (random sample)",
    "stats":        "Statistical analysis + 6-panel overview figure",
    "intensity":    "Intensity distribution analysis across modalities & regions",
    "longitudinal": "Longitudinal patient tracking + volume trajectory plots",
    "qc":           "Quality-control sweep: shapes, spacings, empty masks, outliers",
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="agent.py",
        description="BraTS 2024 MRI Analysis Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join(
            f"  {t:12s}  {d}" for t, d in TOOL_DESCRIPTIONS.items()
        ),
    )
    p.add_argument(
        "--data_dir", required=True,
        help="Path to the BraTS 2024 training data directory (training_data1_v2)",
    )
    p.add_argument(
        "--tool", choices=TOOLS, required=True,
        help="Which analysis tool to run",
    )
    p.add_argument(
        "--case", default=None,
        help="Case ID for --tool visualize (e.g. BraTS-GLI-00005-100)",
    )
    p.add_argument(
        "--n", type=int, default=6,
        help="Number of cases for --tool grid (default: 6)",
    )
    p.add_argument(
        "--output_dir", default="./output",
        help="Directory for output files (default: ./output)",
    )
    p.add_argument(
        "--modality", choices=MODALITIES, default="t1c",
        help="MRI modality for grid / visualize (default: t1c)",
    )
    p.add_argument(
        "--sample", type=int, default=50,
        help="Number of cases to sample for --tool intensity (default: 50)",
    )
    return p


def print_header(tool: str, data_dir: Path, output_dir: Path):
    text = Text()
    text.append("BraTS 2024 MRI Analysis Agent\n", style="bold white")
    text.append(f"Tool:       ", style="dim")
    text.append(f"{tool}\n", style="bold cyan")
    text.append(f"Data dir:   ", style="dim")
    text.append(f"{data_dir}\n", style="white")
    text.append(f"Output dir: ", style="dim")
    text.append(f"{output_dir}", style="white")
    console.print(Panel(text, border_style="blue", padding=(0, 2)))


def main():
    parser = build_parser()
    args = parser.parse_args()

    data_dir = Path(args.data_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    # Validate data_dir
    if not data_dir.exists():
        console.print(f"[bold red]Error:[/] data_dir does not exist: {data_dir}")
        sys.exit(1)
    if not data_dir.is_dir():
        console.print(f"[bold red]Error:[/] data_dir is not a directory: {data_dir}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    print_header(args.tool, data_dir, output_dir)

    # ── Dispatch ──────────────────────────────────────────────────────────
    if args.tool == "explore":
        from tools.explore import explore_dataset
        explore_dataset(str(data_dir), str(output_dir))

    elif args.tool == "visualize":
        if not args.case:
            console.print("[bold red]Error:[/] --case is required for --tool visualize")
            console.print("Example: --case BraTS-GLI-00005-100")
            sys.exit(1)
        from tools.visualize import visualize_case
        visualize_case(str(data_dir), args.case, str(output_dir), args.modality)

    elif args.tool == "grid":
        from tools.grid import visualize_grid
        visualize_grid(str(data_dir), str(output_dir), args.n, args.modality)

    elif args.tool == "stats":
        from tools.stats import analyze_stats
        analyze_stats(str(data_dir), str(output_dir))

    elif args.tool == "intensity":
        from tools.intensity import analyze_intensity
        analyze_intensity(str(data_dir), str(output_dir), args.sample)

    elif args.tool == "longitudinal":
        from tools.longitudinal import analyze_longitudinal
        analyze_longitudinal(str(data_dir), str(output_dir))

    elif args.tool == "qc":
        from tools.qc import run_qc
        run_qc(str(data_dir), str(output_dir))


if __name__ == "__main__":
    main()
