#!/usr/bin/env python
"""Dispatcher for scripts/supplementary/* — one entry point, no logic."""
import argparse, subprocess, sys
from pathlib import Path

HERE = Path(__file__).parent
SUBCMDS = {
    "lstm-bilstm":     HERE / "supplementary" / "lstm_bilstm.py",
    "additional":      HERE / "supplementary" / "analysis.py",
    "conditional-abl": HERE / "supplementary" / "conditional_abl.py",
    "grouped-abl":     HERE / "supplementary" / "grouped_abl.py",
    "spike-eval":      HERE / "supplementary" / "spike_eval.py",
}

def main():
    p = argparse.ArgumentParser(
        description="Run a supplementary analysis. "
                    "All subcommands are standalone scripts under "
                    "scripts/supplementary/.")
    p.add_argument("subcmd", choices=list(SUBCMDS) + ["all"])
    args, rest = p.parse_known_args()

    targets = SUBCMDS.values() if args.subcmd == "all" else [SUBCMDS[args.subcmd]]
    for path in targets:
        print(f"\n===== {path.name} =====", flush=True)
        r = subprocess.run([sys.executable, str(path), *rest])
        if r.returncode != 0:
            sys.exit(r.returncode)

if __name__ == "__main__":
    main()
