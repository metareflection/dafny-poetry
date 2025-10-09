
import argparse, pathlib, sys, time
from .poetry_alg import run_poetry

def main(argv=None):
    ap = argparse.ArgumentParser(prog="dafny-poetry")
    ap.add_argument("--file", required=True, help="Path to the .dfy file")
    ap.add_argument("--out-dir", default="poetry_out", help="Directory for intermediate outputs")
    ap.add_argument("--max-depth", type=int, default=3)
    ap.add_argument("--max-branches", type=int, default=2)
    ap.add_argument("--global-timeout", type=int, default=600, help="Seconds for overall loop budget")
    ap.add_argument("--use-llm", action="store_true", help="Enable LLM-based local refinement")
    ap.add_argument("--llm-tries", type=int, default=2, help="How many LLM proposals per focus goal")
    ap.add_argument("-v","--verbose", action="store_true")
    args = ap.parse_args(argv)

    src = pathlib.Path(args.file)
    if not src.exists():
        sys.exit(f"Input not found: {src}")

    args.out_dir = pathlib.Path(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    status, final_path = run_poetry(src, args)
    if status == 0:
        print(f"[SUCCESS] All obligations solved. Final: {final_path}")
    elif status == 2:
        print(f"[PARTIAL] Remaining admits. Sketch at: {final_path}")
    else:
        print(f"[GAVE UP] Final attempt at: {final_path}")
    return status
