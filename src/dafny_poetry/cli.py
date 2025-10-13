
import argparse, pathlib, sys, time
from .poetry_recursive import run_poetry, PoetryConfig

def main(argv=None):
    ap = argparse.ArgumentParser(prog="dafny-poetry", 
        description="POETRY: Proof repair via recursive best-first search")
    ap.add_argument("--file", required=True, help="Path to the .dfy file")
    ap.add_argument("--out-dir", default="poetry_out", help="Directory for intermediate outputs")
    ap.add_argument("--max-depth", type=int, default=10, help="Maximum recursion depth")
    ap.add_argument("--max-branches", type=int, default=2, help="Number of LLM samples per expansion")
    ap.add_argument("--global-timeout", type=int, default=600, help="Seconds for overall search")
    ap.add_argument("--local-timeout", type=int, default=120, help="Seconds per recursion level (depth > 1)")
    ap.add_argument("--use-llm", action="store_true", help="Enable LLM-based refinement")
    ap.add_argument("--llm-tries", type=int, default=2, help="LLM retries per sample")
    ap.add_argument("-v","--verbose", action="store_true", help="Verbose output")
    ap.add_argument("--legacy", action="store_true", help="Use legacy greedy algorithm (for comparison)")
    args = ap.parse_args(argv)

    src = pathlib.Path(args.file)
    if not src.exists():
        sys.exit(f"Input not found: {src}")

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Use legacy algorithm if requested
    if args.legacy:
        from .poetry_alg import run_poetry as run_poetry_legacy
        if args.verbose:
            print("[CLI] Using legacy greedy algorithm")
        # Convert args to old format
        class LegacyArgs:
            pass
        legacy_args = LegacyArgs()
        legacy_args.out_dir = out_dir
        legacy_args.max_depth = args.max_depth
        legacy_args.use_llm = args.use_llm
        legacy_args.llm_tries = args.llm_tries
        legacy_args.global_timeout = args.global_timeout
        legacy_args.verbose = args.verbose
        status, final_path = run_poetry_legacy(src, legacy_args)
    else:
        # Use new recursive POETRY algorithm
        if args.verbose:
            print("[CLI] Using full POETRY recursive BFS algorithm")
        config = PoetryConfig(
            max_depth=args.max_depth,
            max_branches=args.max_branches,
            global_timeout=args.global_timeout,
            local_timeout=args.local_timeout,
            use_llm=args.use_llm,
            llm_tries=args.llm_tries,
            out_dir=out_dir,
            verbose=args.verbose
        )
        status, final_path = run_poetry(src, config)

    if status == 0:
        print(f"[SUCCESS] All obligations solved. Final: {final_path}")
    elif status == 2:
        print(f"[PARTIAL] Remaining admits. Sketch at: {final_path}")
    else:
        print(f"[GAVE UP] Final attempt at: {final_path}")
    return status
