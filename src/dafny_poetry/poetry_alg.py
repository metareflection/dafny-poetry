
import pathlib, shutil, time, sys, re
from dataclasses import dataclass
from typing import Optional, List, Tuple
from .dafny_io import run_dafny_admitter, run_sketcher, count_admits, collect_first_admit_context, write_version, replace_method_body
from .utils import extract_method_body_region
from . import llm_agent

@dataclass
class State:
    path: pathlib.Path     # path to current dfy
    admits: int
    label: str

def _read(p: pathlib.Path) -> str:
    return p.read_text(encoding="utf-8", errors="ignore")

def _method_body_text(src: str, method: str) -> str:
    start_line, end_line, bl, br = extract_method_body_region(src, None, method_name=method)
    if bl is None:
        return ""
    lines = src.splitlines()
    # body content between '{' line and closing '}' line (exclusive of closing brace)
    return "\n".join(lines[bl+1:br])

def run_poetry(src_path: pathlib.Path, args) -> Tuple[int, pathlib.Path]:
    start = time.time()
    # 0) seed: run admitter to get verifiable sketch
    patched = run_dafny_admitter(src_path, mode="admit", only_failing=True, timeout=min(args.global_timeout, 300))
    base_admits = count_admits(patched)
    curr = State(path=patched, admits=base_admits, label="seed")
    best_path = patched

    if args.verbose:
        print(f"[seed] admits={curr.admits} @ {patched}")

    depth = 0
    visited = set()
    while depth <= args.max_depth and (time.time() - start) < args.global_timeout:
        if curr.admits == 0:
            return 0, curr.path  # success

        # Pick a focus goal (first Admit occurrence)
        ctx = collect_first_admit_context(curr.path)
        if not ctx or not ctx.get("method"):
            # No explicit admits found; try a quick error pass; stop.
            return 2, curr.path

        method = ctx["method"]
        if args.verbose:
            print(f"[depth {depth}] focusing method: {method} @ line {ctx['line']} tag={ctx['tag']}")

        # 1) ACTION: Induction search via sketcher (symbolic)
        before_txt = _read(curr.path)
        out1 = run_sketcher(curr.path, "induction_search", method=method, timeout=120)
        after1 = run_dafny_admitter(curr.path, mode="admit", only_failing=True, timeout=180)
        a1 = count_admits(after1)
        if args.verbose:
            print(f"  [induction_search] admits: {a1} (was {curr.admits})")
        if a1 < curr.admits:
            depth += 1
            curr = State(path=after1, admits=a1, label=f"induction_d{depth}")
            best_path = after1
            continue  # recurse on the improved state

        # 2) ACTION: LLM refine (optional)
        if args.use_llm:
            # Prepare context
            errors = run_sketcher(curr.path, "errors_warnings", method=None, timeout=60)
            # Collect admits within the same method (approx by slicing region)
            src_text = _read(curr.path)
            body_text = _method_body_text(src_text, method)
            admits_snippets = []
            if body_text:
                for line in body_text.splitlines():
                    if "Admit(" in line:
                        admits_snippets.append(line.strip())
            new_body = None
            try:
                new_body = llm_agent.propose_new_body(method, errors, "\n".join(admits_snippets), body_text, tries=max(1,args.llm_tries))
            except Exception as e:
                if args.verbose:
                    print(f"  [llm] skipped: {e}")
            if new_body:
                try:
                    replaced = replace_method_body(src_text, method, new_body)
                    # Write candidate and re-admit
                    cand = write_version(args.out_dir, curr.path, f"llm_{depth}", replaced)
                    _ = run_sketcher(cand, "errors_warnings", method=None, timeout=60)  # benign check
                    cand2 = run_dafny_admitter(cand, mode="admit", only_failing=True, timeout=180)
                    a2 = count_admits(cand2)
                    if args.verbose:
                        print(f"  [llm_refine] admits: {a2} (was {curr.admits})")
                    if a2 < curr.admits:
                        depth += 1
                        curr = State(path=cand2, admits=a2, label=f"llm_d{depth}")
                        best_path = cand2
                        continue
                except Exception as e:
                    if args.verbose:
                        print(f"  [llm] candidate rejected: {e}")

        # 3) No progress on this goal; try next goal or terminate if branching cap hit
        sig = (curr.path.read_text()[:2000], curr.admits)
        if sig in visited:
            break
        visited.add(sig)

        # As a simple branching heuristic, move admit cursor by replacing the first Admit with a comment to focus on the next one
        text = _read(curr.path)
        new_text = re.sub(r'Admit\s*\(', '// Admit(', text, count=1)
        if new_text != text:
            cand = write_version(args.out_dir, curr.path, f"shift_{depth}", new_text)
            cand2 = run_dafny_admitter(cand, mode="admit", only_failing=True, timeout=180)
            a3 = count_admits(cand2)
            if args.verbose:
                print(f"  [shift-focus] admits: {a3} (was {curr.admits})")
            if a3 <= curr.admits and cand2 != curr.path:
                curr = State(path=cand2, admits=a3, label=f"shift_d{depth}")
                best_path = cand2
                continue
        # Give up this depth
        break

    # Finished loop
    return (2 if curr.admits>0 else 0), curr.path
