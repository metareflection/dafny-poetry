"""
Pareto Frontier Prover: Multi-objective proof search for Dafny.

This module implements a Pareto-optimal search strategy that maintains a frontier
of non-dominated states based on two objectives:
1. Number of admits (lower is better)
2. Number of verification errors (lower is better)

Key features:
- Allows initial admit increase (empty body → structured proof with admits)
- Explores multiple solution paths in parallel
- Naturally handles multi-step proofs (structure then fill)
- Oracle and sketcher integration

Usage:
    python -m dafny_poetry.pareto_prover --file path/to/proof.dfy --out-dir pareto_out

Or programmatically:
    from dafny_poetry.pareto_prover import run_pareto, ParetoConfig
    result = run_pareto(Path("proof.dfy"), ParetoConfig())
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import time
from dataclasses import dataclass, asdict
from typing import Callable, Dict, List, Optional, Set

from .dafny_io import (
    run_dafny_admitter,
    run_sketcher,
    quick_verify,
    VerificationResult,
    count_admits,
    collect_first_admit_context,
    write_version,
    replace_method_body,
)
from .utils import extract_method_body_text

# LLM is optional
try:
    from . import llm_agent as _llm
except Exception:
    _llm = None


# ------------------------------- Config ------------------------------------

@dataclass
class ParetoConfig:
    # Limits
    max_iterations: int = 20
    max_frontier_size: int = 20
    global_timeout: int = 600       # seconds
    verify_timeout: int = 20
    admitter_timeout: int = 60
    sketcher_timeout: int = 60

    # Behavior
    use_llm: bool = False
    llm_tries: int = 2
    verbose: bool = False
    oracle: Optional[Callable[[str], list[str]]] = None
    sketch_oracle: Optional[Callable[[str], list[str]]] = None
    max_oracle_guesses: int = 3

    # Output
    out_dir: Optional[pathlib.Path] = None


@dataclass
class ParetoState:
    path: pathlib.Path
    admits: int
    errors: int
    iteration: int
    parent_id: Optional[str] = None
    action: Optional[str] = None

    def id(self) -> str:
        """Hash-based ID for deduplication."""
        try:
            data = self.path.read_bytes()
        except Exception:
            data = b""
        return hashlib.sha1(data).hexdigest()

    def dominates(self, other: ParetoState) -> bool:
        """True if self strictly dominates other (better on at least one, not worse on any)."""
        better_admits = self.admits < other.admits
        better_errors = self.errors < other.errors
        not_worse_admits = self.admits <= other.admits
        not_worse_errors = self.errors <= other.errors

        return (better_admits or better_errors) and not_worse_admits and not_worse_errors


@dataclass
class ParetoResult:
    solved: bool
    final_path: pathlib.Path
    iteration: int
    states_explored: int
    best_partial_path: pathlib.Path
    best_partial_admits: int
    best_partial_errors: int
    initial_admits: int
    initial_errors: int
    journal_path: pathlib.Path


# --------------------------- Helper functions -------------------------------

def _read_text(p: pathlib.Path) -> str:
    return p.read_text(encoding="utf-8", errors="ignore")


def _write_version(out_dir: pathlib.Path, base: pathlib.Path, label: str, text: str) -> pathlib.Path:
    return write_version(out_dir, base, label, text)


def _apply_admit_patch(src_text: str, line_no: int, replacement: str) -> str:
    """Replace exactly one line with the patch, preserving indentation."""
    lines = src_text.splitlines()
    if line_no <= 0 or line_no > len(lines):
        return src_text
    idx = line_no - 1
    indent = re.match(r"\s*", lines[idx]).group(0)
    rep = replacement.strip()
    # Strip code fences
    rep = re.sub(r'^\s*```[a-zA-Z]*\s*', '', rep)
    rep = re.sub(r'\s*```$', '', rep)
    rep_lines = [(indent + ln) if ln.strip() else ln for ln in rep.splitlines()]
    return "\n".join(lines[:idx] + rep_lines + lines[idx+1:])


def _first_admit_info(dfy_path: pathlib.Path) -> Optional[Dict]:
    """Collect info around the first Admit occurrence."""
    info = collect_first_admit_context(dfy_path)
    if not info:
        return None
    src = _read_text(dfy_path)
    lines = src.splitlines()
    line_no = info["line"]
    lo = max(0, line_no - 6)
    hi = min(len(lines), line_no + 5)
    before = "\n".join(lines[lo:line_no-1])
    after = "\n".join(lines[line_no:hi])
    info.update({
        "target_line_text": lines[line_no-1] if 0 <= line_no-1 < len(lines) else "",
        "local_before": before,
        "local_after": after,
        "file_source": src,
        "method_body": extract_method_body_text(src, info.get("method")) or "",
    })
    return info


def pareto_filter(states: List[ParetoState], max_size: int) -> List[ParetoState]:
    """Keep only non-dominated states, limited to max_size."""
    if not states:
        return []

    # Deduplicate by file hash
    seen: Dict[str, ParetoState] = {}
    for state in states:
        sid = state.id()
        if sid not in seen or state.iteration < seen[sid].iteration:
            seen[sid] = state

    unique_states = list(seen.values())

    # Filter to Pareto frontier
    frontier = []
    for state in unique_states:
        dominated = False
        for other in unique_states:
            if other.dominates(state):
                dominated = True
                break
        if not dominated:
            frontier.append(state)

    # If frontier is too large, keep best by (admits, errors) lexicographic order
    if len(frontier) > max_size:
        frontier.sort(key=lambda s: (s.admits, s.errors))
        frontier = frontier[:max_size]

    return frontier


# ------------------------------ Core Search ---------------------------------

def run_pareto(src_path: pathlib.Path, cfg: Optional[ParetoConfig] = None) -> ParetoResult:
    cfg = cfg or ParetoConfig()
    src_path = pathlib.Path(src_path)

    # Output directory
    out_dir = cfg.out_dir or src_path.parent / f"pareto_out_{int(time.time())}"
    out_dir.mkdir(parents=True, exist_ok=True)

    journal_path = out_dir / "pareto_journal.jsonl"
    jh = journal_path.open("w", encoding="utf-8")

    def log(event: Dict):
        event = dict(event)
        event.setdefault("ts", time.time())
        jh.write(json.dumps(event, ensure_ascii=False) + "\n")
        jh.flush()

    start = time.time()

    # Initial state
    vr = quick_verify(src_path, timeout=cfg.verify_timeout)
    init_admits = count_admits(src_path)
    init_errors = vr.errors
    init_state = ParetoState(
        path=src_path,
        admits=init_admits,
        errors=init_errors,
        iteration=0,
        parent_id=None,
        action=None
    )

    frontier = [init_state]
    best_partial = init_state
    explored = 0

    log({"event": "start", "initial_admits": init_admits, "initial_errors": init_errors})

    for iteration in range(1, cfg.max_iterations + 1):
        if (time.time() - start) >= cfg.global_timeout:
            if cfg.verbose:
                print(f"[Pareto] Timeout after {iteration-1} iterations")
            break

        if cfg.verbose:
            print(f"[Pareto] Iteration {iteration}, frontier size: {len(frontier)}")

        new_states = []

        for state in frontier:
            # Check for solution
            if state.admits == 0 and state.errors == 0:
                if cfg.verbose:
                    print(f"[Pareto] SOLVED at iteration {iteration}!")
                state_dict = asdict(state)
                state_dict['path'] = str(state_dict['path'])  # Convert Path to string for JSON
                log({"event": "goal", "iteration": iteration, "state": state_dict})
                jh.close()
                return ParetoResult(
                    solved=True,
                    final_path=state.path,
                    iteration=iteration,
                    states_explored=explored,
                    best_partial_path=state.path,
                    best_partial_admits=state.admits,
                    best_partial_errors=state.errors,
                    initial_admits=init_admits,
                    initial_errors=init_errors,
                    journal_path=journal_path,
                )

            # Update best partial
            if (state.admits < best_partial.admits or
                (state.admits == best_partial.admits and state.errors < best_partial.errors)):
                best_partial = state

            # Expand state
            if cfg.verbose:
                print(f"  [expand] admits={state.admits} errors={state.errors}")

            children = _expand_state(state, cfg, out_dir, log)
            new_states.extend(children)
            explored += len(children)

        if not new_states:
            if cfg.verbose:
                print(f"[Pareto] No new states generated, stopping")
            break

        # Update frontier with Pareto filtering
        frontier = pareto_filter(frontier + new_states, cfg.max_frontier_size)

        if cfg.verbose:
            print(f"  [frontier] New frontier size: {len(frontier)}, best: admits={best_partial.admits} errors={best_partial.errors}")

    jh.close()
    # Timed out or max iterations
    return ParetoResult(
        solved=False,
        final_path=best_partial.path,
        iteration=iteration,
        states_explored=explored,
        best_partial_path=best_partial.path,
        best_partial_admits=best_partial.admits,
        best_partial_errors=best_partial.errors,
        initial_admits=init_admits,
        initial_errors=init_errors,
        journal_path=journal_path,
    )


# ------------------------------ Actions -------------------------------------

def _expand_state(state: ParetoState, cfg: ParetoConfig, out_dir: pathlib.Path, log) -> List[ParetoState]:
    """Expand a state by trying all applicable actions."""
    children = []
    ctx = _first_admit_info(state.path)

    # Determine which actions to try
    actions = []

    # If starting with empty body (0 admits but errors), seed first
    if state.iteration == 0 and state.admits == 0 and state.errors > 0:
        actions.append("seed_admitter")

    # If we have admits, try to reduce them
    if state.admits > 0 and ctx is not None:
        # Try sketch oracle first (full body replacement)
        if cfg.sketch_oracle:
            actions.append("sketch_oracle")
        # Try induction sketcher
        actions.append("sketch_induction")
        # Try oracle for single admit
        if cfg.oracle:
            actions.append("oracle_single_admit")
        # Try LLM
        if cfg.use_llm and _llm:
            actions.append("llm_single_admit")

    # If no admits but still errors, try error-based sketching
    if state.admits == 0 and state.errors > 0:
        actions.append("sketch_errors")

    for action in actions:
        child = _apply_action(state, action, ctx, cfg, out_dir, log)
        if child:
            children.append(child)

    return children


def _apply_action(
    state: ParetoState,
    action: str,
    ctx: Optional[Dict],
    cfg: ParetoConfig,
    out_dir: pathlib.Path,
    log,
) -> Optional[ParetoState]:
    """Apply one action, return a new state if successful."""
    base = state.path

    try:
        if action == "seed_admitter":
            patched = run_dafny_admitter(base, mode="admit", timeout=cfg.admitter_timeout)
            vr = quick_verify(patched, timeout=cfg.verify_timeout)
            a = count_admits(patched)
            child = ParetoState(
                path=patched,
                admits=a,
                errors=vr.errors,
                iteration=state.iteration + 1,
                parent_id=state.id(),
                action=action
            )
            log({"event": "action", "action": action, "from": state.id(), "to": child.id(), "admits": a, "errors": vr.errors})
            return child

        if action == "sketch_oracle" and cfg.sketch_oracle and ctx:
            method = ctx.get("method")
            src_text = _read_text(base)
            src_prompt = replace_method_body(src_text, method, "/*[SKETCH HERE]*/")
            guesses = list(set(cfg.sketch_oracle(src_prompt)))

            if guesses:
                # Try first guess
                sketch_body = guesses[0]
                new_src = replace_method_body(src_text, method, sketch_body)
                new_path = _write_version(out_dir, base, f"i{state.iteration+1}.sketch_oracle", new_src)
                patched = run_dafny_admitter(new_path, mode="admit", timeout=cfg.admitter_timeout)
                vr = quick_verify(patched, timeout=cfg.verify_timeout)
                a = count_admits(patched)
                child = ParetoState(
                    path=patched,
                    admits=a,
                    errors=vr.errors,
                    iteration=state.iteration + 1,
                    parent_id=state.id(),
                    action=action
                )
                log({"event": "action", "action": action, "from": state.id(), "to": child.id(), "admits": a, "errors": vr.errors})
                return child

        if action == "sketch_induction" and ctx:
            method = ctx.get("method")
            sk_body = run_sketcher(base, "induction_search", method=method, timeout=cfg.sketcher_timeout)
            if sk_body and sk_body.strip():
                src_text = _read_text(base)
                replaced = replace_method_body(src_text, method, sk_body.strip())
                new_path = _write_version(out_dir, base, f"i{state.iteration+1}.sketch_induction", replaced)
                patched = run_dafny_admitter(new_path, mode="admit", timeout=cfg.admitter_timeout)
                vr = quick_verify(patched, timeout=cfg.verify_timeout)
                a = count_admits(patched)
                child = ParetoState(
                    path=patched,
                    admits=a,
                    errors=vr.errors,
                    iteration=state.iteration + 1,
                    parent_id=state.id(),
                    action=action
                )
                log({"event": "action", "action": action, "from": state.id(), "to": child.id(), "admits": a, "errors": vr.errors})
                return child

        if action == "sketch_errors":
            sk_body = run_sketcher(base, "errors_warnings", timeout=cfg.sketcher_timeout)
            if sk_body and sk_body.strip() and ctx:
                method = ctx.get("method")
                src_text = _read_text(base)
                replaced = replace_method_body(src_text, method, sk_body.strip())
                new_path = _write_version(out_dir, base, f"i{state.iteration+1}.sketch_errors", replaced)
                patched = run_dafny_admitter(new_path, mode="admit", timeout=cfg.admitter_timeout)
                vr = quick_verify(patched, timeout=cfg.verify_timeout)
                a = count_admits(patched)
                child = ParetoState(
                    path=patched,
                    admits=a,
                    errors=vr.errors,
                    iteration=state.iteration + 1,
                    parent_id=state.id(),
                    action=action
                )
                log({"event": "action", "action": action, "from": state.id(), "to": child.id(), "admits": a, "errors": vr.errors})
                return child

        if action == "oracle_single_admit" and cfg.oracle and ctx:
            src_text = _read_text(base)
            src_prompt = src_text.replace(ctx.get("target_line_text", ""), "/*[CODE HERE]*/")
            guesses = list(set(cfg.oracle(src_prompt)))

            # Try up to max_oracle_guesses
            best_child = None
            best_score = (state.admits, state.errors)

            for idx, patch in enumerate(guesses[:cfg.max_oracle_guesses]):
                try:
                    after_src = _apply_admit_patch(src_text, ctx["line"], patch)
                    cand_path = _write_version(out_dir, base, f"i{state.iteration+1}.oracle_{idx}", after_src)
                    patched = run_dafny_admitter(cand_path, mode="admit", timeout=cfg.admitter_timeout)
                    vr = quick_verify(patched, timeout=cfg.verify_timeout)
                    a = count_admits(patched)

                    if (a, vr.errors) < best_score:
                        best_score = (a, vr.errors)
                        best_child = ParetoState(
                            path=patched,
                            admits=a,
                            errors=vr.errors,
                            iteration=state.iteration + 1,
                            parent_id=state.id(),
                            action=f"{action}_{idx}"
                        )
                except Exception:
                    continue

            if best_child:
                log({"event": "action", "action": action, "from": state.id(), "to": best_child.id(), "admits": best_child.admits, "errors": best_child.errors})
                return best_child

        if action == "llm_single_admit" and cfg.use_llm and _llm and ctx:
            vr = quick_verify(base, timeout=cfg.verify_timeout)
            errs = vr.output or ""
            proposal = _llm.propose_patch_for_admit(
                method=ctx.get("method", ""),
                errors=errs,
                target_line_text=ctx.get("target_line_text", ""),
                local_context_before=ctx.get("local_before", ""),
                local_context_after=ctx.get("local_after", ""),
                file_source=ctx.get("file_source", ""),
                tries=cfg.llm_tries,
            )
            if proposal and proposal.strip():
                src_text = _read_text(base)
                after_src = _apply_admit_patch(src_text, ctx["line"], proposal)
                new_path = _write_version(out_dir, base, f"i{state.iteration+1}.llm", after_src)
                patched = run_dafny_admitter(new_path, mode="admit", timeout=cfg.admitter_timeout)
                vr = quick_verify(patched, timeout=cfg.verify_timeout)
                a = count_admits(patched)
                child = ParetoState(
                    path=patched,
                    admits=a,
                    errors=vr.errors,
                    iteration=state.iteration + 1,
                    parent_id=state.id(),
                    action=action
                )
                log({"event": "action", "action": action, "from": state.id(), "to": child.id(), "admits": a, "errors": vr.errors})
                return child

        return None

    except Exception as e:
        log({"event": "error", "action": action, "state": state.id(), "message": str(e)})
        return None


# ------------------------------- CLI ----------------------------------------

def main(argv: Optional[List[str]] = None):
    ap = argparse.ArgumentParser(
        prog="pareto-prover",
        description="Pareto-optimal multi-objective proof search for Dafny"
    )
    ap.add_argument("--file", required=True, help="Path to the .dfy file to prove")
    ap.add_argument("--out-dir", default=None, help="Directory for outputs; defaults to pareto_out_<timestamp>")
    ap.add_argument("--max-iterations", type=int, default=20, help="Maximum iterations")
    ap.add_argument("--max-frontier-size", type=int, default=20, help="Maximum frontier size")
    ap.add_argument("--global-timeout", type=int, default=600, help="Stop after N seconds")
    ap.add_argument("--verify-timeout", type=int, default=20, help="Seconds for quick verify")
    ap.add_argument("--admitter-timeout", type=int, default=60, help="Seconds for admitter")
    ap.add_argument("--sketcher-timeout", type=int, default=60, help="Seconds for sketcher")
    ap.add_argument("--use-llm", action="store_true", help="Enable LLM actions")
    ap.add_argument("--llm-tries", type=int, default=2, help="LLM retry attempts")
    ap.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    args = ap.parse_args(argv)

    cfg = ParetoConfig(
        max_iterations=args.max_iterations,
        max_frontier_size=args.max_frontier_size,
        global_timeout=args.global_timeout,
        verify_timeout=args.verify_timeout,
        admitter_timeout=args.admitter_timeout,
        sketcher_timeout=args.sketcher_timeout,
        use_llm=args.use_llm,
        llm_tries=args.llm_tries,
        verbose=args.verbose,
        out_dir=pathlib.Path(args.out_dir) if args.out_dir else None,
    )

    src = pathlib.Path(args.file)
    res = run_pareto(src, cfg)

    status = "SOLVED" if res.solved else "PARTIAL"
    print(f"[Pareto] {status} iteration={res.iteration} explored={res.states_explored}")
    print(f"[Pareto] best: admits={res.best_partial_admits} errors={res.best_partial_errors}")
    print(f"[Pareto] output: {res.final_path}")
    print(f"[Pareto] journal: {res.journal_path}")


if __name__ == "__main__":
    main()
