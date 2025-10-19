"""
POETRY Recursive Best-First Search Algorithm.

Implements the full POETRY algorithm from the paper:
- Recursive BFS with greedy sketch exploration
- Best-first node selection by cumulative score
- Sorry edge handling and sub-goal recursion
- Status propagation and backpropagation

Paper note: When a deeper-level search on a sketch fails, convert the paused HP path back to OPEN (HP→OPEN) and resume BFS at the current level (POETRY Fig. 2(c), Appendix A.1).
"""

import pathlib
import time
from typing import Tuple, List, Optional, Callable
from dataclasses import dataclass
import re

from .proof_tree import ProofNode, SorryEdge, SearchTree, NodeStatus, SorryStatus
from .dafny_io import (
    run_dafny_admitter, run_sketcher, count_admits, 
    collect_first_admit_context, write_version, replace_method_body
)
from .utils import extract_method_body_text

# Seed guardrail: allow small admit growth when installing a first sketch.
# Keeps "one-line Admit -> case skeleton" from exploding search space.
STRUCTURE_DELTA_MAX = 3

# Lazy import to avoid requiring LLM setup at import time
llm_agent = None

# Paper-aligned helper: HP→OPEN along the path (Fig. 2(c), Appendix A.1)
# ---------------------------------------------------------------------------
def _hp_path_to_open(node: "ProofNode") -> None:
    """
    Convert the whole HALF_PROVED path containing `node` back to OPEN,
    walking from `node` to the root. This matches the paper's HP→OPEN
    backtracking on recursion failure (Fig. 2(c)).
    """
    current = node
    while current is not None:
        if getattr(current, "status", None) == NodeStatus.HALF_PROVED:
            current.status = NodeStatus.OPEN
        current = current.parent

@dataclass
class PoetryConfig:
    """Configuration for POETRY algorithm."""
    max_depth: int = 10
    max_branches: int = 2  # Number of LLM samples per expansion
    global_timeout: int = 600
    local_timeout: int = 120  # Per level for depth > 1
    use_sketcher: bool = True
    use_llm: bool = True
    llm_tries: int = 2
    out_dir: pathlib.Path = pathlib.Path("poetry_out")
    verbose: bool = False
    oracle: Optional[Callable[[str], list[str]]] = None


def _read(p: pathlib.Path) -> str:
    """Read file content."""
    return p.read_text(encoding="utf-8", errors="ignore")

def _read_lines(p: pathlib.Path) -> List[str]:
    return p.read_text(encoding="utf-8", errors="ignore").splitlines()

def _build_admit_context(file_path: pathlib.Path, ctx: dict, window: int = 12) -> dict:
    """
    Build a small local slice around the focused Admit(...) for patching.
    ctx['line'] is 1-based (from collect_first_admit_context).
    """
    lines = _read_lines(file_path)
    i = max(0, int(ctx.get("line", 1)) - 1)
    lo = max(0, i - window)
    hi = min(len(lines), i + window + 1)
    indent_match = re.match(r'\s*', lines[i]) if i < len(lines) else None
    indent = indent_match.group(0) if indent_match else ""
    return {
        "target_line": i,
        "indent": indent,
        "snippet_before": "\n".join(lines[lo:i]),
        "target_line_text": lines[i] if i < len(lines) else "",
        "snippet_after": "\n".join(lines[i+1:hi]),
        "lo": lo, "hi": hi
    }

def _apply_admit_patch(src_text: str, line_no: int, replacement: str) -> str:
    """Replace exactly one line (the Admit line) with the LLM patch, preserving indentation."""
    lines = src_text.splitlines()
    if line_no < 0 or line_no >= len(lines): return src_text
    indent = re.match(r'\s*', lines[line_no]).group(0)
    rep = replacement.strip()
    # Strip stray fences if the LLM emitted them
    rep = re.sub(r'^\s*```[a-zA-Z]*\s*', '', rep)
    rep = re.sub(r'\s*```$', '', rep)
    rep_lines = [(indent + ln) if ln.strip() else ln for ln in rep.splitlines()]
    lines[line_no:line_no+1] = rep_lines
    return "\n".join(lines)

def _looks_like_method_entry(body: str) -> bool:
    # empty or only comments/whitespace + a single top-level Admit(...)
    stripped = body.strip()
    if not stripped:
        return True
    # Allow leading comments/blank lines
    # Then require the first real token to be Admit(
    import re
    return re.match(r'^(?:\s*//.*\n|\s*)*Admit\s*\(', body) is not None

def  _looks_top_level(node: ProofNode, body: str) -> bool:
    at_level_root = (node.parent is None)   # root of this BFS level (top of theorem OR subgoal)
    at_method_entry = _looks_like_method_entry(body)
    return at_level_root and at_method_entry

def expand_node(node: ProofNode, config: PoetryConfig) -> List[ProofNode]:
    """
    Expand a node by generating candidate proof steps.
    Returns list of child nodes created.
    
    Actions tried:
    1. For top-level method body: sketching (either symbolic induction sketch or LLM-based sketch)
    2. Refinement as a single‑admit patch (multiple samples if enabled): Oracle and/or LLM
    """
    children = []
    
    # Get the focused method (first Admit)
    ctx = collect_first_admit_context(node.file_path)
    if not ctx or not ctx.get("method"):
        return children
    
    method = ctx["method"]
    node.focused_method = method
    
    if config.verbose:
        print(f"  [expand] node={node.admits} admits, method={method}, score={node.score:.2f}")
    
    if config.use_llm:
        # Lazy import of llm_agent
        global llm_agent
        if llm_agent is None:
            from . import llm_agent as llm_agent_module
            llm_agent = llm_agent_module

    # Action 1: Try induction search or LLM sketch
    src_text = _read(node.file_path)
    current_body = extract_method_body_text(src_text, method)
    if _looks_top_level(node, current_body or ""):
        if config.verbose: print(f"    [top-level] method={method}")
        if config.use_sketcher or config.use_llm:
            try:
                if config.use_sketcher:
                    sk_body = run_sketcher(node.file_path, "induction_search", method=method, timeout=120)
                elif config.use_llm:
                    sk_body = llm_agent.propose_new_body(
                        method, "", "",  "", file_source=src_text, 
                        tries=max(1, config.llm_tries), sketch=True)
                if sk_body and sk_body.strip():
                    src_text = _read(node.file_path)
                    replaced = replace_method_body(src_text, method, sk_body.strip())
                    cand = write_version(config.out_dir, node.file_path, f"sketch_{node.depth}", replaced)
                    # Verify the candidate and apply the same admit gate
                    cand_after = run_dafny_admitter(cand, mode="admit", only_failing=True, timeout=180)
                    admits_after = count_admits(cand_after)
                    # Must compile
                    try:
                        _ = run_sketcher(cand_after, "errors_warnings", method=None, timeout=60)
                        compiles = True
                    except Exception as e:
                        compiles = False
                        if config.verbose: print(f"    [sketch] failed to verify: {e}")
                    # Seed acceptance: either strict improvement, or small admit growth at method entry
                    progress = admits_after < node.admits
                    small_growth_ok = admits_after <= node.admits + STRUCTURE_DELTA_MAX
                    if compiles and (progress or small_growth_ok):
                        child =ProofNode(
                            file_path=cand_after,
                            admits=admits_after,
                            parent=node,
                            action_taken="induction" if config.use_sketcher else "llm_sketch",
                            score=node.score + 1.0,
                            depth=node.depth
                        )
                        children.append(child)
                        if config.verbose:
                            print(f"    [sketch] accepted: {admits_after} admits (was {node.admits})")
                    else:
                        if config.verbose: print(f"    [sketch] rejected: progress={progress}, small_growth_ok={small_growth_ok}, compiles={compiles}")
            except Exception as e:
                if config.verbose:
                    print(f"    [sketch] failed: {e}")
    else:
        if config.verbose: print(f"    [non-top-level] method={method}")
 
    # Action 2.1: Oracle refinement as a single‑admit patch (multiple samples)
    oracle_ok = False
    if config.oracle:
        src_text = _read(node.file_path)
        errors = ""
        # Build precise local context around the focused admit
        admit_ctx = _build_admit_context(node.file_path, ctx)

        
        src_prompt = src_text.replace(admit_ctx["target_line_text"], "/*[CODE HERE]*/")
        guesses = list(set(config.oracle(src_prompt)))
        for sample_idx, patch_text in enumerate(guesses):
            try:
                if patch_text and patch_text.strip() and "Admit" not in patch_text:
                    # Apply a *surgical* patch: replace only the target Admit line
                    patched_src = _apply_admit_patch(src_text, admit_ctx["target_line"], patch_text)
                    cand = write_version(
                        config.out_dir, node.file_path, f"oracle_patch_{node.depth}_{sample_idx}", patched_src
                    )

                    # Verify candidate
                    _ = run_sketcher(cand, "errors_warnings", method=None, timeout=60)  # parse/typecheck gate
                    cand_patched = run_dafny_admitter(cand, mode="admit", only_failing=True, timeout=180)
                    admits_after = count_admits(cand_patched)

                    if admits_after <= node.admits:
                        # small bonus for eliminating the focus
                        score_delta = float(node.admits - admits_after) + 0.5 if admits_after < node.admits else -0.5
                        child = ProofNode(
                            file_path=cand_patched,
                            admits=admits_after,
                            parent=node,
                            action_taken=f"oracle_patch_{sample_idx}",
                            score=node.score + score_delta,
                            depth=node.depth
                        )
                        children.append(child)
                        if config.verbose:
                            print(f"    [oracle_patch_{sample_idx}] → {admits_after} admits (was {node.admits})")
                        oracle_ok = True
                        break
            except Exception as e:
                if config.verbose:
                    print(f"    [oracle_{sample_idx}] failed: {e}")
  
    # Action 2.2: LLM refinement as a single‑admit patch (multiple samples)
    if not oracle_ok and config.use_llm:
        src_text = _read(node.file_path)
        errors = ""
        # Build precise local context around the focused admit
        admit_ctx = _build_admit_context(node.file_path, ctx)

        # Generate multiple LLM candidates
        for sample_idx in range(config.max_branches):
            try:
                patch_text = llm_agent.propose_patch_for_admit(
                        method=method,
                        errors=errors,
                        target_line_text=admit_ctx["target_line_text"],
                        local_context_before=admit_ctx["snippet_before"],
                        local_context_after=admit_ctx["snippet_after"],
                        file_source=src_text,
                        tries=max(1, config.llm_tries),
                    )

                if patch_text and patch_text.strip():
                    # Apply a *surgical* patch: replace only the target Admit line
                    patched_src = _apply_admit_patch(src_text, admit_ctx["target_line"], patch_text)
                    cand = write_version(
                        config.out_dir, node.file_path, f"llm_patch_{node.depth}_{sample_idx}", patched_src
                    )

                    # Verify candidate
                    _ = run_sketcher(cand, "errors_warnings", method=None, timeout=60)  # parse/typecheck gate
                    cand_patched = run_dafny_admitter(cand, mode="admit", only_failing=True, timeout=180)
                    admits_after = count_admits(cand_patched)

                    if admits_after < node.admits:
                        score_delta = float(node.admits - admits_after) + 0.5  # small bonus for eliminating the focus
                        child = ProofNode(
                            file_path=cand_patched,
                            admits=admits_after,
                            parent=node,
                            action_taken=f"llm_patch_{sample_idx}",
                            score=node.score + score_delta,
                            depth=node.depth
                        )
                        children.append(child)
                        if config.verbose:
                            print(f"    [llm_patch_{sample_idx}] → {admits_after} admits (was {node.admits})")
            except Exception as e:
                if config.verbose:
                    print(f"    [llm_{sample_idx}] failed: {e}")
    
    return children


def update_node_status(node: ProofNode):
    """
    Update node status based on children's status.
    Implements status propagation rules from the paper.
    """
    if node.is_proved():
        node.status = NodeStatus.PROVED
        return
    
    if not node.children:
        # Leaf node, stays OPEN until expanded or marked FAILED
        return
    
    child_statuses = [c.status for c in node.children]
    
    # If any child is PROVED → parent is PROVED
    if NodeStatus.PROVED in child_statuses:
        node.status = NodeStatus.PROVED
    # If any child is HALF_PROVED → parent is HALF_PROVED
    elif NodeStatus.HALF_PROVED in child_statuses:
        node.status = NodeStatus.HALF_PROVED
    # If all children are FAILED → parent is FAILED
    elif all(s == NodeStatus.FAILED for s in child_statuses):
        node.status = NodeStatus.FAILED
    # Otherwise stays OPEN
    else:
        node.status = NodeStatus.OPEN


def propagate_status_upward(node: ProofNode):
    """Propagate status changes up the tree."""
    current = node
    while current.parent is not None:
        old_status = current.parent.status
        update_node_status(current.parent)
        if current.parent.status == old_status:
            break  # No change, stop propagating
        current = current.parent


def identify_sorry_edges(parent: ProofNode, child: ProofNode) -> List[SorryEdge]:
    """
    Identify new Admit calls introduced by the transition from parent to child.
    These become sorry edges that need recursive proving.
    
    Note: In this implementation, we track Admits that were present before
    but now need focused solving. The child should have fewer admits than parent.
    """
    edges = []
    
    # In Dafny, each Admit represents a sub-goal
    # We identify the current focused admit as a sorry edge
    ctx = collect_first_admit_context(child.file_path)
    if ctx and ctx.get("method"):
        edge = SorryEdge(
            parent_node=parent,
            child_node=child,
            admit_tag=ctx.get("tag", ""),
            admit_line=ctx.get("line", 0),
            sub_goal_root=None,  # Will be created when recursing
            sub_goal_status=SorryStatus.OPEN
        )
        edges.append(edge)
    
    return edges


def recursive_bfs(root: ProofNode, level: int, config: PoetryConfig, 
                  start_time: float) -> Tuple[NodeStatus, ProofNode]:
    """
    Recursive best-first search for proof.
    
    Args:
        root: Root node of current search level
        level: Recursion depth (1 = top-level theorem)
        config: Algorithm configuration
        start_time: Global start time for timeout
        
    Returns:
        (final_status, best_node)
    """
    level_start = time.time()
    timeout = config.local_timeout if level > 1 else config.global_timeout
    
    if config.verbose:
        print(f"\n[LEVEL {level}] Starting BFS, admits={root.admits}, timeout={timeout}s")
    
    # Check depth limit
    if level > config.max_depth:
        if config.verbose:
            print(f"[LEVEL {level}] Max depth reached")
        root.status = NodeStatus.FAILED
        return NodeStatus.FAILED, root
    
    # Check global timeout
    if time.time() - start_time >= config.global_timeout:
        if config.verbose:
            print(f"[LEVEL {level}] Global timeout")
        return root.status, root
    
    # Initialize search tree
    search_tree = SearchTree(root=root)
    
    # Main BFS loop
    iteration = 0
    while (time.time() - level_start < timeout and 
           time.time() - start_time < config.global_timeout):
        
        iteration += 1
        
        # Check termination: root proved or all failed
        if root.status == NodeStatus.PROVED:
            if config.verbose:
                print(f"[LEVEL {level}] PROVED in {iteration} iterations")
            # Return the best proved node (the one with the actual proof), not root
            return NodeStatus.PROVED, search_tree.get_best_node()
        
        if root.status == NodeStatus.FAILED:
            if config.verbose:
                print(f"[LEVEL {level}] FAILED - all paths exhausted")
            return NodeStatus.FAILED, root
        
        # GREEDY RECURSION: Check for sketch (HALF_PROVED path)
        if search_tree.has_half_proved_path():
            sorry_edge = search_tree.get_first_unproved_sorry_edge()
            
            if sorry_edge:
                if config.verbose:
                    print(f"[LEVEL {level}] Sketch found! Recursing on sub-goal at line {sorry_edge.admit_line}")
                
                # Create sub-goal root node
                # The sub-goal is the current state with this admit as focus
                sorry_edge.sub_goal_root = ProofNode(
                    file_path=sorry_edge.child_node.file_path,
                    admits=sorry_edge.child_node.admits,
                    parent=None,  # New tree root
                    score=0.0,
                    depth=level + 1
                )
                
                # RECURSE
                sub_status, sub_proof = recursive_bfs(
                    sorry_edge.sub_goal_root,
                    level=level + 1,
                    config=config,
                    start_time=start_time
                )
                
                # Update sorry edge status
                if sub_status == NodeStatus.PROVED:
                    sorry_edge.sub_goal_status = SorryStatus.PROVED
                    if config.verbose:
                        print(f"[LEVEL {level}] Sub-goal PROVED!")
                    
                    # Check if all sorry edges on path are now proved
                    path = sorry_edge.child_node.get_path_to_root()
                    all_proved = all(
                        all(e.sub_goal_status == SorryStatus.PROVED 
                            for e in n.sorry_edges)
                        for n in path
                    )
                    
                    if all_proved:
                        # Complete proof! Transfer the proved file from sub-goal
                        sorry_edge.child_node.status = NodeStatus.PROVED
                        sorry_edge.child_node.file_path = sub_proof.file_path
                        sorry_edge.child_node.admits = sub_proof.admits
                        propagate_status_upward(sorry_edge.child_node)
                        if config.verbose:
                            print(f"[LEVEL {level}] All sub-goals proved! Complete proof found.")
                        continue  # Will check root.status next iteration
                else:
                    # Recursion failed
                    # paper's HP→OPEN backtrack (Fig. 2(c), Appendix A.1)
                    sorry_edge.sub_goal_status = SorryStatus.FAILED
                    if config.verbose:
                        print(f"[LEVEL {level}] Sub-goal FAILED. HP→OPEN and continue search at current level.")

                    
                    # Convert the entire paused HP path back to OPEN,
                    # then update statuses upward and continue level-`level` BFS.
                    _hp_path_to_open(sorry_edge.child_node)
                    propagate_status_upward(sorry_edge.child_node)
                    continue
        
        # Standard BFS: Select best OPEN node
        current = search_tree.get_best_open_node()
        if current is None:
            # No more open nodes
            if config.verbose:
                print(f"[LEVEL {level}] No more open nodes")
            root.status = NodeStatus.FAILED
            break
        
        if config.verbose:
            print(f"[LEVEL {level}] Iteration {iteration}: expanding node {current}")
        
        # Expand node
        children = expand_node(current, config)
        
        for child in children:
            # Add to tree
            current.children.append(child)
            search_tree.add_node(child)
            
            # Check if proved
            if child.is_proved():
                child.status = NodeStatus.PROVED
                propagate_status_upward(child)
                if config.verbose:
                    print(f"[LEVEL {level}] Complete proof found (0 admits)!")
                continue
            
            # Check for sorry edges (potential sketches)
            sorry_edges = identify_sorry_edges(current, child)
            if sorry_edges:
                child.sorry_edges = sorry_edges
                child.status = NodeStatus.HALF_PROVED
                propagate_status_upward(child)
                if config.verbose:
                    print(f"[LEVEL {level}] Sketch found with {len(sorry_edges)} sorry edges")
            else:
                child.status = NodeStatus.OPEN
        
        # Update current node status
        if not children:
            # No children generated → mark as FAILED
            current.status = NodeStatus.FAILED
        else:
            update_node_status(current)
        
        propagate_status_upward(current)
    
    # Timeout or exhausted
    if config.verbose:
        elapsed = time.time() - level_start
        print(f"[LEVEL {level}] Finished: status={root.status.value}, time={elapsed:.1f}s")
    
    return root.status, search_tree.get_best_node()


def run_poetry(src_path: pathlib.Path, config: PoetryConfig) -> Tuple[int, pathlib.Path]:
    """
    Run POETRY algorithm on a Dafny file.
    
    Returns:
        (exit_code, final_file_path)
        exit_code: 0 = success, 2 = partial progress
    """
    start_time = time.time()
    
    # Create initial verifiable sketch
    if config.verbose:
        print(f"[POETRY] Creating initial sketch from {src_path}")
    
    patched = run_dafny_admitter(src_path, mode="admit", only_failing=True, 
                                timeout=min(config.global_timeout, 300))
    initial_admits = count_admits(patched)
    
    if config.verbose:
        print(f"[POETRY] Initial sketch: {initial_admits} admits")
    
    # Create root node
    root = ProofNode(
        file_path=patched,
        admits=initial_admits,
        parent=None,
        action_taken="seed",
        score=0.0,
        depth=1
    )
    
    # Run recursive BFS
    final_status, best_node = recursive_bfs(root, level=1, config=config, 
                                           start_time=start_time)
    
    # Return results
    if final_status == NodeStatus.PROVED or best_node.admits == 0:
        if config.verbose:
            print(f"\n[POETRY] ✓ SUCCESS! Proof complete.")
        return 0, best_node.file_path
    else:
        if config.verbose:
            print(f"\n[POETRY] ✗ PARTIAL: {best_node.admits} admits remaining")
        return 2, best_node.file_path

