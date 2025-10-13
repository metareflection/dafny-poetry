# Full POETRY Implementation for Dafny

## Overview

This is a **complete implementation** of the POETRY algorithm from the paper "Proving Theorems Recursively" (NeurIPS 2024), adapted for Dafny.

## What Was Implemented

### ✅ Core Data Structures (`proof_tree.py`)

1. **ProofNode**: Nodes in the search tree
   - File state (path, admits count)
   - Tree structure (parent, children)
   - Search metadata (status, score, depth, action)
   - Sorry edges for sub-goals

2. **SorryEdge**: Links between recursion levels
   - Parent/child nodes
   - Admit tag and location
   - Sub-goal root and status

3. **SearchTree**: Manages search at each level
   - Best-first node selection
   - Status tracking
   - Sorry edge management

4. **Status Enums**:
   - `NodeStatus`: OPEN, PROVED, HALF_PROVED, FAILED
   - `SorryStatus`: OPEN, PROVED, FAILED

### ✅ Recursive BFS Algorithm (`poetry_recursive.py`)

Implements the full algorithm from Section 3 of the paper:

1. **Best-First Search** (`recursive_bfs`)
   - Selects best OPEN node by cumulative score
   - Iterative expansion until timeout or success
   - Proper status tracking and propagation

2. **Greedy Recursion** (Algorithm core feature)
   - As soon as a sketch (HALF_PROVED path) is found → pause and recurse
   - Recurse on first unproved sorry edge
   - Continue searching only after recursion fails

3. **Node Expansion** (`expand_node`)
   - **Action 1**: Induction search (symbolic via dafny-sketcher)
   - **Action 2**: LLM refinement (multiple samples = branching)
   - Generates multiple child nodes per expansion

4. **Status Propagation**
   - Upward propagation from children to parents
   - Implements rules from Appendix A.1:
     - Any child PROVED → parent PROVED
     - Any child HALF_PROVED → parent HALF_PROVED
     - All children FAILED → parent FAILED

5. **Backpropagation**
   - When recursion fails, invalidates the sketch
   - Marks HALF_PROVED nodes back to OPEN
   - Continues searching for alternative sketches

6. **Sorry Edge Handling**
   - Identifies Admit calls as sorry edges
   - Creates sub-goal roots for recursion
   - Tracks sub-goal status independently

### ✅ Configuration (`PoetryConfig`)

All algorithm parameters configurable:
- `max_depth`: Maximum recursion depth (default: 10)
- `max_branches`: Number of LLM samples per expansion (default: 2)
- `global_timeout`: Overall time limit (default: 600s)
- `local_timeout`: Per-level timeout for depth > 1 (default: 120s)
- `use_llm`: Enable LLM refinement (default: True)
- `llm_tries`: LLM retries per sample (default: 2)
- `out_dir`: Output directory for intermediate files
- `verbose`: Detailed logging

### ✅ Integration

1. **CLI** (`cli.py`):
   - New implementation is default
   - `--legacy` flag to use old greedy algorithm
   - All parameters exposed as command-line args

2. **API** (`api.py`):
   - Updated to use new implementation
   - Same interface maintained
   - Returns `PoetryResult` with statistics

## Key Algorithm Features

### 1. Recursive Best-First Search

```
For each recursion level:
    1. Select best OPEN node (highest score)
    2. Expand node → generate children
    3. Check for sketch (HALF_PROVED path)
    4. If sketch found → PAUSE and RECURSE immediately
    5. If recursion succeeds → check if proof complete
    6. If recursion fails → BACKPROPAGATE and continue BFS
    7. Repeat until PROVED, FAILED, or timeout
```

### 2. Greedy Sketch Exploration

From the paper:
> "Whenever a proof sketch is found, the current level of the best-first search will be paused, and POETRY will recursively call the best-first search algorithm"

**Implementation**:
- No heuristics to evaluate sketch quality
- First sketch found triggers immediate recursion
- Deterministic, depth-first commitment to sketches
- Only explores alternatives after recursion fails

### 3. Branching via Multiple Actions

Unlike single-path greedy search, POETRY generates multiple candidates:
- **Symbolic**: Induction search via dafny-sketcher
- **LLM**: Multiple samples (controlled by `max_branches`)
- Each child becomes an independent search path
- Best-first selection explores most promising paths

### 4. Status Tracking

Implements precise status semantics from paper:
- **OPEN**: Unexplored or partially explored
- **PROVED**: Complete proof (admits = 0)
- **HALF_PROVED**: Sketch with Admit calls (sorry edges)
- **FAILED**: All attempts exhausted

### 5. Scoring

Current implementation uses:
- Progress-based scoring: admits reduction
- Could be enhanced with LLM log probabilities
- Higher score = selected first in best-first search

## Dafny Adaptations

### Admit(...) ≈ sorry

| Isabelle (Paper) | Dafny (This Implementation) |
|-----------------|----------------------------|
| `sorry` keyword | `Admit("tag", formula)` |
| Proof level tracking | Admit count + line tracking |
| `have ... sorry` | Lemma with admits |

### Verification

- Isabelle: Interactive execution per step
- Dafny: Batch verification via `dafny-admitter`
- Ensures anytime verifiability (all states verify)

## Usage

### Command Line

```bash
# Full POETRY with default settings
dafny-poetry --file example.dfy --use-llm -v

# Custom configuration
dafny-poetry --file example.dfy \
  --max-depth 15 \
  --max-branches 4 \
  --global-timeout 900 \
  --use-llm \
  --llm-tries 3 \
  -v

# Compare with legacy greedy algorithm
dafny-poetry --file example.dfy --legacy -v
```

### Python API

```python
from pathlib import Path
from dafny_poetry.api import verify_dafny_file

result = verify_dafny_file(
    Path("example.dfy"),
    max_depth=10,
    use_llm=True,
    verbose=True
)

print(f"Success: {result.success}")
print(f"Admits: {result.final_admits}/{result.initial_admits}")
```

## What's Different from Paper

### Faithful to Paper

✅ Recursive BFS structure  
✅ Best-first node selection  
✅ Greedy sketch exploration  
✅ Status propagation rules  
✅ Sorry edge handling  
✅ Backpropagation on failure  

### Adaptations for Dafny

🔄 **Scoring**: Uses admit reduction (paper uses LLM log prob)  
🔄 **Actions**: Symbolic + LLM (paper uses trained model)  
🔄 **Verification**: Batch via dafny-admitter (paper uses interactive Isabelle)  
🔄 **Sorry edges**: Identified by Admit calls (paper uses proof level tracking)

### Not Yet Implemented

❌ Value function to prioritize sketches (mentioned in paper as future work)  
❌ Proof step caching  
❌ Parallel exploration of multiple sketches  

## Performance Expectations

From the paper, POETRY achieves:
- **2-3x longer proofs** than baseline GPT-f
- **Better success rate** on multi-level theorems
- **Reduced search space** via sketch verification

Expected behavior:
- **Level 1**: Explores broadly, finds sketches
- **Level 2+**: Focused sub-goal proving (120s timeout)
- **Backtracking**: When sub-goals fail, tries alternative sketches
- **Success**: When all sorry edges on a path are proved

## Debugging

Enable verbose mode to see:
- Level transitions and recursion
- Node expansion details
- Sketch discovery and recursion triggers
- Status updates and propagation
- Timeout and termination reasons

```bash
dafny-poetry --file example.dfy --use-llm -v 2>&1 | tee poetry.log
```

## Testing

Test on examples:
```bash
# Simple example
dafny-poetry --file examples/example.dfy --use-llm -v

# Flatten example (multi-level)
dafny-poetry --file examples/flatten.dfy --use-llm -v

# Compare with legacy
dafny-poetry --file examples/example.dfy --legacy -v
```

## File Structure

```
src/dafny_poetry/
├── proof_tree.py          # NEW: Data structures
├── poetry_recursive.py    # NEW: Full POETRY algorithm
├── poetry_alg.py          # OLD: Legacy greedy algorithm
├── cli.py                 # UPDATED: Uses new implementation
├── api.py                 # UPDATED: Uses new implementation
├── dafny_io.py           # Dafny tool wrappers
├── llm_agent.py          # LLM interaction
└── utils.py              # Utility functions
```

## Known Limitations

1. **Greedy sketch commitment**: No value function yet (future work in paper)
2. **Scoring heuristic**: Uses admit reduction, not LLM probability
3. **Serial recursion**: One sub-goal at a time (could parallelize)
4. **No caching**: Could cache proved lemmas/sub-goals

## Next Steps for Enhancement

1. **Value function**: Train model to score sketch quality
2. **Better scoring**: Integrate LLM log probabilities
3. **Caching**: Cache proved sub-goals across attempts
4. **Parallelization**: Explore multiple sketches in parallel
5. **Benchmarking**: Systematic evaluation on Dafny benchmarks

## References

- Paper: "Proving Theorems Recursively" (NeurIPS 2024)
- Algorithm: Section 3.2 (Recursive BFS)
- Details: Appendix A.1 (Status updates, termination)
- Walkthrough: Figure 2 (Visual example)

