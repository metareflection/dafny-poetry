# dafny-poetry

Implementation of the [POETRY][poetry] algorithm (NeurIPS 2024) for Dafny.
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/metareflection/dafny-poetry)

POETRY proves theorems recursively using:
- **Verifiable proof sketches** with `Admit(...)` placeholders
- **Recursive best-first search** exploring multiple proof strategies
- **Greedy sketch exploration** - pause and recurse when sketch found
- **Hybrid reasoning** - symbolic (induction) + neural (LLM)

## Dependencies

- [`dafny-admitter`][dafny-admitter] – creates verifiable sketches with `Admit("…", φ)` obligations
- [`dafny-sketcher-cli`][dafny-sketcher-cli] – symbolic reasoning (induction search, errors)
- [`dafny-annotator`][dafny-annotator] - oracle training (assertions, helper lemma calls)

## Install

```bash
pip install -e .
```

## Quick Start

```bash
# POETRY with LLM (requires LLM setup - see below)
dafny-poetry --file examples/example.dfy --use-llm -v

# Symbolic only (no LLM required)
dafny-poetry --file examples/example.dfy -v

# Compare with legacy greedy algorithm
dafny-poetry --file examples/example.dfy --legacy -v
```

## LLM Setup (Optional)

For LLM-based refinement, set environment variables as needed. See [llm.py](llm.py).

Without LLM setup, POETRY uses only symbolic reasoning (induction search).

## CLI Usage

```bash
dafny-poetry --file FILE.dfy [OPTIONS]
```

**Key options:**
- `--max-depth N` - Maximum recursion depth (default: 10)
- `--max-branches N` - LLM samples per expansion (default: 2)
- `--global-timeout SEC` - Overall time limit (default: 600)
- `--local-timeout SEC` - Per-level timeout for depth > 1 (default: 120)
- `--use-llm` - Enable LLM refinement
- `--llm-tries N` - LLM retries per sample (default: 2)
- `-v, --verbose` - Detailed logging
- `--legacy` - Use old greedy algorithm (for comparison)

**Example:**
```bash
dafny-poetry --file proof.dfy \
  --max-depth 15 \
  --max-branches 4 \
  --use-llm \
  -v
```

## Python API

```python
from pathlib import Path
from dafny_poetry.api import verify_dafny_file

result = verify_dafny_file(
    Path("proof.dfy"),
    max_depth=10,
    use_llm=True,
    verbose=True
)

print(f"Success: {result.success}")
print(f"Admits: {result.final_admits}/{result.initial_admits}")
```

See [`API.md`](API.md) for complete API documentation.

## How It Works

1. **Create initial sketch** - `dafny-admitter` converts failing proofs to `Admit(...)` calls
2. **Recursive BFS** - Search for proof at each level:
   - Select best OPEN node (highest score)
   - Expand via induction search + LLM samples
   - Create child nodes for each candidate
3. **Greedy recursion** - When sketch found (HALF_PROVED):
   - **Pause** current level
   - **Recurse** on sub-goal (Admit to prove)
   - If sub-goal succeeds → check if proof complete
   - If sub-goal fails → backpropagate, try alternative sketch
4. **Anytime verifiability** - All intermediate states verify with `Admit(...)` placeholders

See [`POETRY_IMPLEMENTATION.md`](POETRY_IMPLEMENTATION.md) for detailed algorithm description.

## Algorithm Features

✅ **Recursive BFS** - Best-first search at each recursion level  
✅ **Greedy sketch exploration** - Immediate recursion on first sketch found  
✅ **Status tracking** - OPEN, PROVED, HALF_PROVED, FAILED  
✅ **Branching** - Multiple LLM samples per expansion  
✅ **Backpropagation** - Invalidate failed sketches, try alternatives  
✅ **Hybrid reasoning** - Symbolic induction + neural LLM

## Documentation

- [`POETRY_IMPLEMENTATION.md`](POETRY_IMPLEMENTATION.md) - Complete algorithm guide
- [`API.md`](API.md) - Python API reference
- [`IMPLEMENTATION_SUMMARY.md`](IMPLEMENTATION_SUMMARY.md) - Implementation overview

## Implementation Notes

This is a **faithful implementation** of POETRY adapted for Dafny:
- Uses `Admit(...)` instead of Isabelle's `sorry`
- Batch verification via `dafny-admitter` (vs. interactive Isabelle)
- Scoring via admit reduction (paper uses LLM log probability)
- Pluggable LLM (paper uses trained model)

See implementation notes in [`POETRY_IMPLEMENTATION.md`](POETRY_IMPLEMENTATION.md) for details.

## References

- **Paper**: ["Proving Theorems Recursively"][poetry] (NeurIPS 2024)
- **Algorithm**: Section 3.2 - Recursive Best-First Search
- **Details**: Appendix A.1 - Status updates and termination

[poetry]: https://neurips.cc/virtual/2024/poster/93034
[dafny-admitter]: https://github.com/metareflection/dafny-admitter
[dafny-sketcher-cli]: https://github.com/namin/dafny-sketcher/blob/main/cli/AGENTS.md
[dafny-annotator]: https://github.com/metareflection/dafny-annotator
