# POETRY Python API

This document describes the Python API for programmatic use of POETRY (Proof repair via LLM and symbolic sketching).

## Installation

```bash
pip install -e .
```

## Quick Start

```python
from pathlib import Path
from dafny_poetry.api import verify_dafny_file

# Verify a Dafny file
result = verify_dafny_file(
    Path("examples/example.dfy"),
    max_depth=3,
    use_llm=True,
    verbose=True
)

print(f"Success: {result.success}")
print(f"Final admits: {result.final_admits}")
```

## API Reference

### `verify_dafny(source, **kwargs)`

Verify a Dafny program from a string.

**Parameters:**
- `dfy_source` (str): Dafny source code
- `max_depth` (int, optional): Maximum POETRY loop depth (default: 3)
- `use_llm` (bool, optional): Use LLM for proof repair (default: True)
- `llm_tries` (int, optional): LLM attempts per iteration (default: 2)
- `timeout` (int, optional): Global timeout in seconds (default: 600)
- `verbose` (bool, optional): Print progress (default: False)
- `out_dir` (Path, optional): Output directory (default: temp dir)

**Returns:** `PoetryResult` object

**Example:**
```python
from dafny_poetry.api import verify_dafny

source = """
function sum(n: nat): nat
{
  if n == 0 then 0 else n + sum(n-1)
}

lemma {:induction false} SumFormula(n: nat)
  ensures 2 * sum(n) == n * (n + 1)
{
}
"""

result = verify_dafny(source, max_depth=3, use_llm=True)
print(f"Solved: {result.success}")
```

### `verify_dafny_file(path, **kwargs)`

Verify a Dafny file.

**Parameters:**
- `dfy_path` (Path): Path to Dafny source file
- Same optional parameters as `verify_dafny()`

**Returns:** `PoetryResult` object

**Example:**
```python
from pathlib import Path
from dafny_poetry.api import verify_dafny_file

result = verify_dafny_file(
    Path("myproof.dfy"),
    max_depth=5,
    timeout=300
)
```

### `PoetryResult` Class

Result object returned by verification functions.

**Attributes:**
- `success` (bool): True if verification succeeded (0 admits remaining)
- `final_admits` (int): Number of remaining admits
- `initial_admits` (int): Number of admits at start
- `final_path` (Path): Path to final output file
- `iterations` (int): Number of POETRY iterations performed
- `error` (str | None): Error message if failed

**Example:**
```python
if result.success:
    print(f"Proof completed in {result.iterations} iterations")
    print(f"Final file: {result.final_path}")
else:
    print(f"Failed: {result.error or 'timeout or partial progress'}")
    print(f"Reduced admits from {result.initial_admits} to {result.final_admits}")
```

## Benchmarking

See `benchmark_example.py` for a complete benchmarking script.

**Example:**
```python
from pathlib import Path
from dafny_poetry.api import verify_dafny_file

# Benchmark a suite of files
examples = Path("examples").glob("*.dfy")
results = []

for dfy_file in examples:
    result = verify_dafny_file(
        dfy_file,
        max_depth=3,
        use_llm=True,
        timeout=600
    )
    results.append((dfy_file.name, result))

# Print summary
successful = sum(1 for _, r in results if r.success)
print(f"Solved {successful}/{len(results)} benchmarks")

for name, result in results:
    status = "✓" if result.success else "✗"
    admits = f"{result.final_admits}/{result.initial_admits}"
    print(f"{status} {name}: {admits} admits")
```

Run the included benchmark script:
```bash
python benchmark_example.py
```

## Environment Variables

The API respects the same environment variables as the CLI:

- `AWS_BEARER_TOKEN_BEDROCK`: AWS Bedrock authentication token
- `CLAUDE_MODEL`: Claude model to use (e.g., `sonnet45`, `opus`)
- `LLM_PROVIDER`: LLM provider (`claude_aws`, `claude_vertex`, etc.)
- `DEBUG_LLM`: Set to `true` to print LLM prompts

**Example:**
```bash
export AWS_BEARER_TOKEN_BEDROCK="your-token"
export CLAUDE_MODEL=sonnet45
python your_benchmark.py
```

## Error Handling

The API catches exceptions and returns them in the `error` field:

```python
result = verify_dafny(malformed_source)
if result.error:
    print(f"Error occurred: {result.error}")
else:
    print(f"Success: {result.success}")
```

## Tips for Benchmarking

1. **Set appropriate timeouts**: For harder problems, increase `timeout` parameter
2. **Adjust depth**: Increase `max_depth` for problems requiring more iterations
3. **LLM retries**: Set `llm_tries=3` or higher for more robust LLM attempts
4. **Parallel execution**: Use multiprocessing to run benchmarks in parallel
5. **Output directories**: Use separate `out_dir` for each benchmark to avoid conflicts

**Parallel Example:**
```python
from multiprocessing import Pool
from pathlib import Path
from dafny_poetry.api import verify_dafny_file

def run_one(dfy_file):
    return verify_dafny_file(
        dfy_file,
        out_dir=Path(f"benchmark_out/{dfy_file.stem}")
    )

files = list(Path("examples").glob("*.dfy"))
with Pool(4) as pool:
    results = pool.map(run_one, files)
```

## See Also

- [CLI Documentation](README.md)
- [Example Benchmark Script](benchmark_example.py)
