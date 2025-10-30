"""
Python API for POETRY - Proof repair via LLM and symbolic sketching.

This module provides a programmatic interface for benchmarking and automation.
"""

import pathlib
import tempfile
import shutil
from typing import Optional, Callable
from dataclasses import dataclass

from .poetry_recursive import run_poetry, PoetryConfig


@dataclass
class PoetryResult:
    """Result of running POETRY on a Dafny file."""
    success: bool
    final_admits: int
    final_path: pathlib.Path
    initial_admits: int
    iterations: int
    error: Optional[str] = None


def verify_dafny(
    dfy_source: str,
    max_depth: int = 3,
    max_branches: int = 2,
    max_iterations: int = 20,
    use_sketcher: bool = True,
    use_llm: bool = True,
    llm_tries: int = 2,
    timeout: int = 600,
    verbose: bool = False,
    out_dir: Optional[pathlib.Path] = None,
    oracle: Optional[Callable[[str], list[str]]] = None,
    sketch_oracle: Optional[Callable[[str], list[str]]] = None
) -> PoetryResult:
    """
    Attempt to verify a Dafny program using POETRY.

    Args:
        dfy_source: Dafny source code as a string
        max_depth: Maximum POETRY loop depth (default: 3)
        max_branches: Maximum candidates per expansion (LLM and Oracle) (default: 2)
        max_iterations: Maximum BFS iterations per level (default: 20)
        use_sketcher: Whether to use symbolic sketcher (default: True)
        use_llm: Whether to use LLM for proof repair (default: True)
        llm_tries: Number of LLM attempts per iteration (default: 2)
        timeout: Global timeout in seconds (default: 600)
        verbose: Print progress messages (default: False)
        out_dir: Output directory for intermediate files (default: temp dir)
        oracle: Optional oracle function for generating proof candidates
        sketch_oracle: Optional oracle function for generating sketch candidates

    Returns:
        PoetryResult with success status, final file, and statistics

    Example:
        >>> source = '''
        ... lemma {:induction false} SumFormula(n: nat)
        ...   ensures 2 * sum(n) == n * (n + 1)
        ... { }
        ... '''
        >>> result = verify_dafny(source, max_depth=3, use_llm=True)
        >>> print(f"Success: {result.success}, Admits: {result.final_admits}")
    """
    # Create temporary directory for processing if not provided
    cleanup_temp = out_dir is None
    if out_dir is None:
        out_dir = pathlib.Path(tempfile.mkdtemp(prefix="poetry_"))
    else:
        out_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Write source to temporary file
        src_file = out_dir / "input.dfy"
        src_file.write_text(dfy_source, encoding="utf-8")

        # Create config object
        config = PoetryConfig(
            max_depth=max_depth,
            max_branches=max_branches,
            max_iterations=max_iterations,
            global_timeout=timeout,
            local_timeout=120,  # Default
            use_sketcher=use_sketcher,
            use_llm=use_llm,
            llm_tries=llm_tries,
            out_dir=out_dir,
            verbose=verbose,
            oracle=oracle,
            sketch_oracle=sketch_oracle
        )

        # Run POETRY
        exit_code, final_path = run_poetry(src_file, config)

        # Count final admits
        from .dafny_io import count_admits
        final_admits = count_admits(final_path)

        # Count initial admits (from seed file)
        seed_file = final_path.parent / f"{src_file.stem}.patched.dfy"
        if seed_file.exists():
            initial_admits = count_admits(seed_file)
        else:
            initial_admits = final_admits

        # Estimate iterations from final path name
        iterations = 0
        if "_llm_" in final_path.stem or "_shift_" in final_path.stem:
            # Extract depth from filename like "input.patched.llm_1.patched.dfy"
            parts = final_path.stem.split('_')
            for i, part in enumerate(parts):
                if part in ('llm', 'shift', 'induction') and i + 1 < len(parts):
                    try:
                        iterations = max(iterations, int(parts[i + 1].split('.')[0]) + 1)
                    except (ValueError, IndexError):
                        pass

        return PoetryResult(
            success=(exit_code == 0),
            final_admits=final_admits,
            final_path=final_path,
            initial_admits=initial_admits,
            iterations=iterations
        )

    except Exception as e:
        return PoetryResult(
            success=False,
            final_admits=-1,
            final_path=out_dir / "error.dfy",
            initial_admits=-1,
            iterations=0,
            error=str(e)
        )

    finally:
        # Clean up temporary directory if we created it
        if cleanup_temp and out_dir.exists():
            shutil.rmtree(out_dir, ignore_errors=True)


def verify_dafny_file(
    dfy_path: pathlib.Path,
    max_depth: int = 3,
    use_llm: bool = True,
    llm_tries: int = 2,
    timeout: int = 600,
    verbose: bool = False,
    out_dir: Optional[pathlib.Path] = None
) -> PoetryResult:
    """
    Attempt to verify a Dafny file using POETRY.

    Args:
        dfy_path: Path to Dafny source file
        max_depth: Maximum POETRY loop depth (default: 3)
        use_llm: Whether to use LLM for proof repair (default: True)
        llm_tries: Number of LLM attempts per iteration (default: 2)
        timeout: Global timeout in seconds (default: 600)
        verbose: Print progress messages (default: False)
        out_dir: Output directory for intermediate files (default: same dir as input)

    Returns:
        PoetryResult with success status, final file, and statistics
    """
    source = dfy_path.read_text(encoding="utf-8")
    if out_dir is None:
        out_dir = dfy_path.parent / "poetry_out"
        out_dir.mkdir(exist_ok=True)

    return verify_dafny(source, max_depth, use_llm, llm_tries, timeout, verbose, out_dir)


if __name__ == "__main__":
    # Example usage
    example = """
    function sum(n: nat): nat
    {
      if n == 0 then 0 else n + sum(n-1)
    }

    lemma {:induction false} SumFormula(n: nat)
      ensures 2 * sum(n) == n * (n + 1)
    {
    }
    """

    result = verify_dafny(example, verbose=True, out_dir=pathlib.Path("api_test"))
    print(f"\nResult: success={result.success}, admits={result.final_admits}/{result.initial_admits}, iterations={result.iterations}")
    if result.error:
        print(f"Error: {result.error}")
    else:
        print(f"Final file: {result.final_path}")
