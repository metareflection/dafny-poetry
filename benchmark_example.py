#!/usr/bin/env python3
"""
Example benchmarking script for POETRY.

This demonstrates how to use the Python API to benchmark POETRY on a suite of examples.
"""

import pathlib
import time
from dataclasses import dataclass
from typing import List, Optional
from src.dafny_poetry.api import verify_dafny_file, PoetryResult


@dataclass
class BenchmarkResult:
    """Results for a single benchmark."""
    name: str
    success: bool
    time_seconds: float
    initial_admits: int
    final_admits: int
    iterations: int
    error: Optional[str] = None


def run_benchmark(
    dfy_file: pathlib.Path,
    max_depth: int = 3,
    use_llm: bool = True,
    timeout: int = 600
) -> BenchmarkResult:
    """Run POETRY on a single file and collect statistics."""
    name = dfy_file.stem
    out_dir = pathlib.Path(f"benchmark_out/{name}")

    start_time = time.time()
    result = verify_dafny_file(
        dfy_file,
        max_depth=max_depth,
        use_llm=use_llm,
        llm_tries=2,
        timeout=timeout,
        verbose=False,
        out_dir=out_dir
    )
    elapsed = time.time() - start_time

    return BenchmarkResult(
        name=name,
        success=result.success,
        time_seconds=elapsed,
        initial_admits=result.initial_admits,
        final_admits=result.final_admits,
        iterations=result.iterations,
        error=result.error
    )


def run_benchmark_suite(
    examples_dir: pathlib.Path,
    max_depth: int = 3,
    use_llm: bool = True,
    timeout: int = 600
) -> List[BenchmarkResult]:
    """Run POETRY on all .dfy files in a directory."""
    results = []

    dfy_files = sorted(examples_dir.glob("*.dfy"))
    print(f"Running benchmark on {len(dfy_files)} files...")
    print()

    for dfy_file in dfy_files:
        print(f"Processing {dfy_file.name}...", end=" ", flush=True)
        result = run_benchmark(dfy_file, max_depth, use_llm, timeout)

        status = "✓" if result.success else "✗"
        admits_str = f"{result.final_admits}/{result.initial_admits}" if result.initial_admits >= 0 else "N/A"
        print(f"{status} {result.time_seconds:.1f}s (admits: {admits_str}, iter: {result.iterations})")

        results.append(result)

    return results


def print_summary(results: List[BenchmarkResult]):
    """Print a summary of benchmark results."""
    print()
    print("=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)

    total = len(results)
    successful = sum(1 for r in results if r.success)
    failed = total - successful

    total_time = sum(r.time_seconds for r in results)
    avg_time = total_time / total if total > 0 else 0

    # Admits reduction statistics
    admits_reduced = [r for r in results if r.initial_admits > 0 and r.final_admits < r.initial_admits]
    fully_solved = [r for r in results if r.success and r.final_admits == 0]

    print(f"Total files:        {total}")
    print(f"Successful:         {successful} ({100*successful/total:.1f}%)")
    print(f"Failed:             {failed}")
    print(f"Fully solved:       {len(fully_solved)} (0 admits remaining)")
    print(f"Partial progress:   {len(admits_reduced) - len(fully_solved)}")
    print()
    print(f"Total time:         {total_time:.1f}s")
    print(f"Average time:       {avg_time:.1f}s")
    print()

    # Detailed results table
    print("DETAILED RESULTS:")
    print("-" * 80)
    print(f"{'File':<20} {'Status':<8} {'Time':<8} {'Admits':<12} {'Iter':<6}")
    print("-" * 80)

    for r in sorted(results, key=lambda x: (not x.success, x.name)):
        status = "SUCCESS" if r.success else "FAIL"
        admits_str = f"{r.final_admits}/{r.initial_admits}" if r.initial_admits >= 0 else "N/A"
        print(f"{r.name:<20} {status:<8} {r.time_seconds:>6.1f}s  {admits_str:<12} {r.iterations:<6}")

    print("-" * 80)


if __name__ == "__main__":
    import sys
    from typing import Optional

    # Run benchmark on examples directory
    examples_dir = pathlib.Path("examples")

    if not examples_dir.exists():
        print(f"Error: {examples_dir} not found")
        sys.exit(1)

    results = run_benchmark_suite(
        examples_dir,
        max_depth=3,
        use_llm=True,
        timeout=600
    )

    print_summary(results)

    # Exit with non-zero if any benchmarks failed
    failed = sum(1 for r in results if not r.success)
    sys.exit(1 if failed > 0 else 0)
