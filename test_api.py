#!/usr/bin/env python3
"""Quick test of the POETRY Python API."""

from pathlib import Path
from src.dafny_poetry.api import verify_dafny, verify_dafny_file

def test_string_input():
    """Test verifying from string input."""
    print("Test 1: Verifying from string...")

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

    result = verify_dafny(
        source,
        max_depth=3,
        use_llm=True,
        verbose=False,
        out_dir=Path("test_out1")
    )

    print(f"  Success: {result.success}")
    print(f"  Admits: {result.final_admits}/{result.initial_admits}")
    print(f"  Iterations: {result.iterations}")
    print(f"  Final file: {result.final_path}")

    assert result.success, f"Expected success, got failure: {result.error}"
    assert result.final_admits == 0, f"Expected 0 admits, got {result.final_admits}"
    print("  ✓ PASSED\n")


def test_file_input():
    """Test verifying from file input."""
    print("Test 2: Verifying from file...")

    result = verify_dafny_file(
        Path("examples/flatten.dfy"),
        max_depth=3,
        use_llm=True,
        verbose=False,
        out_dir=Path("test_out2")
    )

    print(f"  Success: {result.success}")
    print(f"  Admits: {result.final_admits}/{result.initial_admits}")
    print(f"  Iterations: {result.iterations}")
    print(f"  Final file: {result.final_path}")

    assert result.success, f"Expected success, got failure: {result.error}"
    assert result.final_admits == 0, f"Expected 0 admits, got {result.final_admits}"
    print("  ✓ PASSED\n")


def test_no_llm():
    """Test without LLM (should not solve but should run)."""
    print("Test 3: Without LLM (expect partial progress)...")

    source = """
    lemma {:induction false} Simple(n: nat)
      ensures n >= 0
    {
    }
    """

    result = verify_dafny(
        source,
        max_depth=2,
        use_llm=False,
        verbose=False,
        out_dir=Path("test_out3")
    )

    print(f"  Success: {result.success}")
    print(f"  Admits: {result.final_admits}/{result.initial_admits}")
    print(f"  Error: {result.error}")

    # This trivial example might succeed even without LLM
    # Just check it doesn't crash
    print("  ✓ PASSED (no crash)\n")


if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("POETRY Python API Tests")
    print("=" * 60)
    print()

    try:
        test_string_input()
        test_file_input()
        test_no_llm()

        print("=" * 60)
        print("All tests passed!")
        print("=" * 60)
        sys.exit(0)

    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
