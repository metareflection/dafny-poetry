
import subprocess, pathlib, re, tempfile, shutil, json, os
from typing import List, Tuple, Optional, Dict
from .utils import find_enclosing_decl, extract_method_body_region

def _run(cmd: list, cwd=None, check=True, capture_output=True, text=True, timeout=None):
    p = subprocess.run(cmd, cwd=cwd, check=False, capture_output=capture_output, text=text, timeout=timeout)
    if check and p.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\nstdout:\n{p.stdout}\nstderr:\n{p.stderr}")
    return p

def run_dafny_admitter(dfy_path: pathlib.Path, mode: str="admit", only_failing: bool=True, timeout: Optional[int]=None) -> pathlib.Path:
    """Return path to patched .dfy (same folder with .patched.dfy suffix)."""
    out = _run(
        ["dafny-admitter", str(dfy_path), "--mode", mode] + ([] if only_failing else ["--no-only-failing"]),
        timeout=timeout, check=False
    )
    # The admitter writes a *.patched.dfy next to the input
    # Handle case where input already has .patched in the name
    if dfy_path.stem.endswith('.patched'):
        # Input is like "foo.patched.dfy" -> output is "foo.patched.patched.dfy"
        patched = dfy_path.parent / (dfy_path.stem + ".patched.dfy")
    else:
        # Input is like "foo.dfy" -> output is "foo.patched.dfy"
        patched = dfy_path.with_suffix(".patched.dfy")

    if not patched.exists():
        # If admitter supports stdout mode: detect and write
        if out.stdout.strip():
            patched.write_text(out.stdout)
        else:
            raise RuntimeError(f"dafny-admitter did not produce a patched file. Expected: {patched}\nstdout: {out.stdout}\nstderr: {out.stderr}")
    return patched

ADMIT_RE = re.compile(r'\bAdmit\s*\(')

def count_admits(dfy_path: pathlib.Path) -> int:
    try:
        src = dfy_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return 0
    return len(ADMIT_RE.findall(src))

def collect_first_admit_context(dfy_path: pathlib.Path) -> Optional[Dict]:
    """Return info for the first Admit occurrence: line, col, method name, body region."""
    src = dfy_path.read_text(encoding="utf-8", errors="ignore")
    lines = src.splitlines()
    for i, line in enumerate(lines, start=1):
        m = re.search(r'\bAdmit\s*\(\s*"(.*?)"', line)
        if m:
            method = find_enclosing_decl(src, i)
            body_span = extract_method_body_region(src, i)
            return {"line": i, "tag": m.group(1), "method": method, "body_span": body_span}
    return None

def run_sketcher(dfy_path: pathlib.Path, sketch: str, method: Optional[str]=None, timeout: Optional[int]=None) -> str:
    """Return stdout (even on failure)."""
    cmd = ["dafny-sketcher-cli", "--file", str(dfy_path), "--sketch", sketch]
    if method:
        cmd += ["--method", method]
    p = _run(cmd, check=False, timeout=timeout)
    return p.stdout + ("\n" + p.stderr if p.stderr else "")

def write_version(out_dir: pathlib.Path, base: pathlib.Path, label: str, text: str) -> pathlib.Path:
    cand = out_dir / f"{base.stem}.{label}.dfy"
    cand.write_text(text, encoding="utf-8")
    return cand

def replace_method_body(dfy_text: str, method_name: str, new_body: str) -> str:
    """Replace the *body* (between braces) of the given method/lemma/function by new_body.
       Returns new source or raises ValueError.
    """
    start_line, end_line, body_l, body_r = extract_method_body_region(dfy_text, None, method_name=method_name)
    if start_line is None:
        raise ValueError(f"Cannot locate body of method {method_name}")
    lines = dfy_text.splitlines()
    # Keep header and closing brace, replace the middle region [body_l+1 : body_r]
    new_lines = lines[:body_l+1] + [new_body.rstrip("\n")] + lines[body_r:]
    return "\n".join(new_lines)
