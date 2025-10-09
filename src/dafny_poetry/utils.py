
import re
from typing import Tuple, Optional

DECL_RE = re.compile(r'^\s*(method|lemma|function)\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(')

def find_enclosing_decl(src: str, line_no: int) -> Optional[str]:
    """Return name of the closest method/lemma/function above line_no."""
    lines = src.splitlines()
    i = min(max(line_no-1, 0), len(lines)-1)
    while i >= 0:
        m = DECL_RE.match(lines[i])
        if m:
            return m.group(2)
        i -= 1
    return None

def _find_matching_brace(lines, start_idx: int) -> int:
    depth = 0
    for i in range(start_idx, len(lines)):
        for ch in lines[i]:
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    return i
    return -1

def extract_method_body_region(src: str, line_no: Optional[int]=None, method_name: Optional[str]=None):
    """Return (header_line_idx, end_line_idx, body_start_idx, body_end_idx) [0-based]
       where body region in lines is (body_start_idx, body_end_idx) with body_start_idx pointing to the '{' line,
       and body_end_idx pointing to the closing '}' line.
    """
    lines = src.splitlines()
    # Decide which decl to locate
    if method_name is None:
        method_name = find_enclosing_decl(src, line_no or 1)
    if method_name is None:
        return (None, None, None, None)
    # Find declaration line
    decl_line = None
    for i, ln in enumerate(lines):
        if re.match(rf'^\s*(method|lemma|function)\s+{re.escape(method_name)}\s*\(', ln):
            decl_line = i
            break
    if decl_line is None:
        return (None, None, None, None)
    # Find first '{' after decl_line
    brace_line = None
    for j in range(decl_line, len(lines)):
        if '{' in lines[j]:
            brace_line = j
            break
    if brace_line is None:
        return (None, None, None, None)
    # Find matching '}' line
    depth = 0
    end_line = -1
    for i in range(brace_line, len(lines)):
        for ch in lines[i]:
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    end_line = i
                    break
        if end_line >= 0:
            break
    if end_line < 0:
        return (None, None, None, None)
    return (decl_line, end_line, brace_line, end_line)
