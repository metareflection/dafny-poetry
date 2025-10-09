
import re
from typing import Tuple, Optional

DECL_RE = re.compile(r'^\s*(method|lemma|function)\s+(\{[^}]*\}\s+)?([A-Za-z_][A-Za-z0-9_]*)(<[^>]+>)?\s*\(')

def find_enclosing_decl(src: str, line_no: int) -> Optional[str]:
    """Return name of the method/lemma/function that encloses line_no (i.e., line_no is inside its body)."""
    lines = src.splitlines()
    # Find all declarations and their body ranges (inline to avoid circular dependency)
    candidates = []
    for i, line in enumerate(lines):
        m = DECL_RE.match(line)
        if m:
            method_name = m.group(3)  # group 3 is the method name (group 2 is optional attributes)
            # Find body range inline
            # Find first '{' after declaration that starts a line (body opening brace)
            # Skip braces in attributes like {:induction false}
            brace_line = None
            for j in range(i, len(lines)):
                stripped = lines[j].strip()
                # Look for a line that starts with '{' (possibly after whitespace)
                # This is the body opening brace, not an attribute
                if stripped.startswith('{') and not stripped.startswith('{:'):
                    brace_line = j
                    break
            if brace_line is None:
                continue
            # Find matching '}'
            depth = 0
            end_line = -1
            for k in range(brace_line, len(lines)):
                for ch in lines[k]:
                    if ch == '{':
                        depth += 1
                    elif ch == '}':
                        depth -= 1
                        if depth == 0:
                            end_line = k
                            break
                if end_line >= 0:
                    break
            if end_line < 0:
                continue
            # Check if line_no-1 (0-based) is within the body range [brace_line, end_line]
            if brace_line <= line_no - 1 <= end_line:
                candidates.append((method_name, i))

    # Return the innermost (latest in file) enclosing declaration
    if candidates:
        return candidates[-1][0]
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
    # Find declaration line (handle optional attributes and type parameters)
    decl_line = None
    for i, ln in enumerate(lines):
        if re.match(rf'^\s*(method|lemma|function)\s+(\{{[^}}]*\}}\s+)?{re.escape(method_name)}(<[^>]+>)?\s*\(', ln):
            decl_line = i
            break
    if decl_line is None:
        return (None, None, None, None)
    # Find first '{' after decl_line that starts a line (body opening brace, not attribute)
    brace_line = None
    for j in range(decl_line, len(lines)):
        stripped = lines[j].strip()
        if stripped.startswith('{') and not stripped.startswith('{:'):
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

def extract_method_declaration(src: str, method_name: str) -> str:
    """Extract the full declaration (signature + contracts) of a method/lemma/function, up to but not including the body."""
    lines = src.splitlines()
    # Find declaration line (handle optional attributes and type parameters)
    decl_line = None
    for i, ln in enumerate(lines):
        if re.match(rf'^\s*(method|lemma|function)\s+(\{{[^}}]*\}}\s+)?{re.escape(method_name)}(<[^>]+>)?\s*\(', ln):
            decl_line = i
            break
    if decl_line is None:
        return ""
    # Find first '{' after decl_line that starts a line (body opening brace, not attribute)
    brace_line = None
    for j in range(decl_line, len(lines)):
        stripped = lines[j].strip()
        if stripped.startswith('{') and not stripped.startswith('{:'):
            brace_line = j
            break
    if brace_line is None:
        return ""
    # Return all lines from declaration to opening brace (inclusive)
    decl_lines = lines[decl_line:brace_line+1]
    return "\n".join(decl_lines)
