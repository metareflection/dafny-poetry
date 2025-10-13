
import re
from typing import Tuple, Optional, List

DECL_RE = re.compile(r'^\s*(method|lemma|function)\s+(\{[^}]*\}\s+)?([A-Za-z_][A-Za-z0-9_]*)(<[^>]+>)?\s*\(')

def _find_body_opening_brace(lines: List[str], start_line: int) -> Optional[int]:
    """
    Find the opening brace of a method/lemma/function body, starting from start_line.
    Properly skips {:attribute} braces and handles braces anywhere in the line.
    Returns the line number containing the opening brace, or None if not found.
    """
    for j in range(start_line, len(lines)):
        line = lines[j]
        # Skip {:attribute} style braces, find the actual body opening brace
        pos = 0
        while pos < len(line):
            if line[pos:pos+2] == '{:':
                # Skip attribute - find its closing }
                close_pos = line.find('}', pos + 2)
                pos = close_pos + 1 if close_pos >= 0 else pos + 2
            elif line[pos] == '{':
                # Found the opening brace!
                return j
            else:
                pos += 1
    return None

def _find_matching_closing_brace(lines: List[str], brace_line: int) -> Optional[int]:
    """
    Find the closing brace that matches the opening brace at brace_line.
    Uses depth counting to handle nested braces.
    Returns the line number containing the closing brace, or None if not found.
    """
    depth = 0
    for i in range(brace_line, len(lines)):
        for ch in lines[i]:
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    return i
    return None

def find_enclosing_decl(src: str, line_no: int) -> Optional[str]:
    """Return name of the method/lemma/function that encloses line_no (i.e., line_no is inside its body)."""
    lines = src.splitlines()
    # Find all declarations and their body ranges (inline to avoid circular dependency)
    candidates = []
    for i, line in enumerate(lines):
        m = DECL_RE.match(line)
        if m:
            method_name = m.group(3)  # group 3 is the method name (group 2 is optional attributes)
            # Find body range
            brace_line = _find_body_opening_brace(lines, i)
            if brace_line is None:
                continue
            end_line = _find_matching_closing_brace(lines, brace_line)
            if end_line is None:
                continue
            # Check if line_no-1 (0-based) is within the body range [brace_line, end_line]
            if brace_line <= line_no - 1 <= end_line:
                candidates.append((method_name, i))

    # Return the innermost (latest in file) enclosing declaration
    if candidates:
        return candidates[-1][0]
    return None

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
    # Find opening brace and matching closing brace
    brace_line = _find_body_opening_brace(lines, decl_line)
    if brace_line is None:
        return (None, None, None, None)
    end_line = _find_matching_closing_brace(lines, brace_line)
    if end_line is None:
        return (None, None, None, None)
    return (decl_line, end_line, brace_line, end_line)

def extract_method_body_text(src: str, method_name: str) -> str:
    """Extract the body text content (between braces) of a method/lemma/function.
       Returns just the body content, excluding the braces themselves.
    """
    start_line, end_line, body_l, body_r = extract_method_body_region(src, None, method_name=method_name)
    if body_l is None:
        return ""
    
    lines = src.splitlines()
    
    # Extract lines between opening and closing braces (inclusive)
    # body_l is the line with '{', body_r is the line with '}'
    if body_l == body_r:
        # Single line: { body }
        line = lines[body_l]
        open_pos = line.find('{')
        close_pos = line.rfind('}')
        if open_pos >= 0 and close_pos > open_pos:
            return line[open_pos+1:close_pos].strip()
        return ""
    
    # Multi-line case
    result_lines = []
    
    # First line: everything after '{'
    first = lines[body_l]
    open_pos = first.find('{')
    if open_pos >= 0 and open_pos < len(first) - 1:
        after_brace = first[open_pos+1:].rstrip()
        if after_brace:
            result_lines.append(after_brace)
    
    # Middle lines: take as-is
    if body_r > body_l + 1:
        result_lines.extend(lines[body_l+1:body_r])
    
    # Last line: everything before '}'
    if body_r > body_l:
        last = lines[body_r]
        close_pos = last.rfind('}')
        if close_pos > 0:
            before_brace = last[:close_pos].rstrip()
            if before_brace:
                result_lines.append(before_brace)
    
    return "\n".join(result_lines)

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
    # Find opening brace
    brace_line = _find_body_opening_brace(lines, decl_line)
    if brace_line is None:
        return ""
    # Return all lines from declaration to opening brace (inclusive)
    decl_lines = lines[decl_line:brace_line+1]
    return "\n".join(decl_lines)
