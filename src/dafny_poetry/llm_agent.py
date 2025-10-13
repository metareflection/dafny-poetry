
from typing import Optional, List, Dict
import re, textwrap

from .llm import default_generate as generate

from .prompts import BASE_PROMPT

def _between(s: str, left: str, right: str) -> Optional[str]:
    i = s.find(left)
    if i < 0: return None
    j = s.find(right, i + len(left))
    if j < 0: return None
    return s[i+len(left):j].strip("\n\r ")

def propose_new_body(method: str, errors: str, admits: str, method_body: str, file_source: str = "", tries: int = 1) -> Optional[str]:
    """Ask the LLM for a repaired body for `method`. Returns the body text or None."""
    # Escape curly braces in the content to avoid format string errors
    errors_safe = errors.strip()[:4000].replace('{', '{{').replace('}', '}}')
    admits_safe = admits.strip()[:2000].replace('{', '{{').replace('}', '}}')
    method_body_safe = method_body.strip()[:4000].replace('{', '{{').replace('}', '}}')
    file_source_safe = file_source.strip()[:8000].replace('{', '{{').replace('}', '}}')
    prompt = BASE_PROMPT.format(method=method, errors=errors_safe, admits=admits_safe, method_body=method_body_safe, file_source=file_source_safe)
    last = None
    for k in range(tries):
        full_prompt = prompt + "\n\nYour previous output did not include a valid body block. Return ONLY the body between the markers."
        out = generate(full_prompt)
        body = _between(out, "<<BEGIN_BODY>>", "<<END_BODY>>")
        if body and body.strip():
            return body.strip()
        last = out
    return None
