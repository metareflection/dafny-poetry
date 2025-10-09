
from typing import Optional, List, Dict
import re, textwrap

# Import the LLM entrypoint expected by the user.
try:
    from llm import default_generate as generate  # type: ignore
except Exception as e:
    generate = None  # We'll check at call time.

from .prompts import BASE_PROMPT

def _between(s: str, left: str, right: str) -> Optional[str]:
    i = s.find(left)
    if i < 0: return None
    j = s.find(right, i + len(left))
    if j < 0: return None
    return s[i+len(left):j].strip("\n\r ")

def propose_new_body(method: str, errors: str, admits: str, method_body: str, tries: int = 1) -> Optional[str]:
    """Ask the LLM for a repaired body for `method`. Returns the body text or None."""
    if generate is None:
        raise RuntimeError("LLM not available: import from llm.default_generate failed. Run with --use-llm only if installed.")
    prompt = BASE_PROMPT.format(method=method, errors=errors.strip()[:4000], admits=admits.strip()[:2000], method_body=method_body.strip()[:4000])
    last = None
    for k in range(tries):
        out = generate(prompt)
        body = _between(out, "<<BEGIN_BODY>>", "<<END_BODY>>")
        if body and body.strip():
            return body.strip()
        last = out
        # Be stricter in the next attempt
        prompt = prompt + "\n\nYour previous output did not include a valid body block. Return ONLY the body between the markers."
    return None
