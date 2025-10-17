
from typing import Optional, List, Dict
import re, textwrap
from string import Template

from .llm import default_generate as generate

from .prompts import BASE_PROMPT, SKETCH_PROMPT, SINGLE_ADMIT_PATCH_PROMPT

def _between(s: str, left: str, right: str, marker: str = "BODY") -> Optional[str]:
    i = s.find(left)
    if i < 0: return None
    j = s.find(right, i + len(left))
    if j < 0: return None
    return s[i+len(left):j].strip("\n\r ")

def generate_from_prompt(prompt: str, tries: int = 1, marker: str = "BODY") -> Optional[str]:
    last = None
    for k in range(tries):
        full_prompt = prompt + ("\n\nYour previous output did not include a valid marker block. Return ONLY the code between the markers." if k > 0 else "")
        out = generate(full_prompt)
        body = _between(out, f"<<BEGIN_{marker}>>", f"<<END_{marker}>>")
        if body and body.strip():
            return body.strip()
        last = out
    return None

def propose_new_body(method: str, errors: str, admits: str, method_body: str, file_source: str = "", tries: int = 1, sketch: bool = False) -> Optional[str]:
    """Ask the LLM for a repaired body for `method`. Returns the body text or None."""
    # Use Template to avoid needing to escape braces in code
    template = Template(BASE_PROMPT if not sketch else SKETCH_PROMPT)
    prompt = template.substitute(
        method=method,
        errors=errors.strip()[:4000],
        admits=admits.strip()[:2000],
        method_body=method_body.strip()[:4000],
        file_source=file_source.strip()[:8000]
    )
    return generate_from_prompt(prompt, tries)

def propose_patch_for_admit(
    method: str,
    errors: str,
    target_line_text: str,
    local_context_before: str,
    local_context_after: str,
    file_source: str,
    tries: int = 1,
):
    """Ask an LLM to propose a single-admit patch in a given method."""
    template = Template(SINGLE_ADMIT_PATCH_PROMPT)
    prompt = template.substitute(
        method=method,
        errors=errors.strip()[:4000],
        target_line_text=target_line_text.strip()[:2000],
        local_context_before=local_context_before.strip()[:2000],
        local_context_after=local_context_after.strip()[:2000],
        file_source=file_source.strip()[:8000]
    )
    return generate_from_prompt(prompt, tries, marker="DAFNY")