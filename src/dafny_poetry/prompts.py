
BASE_PROMPT = """You are a Dafny proof repair assistant inside a recursive POETRY-style loop.
Goal: reduce the number of outstanding obligations in the current file by editing ONLY the **body**
of the method/lemma `$method`. Keep the signature and name unchanged.

Context:
--- full file source (for reference) ---
$file_source
--- errors/warnings ---
$errors
--- admits in this method ---
$admits
--- method body (current) ---
$method_body

Guidance:
- Prefer small, local edits (assertions, helper calls, lemma invocations, simple case splits).
- Do **not** mark anything `:axiom` and do **not** add new global declarations unless strictly necessary.
- If induction is needed, set up a clean structural or rule induction *inside the body*.
- Ensure the program remains syntactically valid Dafny.
- Avoid introducing new unattached `Admit(...)`; the point is to **reduce** total admits.
- If you add an `assert`, add a short `by {proof}` block if needed.
- Use existing helper lemmas when available.

Return ONLY the new body content between the markers below (no surrounding braces, no extra prose):

<<BEGIN_BODY>>
... your revised body statements ...
<<END_BODY>>
"""

SKETCH_PROMPT = """You are a Dafny proof repair assistant inside a recursive POETRY-style loop.
Goal: Sketch ONLY the **body** of the method/lemma `$method`, so that further refinements can resolve all outstanding issues.
Keep the signature and name unchanged.

Context:
--- full file source (for reference) ---
$file_source
--- errors/warnings ---
$errors
--- admits in this method ---
$admits
--- method body (current) ---
$method_body

Guidance:
- Do **not** mark anything `:axiom` and do **not** add new global declarations unless strictly necessary.
- If induction is needed, set up a clean structural or rule induction *inside the body*.
- Ensure the program remains syntactically valid Dafny.
- Avoid introducing new unattached `Admit(...)`; the point is to **reduce** total admits.
- If you add an `assert`, add a short `by {proof}` block if needed.
- Use existing helper lemmas when available.

Return ONLY the new body content between the markers below (no surrounding braces, no extra prose):

<<BEGIN_BODY>>
... your revised body statements ...
<<END_BODY>>
"""
