
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

SINGLE_ADMIT_PATCH_PROMPT = """You are editing Dafny. Replace only the TARGET statement with code that advances the proof in this scope.
Do not modify any other lines. Do not change the method signature/specs. 
Do not use 'assume' or 'reveal'. You may add a small block, 'assert', 'calc { ... }', 
and calls to existing lemmas. Output only the replacement code.

Context:
--- full file source (for reference) ---
$file_source
--- errors/warnings ---
$errors
--- method/lemma ---
$method
--- target line text ---
$target_line_text
--- local context before ---
$local_context_before
--- local context after ---
$local_context_after

Return ONLY the new Dafny code between the markers below:

<<BEGIN_DAFNY>>
... your Dafny code ...
<<END_DAFNY>>
"""
