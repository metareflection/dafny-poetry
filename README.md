# dafny-poetry

A small orchestrator that runs a POETRY‑style loop over Dafny programs.
It depends on:

- `dafny-admitter` – to turn outstanding verification failures into `Admit("…", φ)` obligations (verifiable sketches)
- `dafny-sketcher-cli` – to generate inductive skeletons, counterexamples, and to query errors/warnings

## Install (editable)

```bash
python -m pip install -e ./dafny-poetry
```

## CLI

```bash
dafny-poetry --file path/to/FILE.dfy --out-dir poetry_out -v   --max-depth 3 --max-branches 2 --global-timeout 600   --use-llm --llm-tries 2
```

- If `--use-llm` is set, `dafny-poetry` will import `from llm import default_generate as generate`
  and use it to propose local, minimal edits to the *body* of the lemma/method that contains the
  next `Admit(...)` obligation. The LLM is asked to return a replacement **body** between
  `<<BEGIN_BODY>>` and `<<END_BODY>>`; the signature/header must remain identical.

The loop maintains **anytime verifiability**: after each action, it re-runs `dafny-admitter` to ensure
the file verifies with `Admit(...)` placeholders and to measure progress (# of admits).

## Notes

- This orchestrator keeps things simple and deterministic; it's easy to extend with more actions.
- It is agnostic to the `dafny-admitter` Boogie/Dafny internals; it only reads the produced admits.
- When `--use-llm` is off or the `llm` module is unavailable, it runs purely symbolically.
