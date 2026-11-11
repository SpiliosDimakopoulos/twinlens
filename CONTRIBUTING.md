# Contributing to TwinLens

Thanks for looking. Bug reports, questions and small pull requests are all welcome.

## Ways to help

- **Try it on your prompt** and open an issue if a result looks wrong: `twinlens audit "your prompt"`.
- **Pick an issue** labelled [`good first issue`](https://github.com/SpiliosDimakopoulos/twinlens/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22).
- **Port the twin to another model** (Gemma-2-2B with circuit-tracer transcoders is the obvious next one).

## Set up

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"
cd web && npm ci && cd ..
pre-commit install          # optional: ruff and whitespace checks on every commit
```

`pip install -e ".[dev]"` includes the `server` extra (FastAPI and uvicorn), so the API and the
test suite work out of the box. A plain `pip install twinlens` ships only the CLI and library.

**Windows:** run `make` from Git Bash, or run the commands from the Makefile directly. Run
`git config core.autocrlf input` so line endings stay LF.

The first test run downloads GPT-2 small (0.5 GB) and its 12 transcoders (1.8 GB).

## Before you open a pull request

```bash
make lint          # ruff + eslint
make test          # Python tests (unit, reproduction, twin invariants, API)
make web-test      # type-check + vitest
```

- Keep changes focused: one fix or feature per pull request.
- New behaviour needs a test. The twin has strong invariants (every level reproduces the clean
  run, linear levels are affine); use them.
- If you change anything under `experiments/`, run `make analyze` and commit the regenerated
  `summary.json` so the report's numbers stay reproducible.
- Numbers in `paper/REPORT.md` must come from `summary.json`; do not edit them by hand.

## Reporting a bug

Include the prompt, the command or the page and the feature you clicked, what you expected and
what you saw. For the dashboard, the URL is enough: every view is a shareable link.
