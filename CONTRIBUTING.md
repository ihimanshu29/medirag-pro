# Contributing to MediRAG Pro

Thank you for your interest in contributing. This document covers everything you need to go from zero to a merged pull request.

---

## Table of Contents
- [Development Setup](#development-setup)
- [Running Tests](#running-tests)
- [Code Standards](#code-standards)
- [Submitting a PR](#submitting-a-pr)
- [Commit Message Format](#commit-message-format)
- [Architecture Overview](#architecture-overview)

---

## Development Setup

### Prerequisites
- Python 3.11+
- Docker + Docker Compose
- A Groq API key (free at https://console.groq.com)

### Steps

```bash
# 1. Fork and clone
git clone https://github.com/yourusername/medirag-pro.git
cd medirag-pro

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 3. Install dependencies (CPU-only PyTorch)
pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

# 4. Install pre-commit hooks
pip install pre-commit
pre-commit install

# 5. Configure environment
cp .env.example .env
# Edit .env and add GROQ_API_KEY

# 6. Start infrastructure
docker compose up qdrant postgres -d

# 7. Verify everything works
GROQ_API_KEY=test-key pytest tests/ -q
# Expected: 82 passed
```

---

## Running Tests

```bash
# All tests (fast — all external deps mocked)
pytest tests/ -v

# Single phase
pytest tests/test_phase3.py -v

# With coverage report
pytest tests/ --cov=src --cov-report=html
open htmlcov/index.html

# Integration tests (requires running Qdrant + Postgres)
pytest tests/integration/ -v -m integration
```

**Coverage requirement:** New code must maintain ≥ 70% line coverage.

---

## Code Standards

### Before every commit, these must pass:

```bash
# Lint
ruff check src/ tests/

# Format
ruff format src/ tests/

# Type check
mypy src/ --ignore-missing-imports

# Tests
pytest tests/ -q
```

If you installed pre-commit hooks (`pre-commit install`), these run automatically on `git commit`.

### Style rules
- Line length: 100 characters
- Quotes: double
- All public functions must have docstrings
- All public functions must have type hints on parameters and return value
- No `print()` statements — use `from src.logging_config import get_logger`
- No bare `except:` — catch specific exception types

### Adding a new component

If you add a new module (e.g., `src/retrieval/new_thing.py`):
1. Add `__init__.py` if creating a new package
2. Add unit tests in `tests/test_new_thing.py`
3. Add a docstring explaining the component's purpose and design decisions
4. Update `CHANGELOG.md` under `[Unreleased]`

---

## Submitting a PR

### PR checklist
- [ ] `ruff check src/ tests/` passes
- [ ] `ruff format --check src/ tests/` passes
- [ ] `mypy src/ --ignore-missing-imports` passes
- [ ] `pytest tests/ -q` passes (all 82+ tests green)
- [ ] Coverage does not drop below 70%
- [ ] New functionality has tests
- [ ] Docstring added to all new public functions
- [ ] `CHANGELOG.md` updated under `[Unreleased]`
- [ ] PR description explains what changed and why

### PR description template

```
## What does this PR do?
<!-- One paragraph summary -->

## Why?
<!-- Motivation: what problem does this solve? -->

## How?
<!-- Key implementation decisions -->

## Testing
<!-- How did you test this? What edge cases did you consider? -->

## Breaking changes
<!-- Does this break any existing API or behaviour? -->
```

---

## Commit Message Format

```
type(scope): short description (max 72 chars)

Optional longer body explaining motivation and implementation decisions.
```

**Types:**
| Type | When to use |
|---|---|
| `feat` | New feature or capability |
| `fix` | Bug fix |
| `perf` | Performance improvement |
| `refactor` | Code restructure (no behaviour change) |
| `test` | Adding or fixing tests |
| `docs` | Documentation only |
| `ci` | CI/CD pipeline changes |
| `chore` | Dependency updates, build changes |

**Examples:**
```
feat(retrieval): add async Qdrant client for parallel dense+BM25 search
fix(bm25): handle negative IDF scores on small corpora with min-max normalisation
perf(cache): replace linear cosine scan with FAISS index for O(log n) lookup
test(guardrails): add regression tests for indirect emergency phrasing patterns
docs(readme): add latency benchmark table for 10 concurrent users
```

---

## Architecture Overview

For a full architecture diagram, see [docs/architecture.md](docs/architecture.md).

Key design decisions that should not be changed without discussion:
- **Parent-child chunking** — changing chunk sizes requires re-running the full retrieval ablation
- **RRF fusion** — replacing with weighted fusion requires per-dataset tuning
- **Pre-retrieval emergency detection** — must remain hardcoded, never LLM-routed
- **Idempotent ingestion** — delete-before-upsert is intentional, not a bug

---

## Good First Issues

Look for issues labelled [`good first issue`](https://github.com/yourusername/medirag-pro/labels/good%20first%20issue) on GitHub.

Current good first issues:
- Add Spanish-language emergency detection keywords
- Add DOCX table extraction via `python-docx`
- Write integration test for full ingest → query cycle
- Add Grafana dashboard JSON export

---

## Questions?

Open a [GitHub Discussion](https://github.com/yourusername/medirag-pro/discussions) for design questions.
Open a [GitHub Issue](https://github.com/yourusername/medirag-pro/issues) for bugs or feature requests.
