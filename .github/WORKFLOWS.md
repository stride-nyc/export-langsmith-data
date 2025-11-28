# GitHub Actions CI/CD

This directory contains GitHub Actions workflows for continuous integration and code quality checks.

## Workflows

### 🧪 test.yml - Test Suite
**Trigger:** Push to any branch (Python changes only), PRs to main (Python changes only)

**What it does:**
- Matrix testing on Python 3.11 and 3.12
- Runs pytest with coverage
- Uploads coverage to Codecov (Python 3.11 only)

**Required:** ✅ Must pass for PR merge

---

### 🎨 lint.yml - Code Quality
**Trigger:** Push to any branch (Python changes only), PRs to main (Python changes only)

**What it does:**
- **Ruff**: Fast Python linter (checks imports, unused variables, etc.)
- **Black**: Code formatting verification (100 char line length)
- **mypy**: Static type checking

**Required:** ⚠️ Non-blocking but should be addressed

**Fix issues:**
```bash
# Auto-fix formatting
black .

# Auto-fix some linting issues
ruff check --fix .

# Check types
mypy . --ignore-missing-imports
```

---

### 🔒 security.yml - Security Scanning
**Trigger:** Push to main (Python changes only), PRs (Python changes only), Weekly (Sundays)

**What it does:**
- **Bandit**: Scans for common security issues
- **Safety**: Checks for vulnerable dependencies
- Uploads security reports as artifacts

**Required:** ⚠️ Review findings, critical issues must be fixed

---

### 📊 pr-metrics.yml - PR Analysis
**Trigger:** PR opened/updated

**What it does:**
- Analyzes PR size (small/medium/large/extra-large) for Python code
- Calculates test-to-production code ratio
- Measures commit quality:
  - Large commits (>100 lines): Target <20%
  - Sprawling commits (>5 files): Target <10%
- Posts analysis comment on PR

**Based on:** [PDCA Framework](https://github.com/kenjudy/human-ai-collaboration-process)

---

## Configuration Files

Configuration can be added to `pyproject.toml` for:
- **Black** (formatter)
- **Ruff** (linter)
- **mypy** (type checker)
- **pytest** (test runner)
- **coverage** (coverage reporting)
- **Bandit** (security scanner)

---

## Local Development

### Run all quality checks locally:

```bash
# Activate virtual environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Run tests with coverage
pytest . --cov=. --cov-report=term-missing

# Check formatting
black --check .

# Auto-format code
black .

# Lint code
ruff check .

# Auto-fix linting issues
ruff check --fix .

# Type check
mypy . --ignore-missing-imports

# Security scan
bandit -r . --exclude ./.venv,./.pytest_cache
safety check
```

### Pre-commit checks:

Before committing, run:
```bash
black . && ruff check --fix . && pytest .
```

---

## Quality Targets

### Test Coverage
- **Target:** >80%
- **Location:** All Python files

### Commit Quality
- **Large commits:** <20% (>100 production lines)
- **Sprawling commits:** <10% (>5 files changed)
- **Test ratio:** 0.5-2.0 (test lines : production lines)

### PR Size
- **Small:** <100 production lines
- **Medium:** 100-200 lines
- **Large:** 200-500 lines (harder to review)
- **Extra-large:** >500 lines (should be split)

### Code Quality
- **Complexity:** Max 10 (ruff C901)
- **Line length:** 100 characters (black)
- **Import organization:** Automatic (ruff I)

---

## Troubleshooting

### CI failures

**Tests failing:**
```bash
# Run locally with same environment
LANGSMITH_API_KEY=test-api-key pytest . -v
```

**Coverage too low:**
- Add tests for untested code
- Check coverage report: `pytest --cov=. --cov-report=html`
- Open `htmlcov/index.html` in browser

**Linting failures:**
```bash
# See what needs fixing
ruff check .

# Auto-fix
ruff check --fix .
black .
```

**Security issues:**
```bash
# See details
bandit -r . --exclude ./.venv,./.pytest_cache -f screen
safety check

# Review and fix critical issues
# Update dependencies if needed
```

---

## Badges

Add to README.md:

```markdown
![Tests](https://github.com/YOUR_USERNAME/export-langsmith-data/workflows/Tests/badge.svg)
![Lint](https://github.com/YOUR_USERNAME/export-langsmith-data/workflows/Lint/badge.svg)
![Security](https://github.com/YOUR_USERNAME/export-langsmith-data/workflows/Security/badge.svg)
[![codecov](https://codecov.io/gh/YOUR_USERNAME/export-langsmith-data/branch/main/graph/badge.svg)](https://codecov.io/gh/YOUR_USERNAME/export-langsmith-data)
```

---

## References

- [PDCA Framework for AI-Assisted Code Generation](https://github.com/kenjudy/human-ai-collaboration-process)
- [Ruff Documentation](https://docs.astral.sh/ruff/)
- [Black Documentation](https://black.readthedocs.io/)
- [pytest Documentation](https://docs.pytest.org/)
