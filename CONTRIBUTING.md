# Contributing to Crop Disease Detection

Thank you for considering a contribution! This guide explains how to set up the project
locally, the branching workflow, coding standards, and how to submit changes.

---

## Table of Contents

- [Setting Up Your Environment](#setting-up-your-environment)
- [Branching Workflow](#branching-workflow)
- [Coding Standards](#coding-standards)
- [Writing Tests](#writing-tests)
- [Submitting a Pull Request](#submitting-a-pull-request)
- [Reporting Issues](#reporting-issues)

---

## Setting Up Your Environment

### Prerequisites

| Tool | Minimum Version |
|------|----------------|
| Python | 3.10 |
| pip | 23.0 |
| Git | 2.40 |

### Clone and install

```bash
git clone https://github.com/umeshpandeysh/Crop-Disease-Detection.git
cd Crop-Disease-Detection
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Install development extras

```bash
pip install flake8 pytest pytest-cov black isort
```

---

## Branching Workflow

We follow a simplified **Git Flow**:

```
main          <- production-ready code, protected
develop       <- integration branch for features
feature/<x>   <- individual feature or bug-fix branches
```

Always branch from `develop`:

```bash
git checkout develop
git pull origin develop
git checkout -b feature/my-new-feature
```

---

## Coding Standards

This project follows **PEP 8** with a line-length limit of **120 characters**.

| Tool | Command |
|------|--------|
| Format | `black src/ tests/` |
| Sort imports | `isort src/ tests/` |
| Lint | `flake8 src/ tests/ --ignore=E501,W503` |

### Docstrings

- Use **NumPy-style** docstrings for all public functions and classes.
- Every module must have a module-level docstring.

### Type Hints

- All function signatures must include parameter and return-type annotations.
- Use `from __future__ import annotations` at the top of every module.

---

## Writing Tests

All new features must be accompanied by unit tests in `tests/`.

```bash
# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=term-missing
```

Guidelines:

- Use `unittest` or `pytest` (both accepted).
- Mock filesystem operations with `unittest.mock.patch`.
- Use `tempfile` for on-disk fixtures.

---

## Submitting a Pull Request

1. Ensure all tests pass and linting is clean.
2. Write a clear PR description: *what* changed and *why*.
3. Reference any related issue (`Closes #42`).
4. Request review from at least one maintainer.
5. Squash commits before merging into `develop`.

---

## Reporting Issues

Use the [GitHub Issues](https://github.com/umeshpandeysh/Crop-Disease-Detection/issues) tracker.

When reporting a bug, include:

- Python version and OS
- Full traceback
- Minimal reproducible example
- Expected vs. actual behaviour

---

*Happy coding! 🌿*
