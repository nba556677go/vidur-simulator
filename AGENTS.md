# Vidur Agent Guidelines

## Overview
This repository contains a Python 3.10 project for the Vidur LLM inference simulator.
The main source lives in `vidur/` and tests are in `tests/`.

## Environment Setup
- Use **Python 3.10**.  The version is pinned in `.python-version`.
- Install dependencies using `pip install -r requirements.txt` (or `uv sync` if available).

## Formatting and Linting
- Run `make format` after modifying Python files.  This runs **isort** and **black** over the `vidur/` package.
- Optional: `make lint` runs the style checks (`flake8`, `black --check`, `isort --check-only`).

## Testing
- Execute `pytest` from the repository root.  Tests are located under `tests/`.
- GPU/FlashInfer specific tests will be skipped automatically if the required hardware or packages are missing, but the command should still be run.

## Documentation
- Update files in `docs/` when user-facing behavior changes.

## Commit and PR
- Ensure formatting and tests succeed before committing.
- Keep commit messages descriptive.  Refer to `.github/PULL_REQUEST_TEMPLATE.md` for PR expectations.

