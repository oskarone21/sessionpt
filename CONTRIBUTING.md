# Contributing to sessionpt

Thank you for your interest in contributing! This guide covers the basics.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install
```

## Development

- **Formatting & linting:** `ruff check .` and `ruff format .`
- **Type checking:** `mypy src/`
- **Tests:** `pytest`

All three are run in CI and must pass before merge.

## Pull Requests

1. Fork the repository and create a feature branch.
2. Make your changes with clear, focused commits.
3. Ensure all checks pass (`ruff`, `mypy`, `pytest`).
4. Open a pull request against `main` with a description of the change.

## Code Style

- Follow PEP 8 (enforced by ruff).
- Use type hints for all public APIs.
- Keep functions small and focused.
- Write tests for new functionality.

## Reporting Issues

- Use GitHub Issues for bug reports and feature requests.
- Include a minimal reproducible example when possible.

## License

By contributing, you agree that your contributions will be licensed under the same license as the project.
