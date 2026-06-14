# Contributing to DeltaForge

Thank you for your interest in contributing to DeltaForge! This document provides guidelines for contributing to the project.

## Development Setup

```bash
# Clone and install
git clone https://github.com/your-username/deltaforge.git
cd deltaforge
make install

# Configure (optional -- needed for broker/alert tests)
cp .env.example .env
# Edit .env with test credentials

# Run tests
make test
```

## Code Style

- Python 3.9+ compatible
- Imports at the top of every file (no inline imports)
- Type hints on public function signatures
- Docstrings on modules and public classes/functions
- Use `loguru` for logging (not `print()` in library code)

## Making Changes

1. Fork the repository and create a feature branch from `main`.
2. Write or update tests for your changes.
3. Ensure `make test` passes.
4. Keep commits focused -- one logical change per commit.
5. Use [conventional commits](https://www.conventionalcommits.org/): `feat:`, `fix:`, `chore:`, `docs:`, `refactor:`, `test:`.

## Pull Request Process

1. Update documentation if your change affects user-facing behavior.
2. Add your changes to the [CHANGELOG](CHANGELOG.md) under `Unreleased`.
3. Ensure all tests pass and there are no lint errors.
4. Describe **what** and **why** in the PR description.

## Testing

```bash
make test               # Full suite
make test-e2e           # End-to-end tests
make test-component     # Component tests
make test-dashboard     # Dashboard API tests
```

Tests use `pytest` and mock all external dependencies (broker API, alert channels). No real API calls are made during testing.

## Reporting Issues

- Use GitHub Issues for bug reports and feature requests.
- Include steps to reproduce for bugs.
- For security vulnerabilities, see [SECURITY.md](SECURITY.md).

## Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating, you agree to uphold this code.
