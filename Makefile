# Virtual environment activation
VENV_ACTIVATE = . .venv/bin/activate

.PHONY: help install install-dev \
        format format-check \
        lint lint-fix \
        type-check \
        fix check test clean pre-commit \
        all

help:
	@echo "Available commands:"
	@echo "  make install       - Install production dependencies"
	@echo "  make install-dev   - Install all dependencies including dev tools"
	@echo "  make format        - Format code with ruff"
	@echo "  make format-check  - Verify formatting (non-mutating)"
	@echo "  make lint          - Run ruff linting (non-mutating)"
	@echo "  make lint-fix      - Run ruff linting with auto-fix"
	@echo "  make type-check    - Run mypy type checking"
	@echo "  make check         - Run all checks (format-check, lint, type-check)"
	@echo "  make fix           - Auto-apply formatting and lint fixes"
	@echo "  make test          - Run test script"
	@echo "  make clean         - Clean up cache and build files"
	@echo "  make pre-commit    - Install pre-commit hooks"

all: check test
	@echo "All targets completed successfully!"

install:
	uv sync

install-dev:
	uv sync --extra dev

format:
	$(VENV_ACTIVATE); ruff format src/

format-check:
	$(VENV_ACTIVATE); ruff format src/ --check

lint:
	$(VENV_ACTIVATE); ruff check src/

lint-fix:
	$(VENV_ACTIVATE); ruff check src/ --fix

type-check:
	$(VENV_ACTIVATE); mypy src/

fix: format lint-fix
	@echo "All fixes applied!"

check: format-check lint type-check
	@echo "All checks passed!"

test:
	bash test_cli.sh

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.coverage" -delete
	find . -type d -name "htmlcov" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +

pre-commit:
	$(VENV_ACTIVATE); pre-commit install
	$(VENV_ACTIVATE); pre-commit run --all-files
