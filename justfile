# justfile

# Configuration
ruff_config := "~/ruffconfigs/default/ruff.toml"

# Default recipe to display help
default:
    @just --list

# Clean Python cache files and directories
clean:
    @find . -type f \( -name "*.pyc" -o -name "*.pyo" \) -delete
    @find . -type d \( -name ".mypy_cache" -o -name "__pycache__" -o -name ".ruff_cache" \) -exec rm -rf {} + 2>/dev/null || true

# Format Python files and sort requirements
format:
    #!/usr/bin/env bash
    set -euo pipefail
    if [ -n "$(find . -name "*.py" -type f)" ]; then
        ruff format --config "{{ruff_config}}" . || exit 1
    fi
    if [ -n "$(find . -name "requirements.txt" -type f)" ]; then
        for file in $(find . -name "requirements.txt" -type f); do
            sort-requirements "$file" || exit 1
        done
    fi

# Deploy git hooks from .githooks directory
deployhooks:
    #!/usr/bin/env bash
    set -euo pipefail
    if [ -d ./.githooks ]; then
        cp -f ./.githooks/* ./.git/hooks/
        chmod +x ./.git/hooks/*
    else
        exit 1
    fi

# Update pre-commit hooks
precau:
    @pre-commit autoupdate

# Run pre-commit on all files
precra:
    @pre-commit run --all-files

# Check if git repository has changes
check-git-status:
    #!/usr/bin/env bash
    if [ -z "$(git status --porcelain)" ]; then
        exit 1
    fi

# Add all changes, commit, and push (requires changes)
gitall: check-git-status
    @git add -A
    @git commit --all
    @git push

# Alias for precra
lint: precra

# Alias for format
fmt: format

# Prepare for git commit: format, update hooks, run checks, clean
gitpre: format precau precra clean

# Complete git workflow: format, update hooks, run checks, clean, and push
gitpush: format precau precra clean gitall

# Clean and format
clfmt: format clean
