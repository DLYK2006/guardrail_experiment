"""Standalone entry point for the VA experiment; no legacy runner import."""

from pathlib import Path

from va_experiment.env_file import load_env_file
from va_experiment.experiment import main


if __name__ == "__main__":
    # Credentials may live in the Git-ignored .env beside this file; variables
    # already set in the terminal take precedence.
    load_env_file(Path(__file__).resolve().parent / ".env")
    raise SystemExit(main())
