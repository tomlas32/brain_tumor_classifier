"""
Pytest configuration to keep logs readable during CI.

- Suppresses noisy libraries (PIL, matplotlib) down to WARNING.
- Leaves project loggers (e.g., src.training.runner) untouched so tests can assert on them.
"""
import logging
import pytest

@pytest.fixture(autouse=True)
def _reduce_log_noise():
    # Suppress common noisy libraries
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    # Don't suppress your own structured logs, so caplog assertions still work
    yield
