"""
Pytest configuration to keep logs readable during CI.
"""
import logging
import pytest

@pytest.fixture(autouse=True)
def _reduce_log_noise():
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
