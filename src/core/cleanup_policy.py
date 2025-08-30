"""
Policies and defaults for the cleanup stage.

Defines:
- STRICT_ERROR_CODES: codes acted on in strict mode
- NEVER_AUTO_MOVE: codes that are never moved automatically
- DEFAULT_POLICY: default cleanup policy
- DEFAULT_ACT_ON: default severity scope
"""

# Error codes we will act on by default under --policy strict
STRICT_ERROR_CODES = {
    "UNREADABLE", "READ_FAIL", "STAT_FAIL", "TINY_FILE",
    "NOT_RGB", "BAD_SIZE", "ALL_BLACK", "ALL_WHITE",
    "DUPLICATE","NEAR_DUP_PHASH",
}

# Codes we never auto-move (contract/mapping issues)
NEVER_AUTO_MOVE = {"BAD_LABEL"}

DEFAULT_POLICY = "strict"      # strict | within_class | report_only
DEFAULT_ACT_ON = "both"      # errors | warnings | both