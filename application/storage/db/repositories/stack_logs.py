"""Repository for the ``stack_logs`` table.

Covers the single operation the legacy Mongo code performs:

1. ``insert_one`` in logging.py ``_log_to_mongodb`` — append-only debug/error
   activity log. The Mongo collection is ``stack_logs``; the Mongo variable
   inside ``_log_to_mongodb`` is misleadingly named ``user_logs_collection``.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Optional

from application.storage.db.redaction import redact_secrets
from application.storage.db.serialization import PGNativeJSONEncoder
from application.utils import strip_null_bytes

from sqlalchemy import Connection, text

# Longest string kept inside ``stacks``. The scalar columns are truncated
# by the caller, but stacks used to go in whole — one uncapped tool
# result (634k tokens, 07-17) rode into the activity log through here.
_STACKS_STRING_MAX_LEN = 10000


def _bound_strings(value):
    """Recursively truncate strings in ``value`` to ``_STACKS_STRING_MAX_LEN``."""
    if isinstance(value, str):
        if len(value) <= _STACKS_STRING_MAX_LEN:
            return value
        return value[:_STACKS_STRING_MAX_LEN] + "...[truncated]"
    if isinstance(value, dict):
        return {k: _bound_strings(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_bound_strings(item) for item in value]
    return value


class StackLogsRepository:
    """Postgres-backed replacement for Mongo ``stack_logs`` collection."""

    def __init__(self, conn: Connection) -> None:
        self._conn = conn

    def insert(
        self,
        *,
        activity_id: str,
        endpoint: Optional[str] = None,
        level: Optional[str] = None,
        user_id: Optional[str] = None,
        api_key: Optional[str] = None,
        query: Optional[str] = None,
        stacks: Optional[list] = None,
        timestamp: Optional[datetime] = None,
    ) -> None:
        self._conn.execute(
            text(
                """
                INSERT INTO stack_logs (activity_id, endpoint, level, user_id, api_key, query, stacks, timestamp)
                VALUES (
                    :activity_id, :endpoint, :level, :user_id, :api_key, :query,
                    CAST(:stacks AS jsonb),
                    COALESCE(:timestamp, now())
                )
                """
            ),
            {
                "activity_id": activity_id,
                "endpoint": strip_null_bytes(endpoint),
                "level": strip_null_bytes(level),
                "user_id": strip_null_bytes(user_id),
                "api_key": strip_null_bytes(api_key),
                "query": strip_null_bytes(query),
                "stacks": json.dumps(
                    redact_secrets(
                        _bound_strings(strip_null_bytes(stacks or []))
                    ),
                    cls=PGNativeJSONEncoder,
                ),
                "timestamp": timestamp,
            },
        )
