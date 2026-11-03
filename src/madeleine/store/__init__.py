"""Storage backends: SQLite (+ sqlite-vec) for real use, in-memory for tests."""

from madeleine.store.memory import InMemoryRecollectionQueue, InMemoryStore
from madeleine.store.sqlite import (
    SQLiteDatabase,
    SQLiteMemoryStore,
    SQLiteRecollectionQueue,
    open_sqlite,
)

__all__ = [
    "InMemoryRecollectionQueue",
    "InMemoryStore",
    "SQLiteDatabase",
    "SQLiteMemoryStore",
    "SQLiteRecollectionQueue",
    "open_sqlite",
]
