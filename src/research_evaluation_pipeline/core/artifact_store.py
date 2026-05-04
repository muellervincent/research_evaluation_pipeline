"""
SQLite-backed Artifact Store for resuming pipeline executions and caching results.
"""

import json
import sqlite3
from pathlib import Path
from typing import Any

from loguru import logger


class ArtifactStore:
    """
    Persists pipeline artifacts (JSON) to a SQLite database.

    Contains two tables:
    1. artifacts: The 'active' cache for pipeline resumption.
    2. runs: The immutable history of all executed runs.
    """

    def __init__(self, database_path: Path = Path("resources/artifacts.db")):
        """
        Initialize the artifact store with a specific database path.

        Args:
            database_path: Absolute or relative path to the SQLite database file.
        """
        self.database_path = database_path
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()

    def _init_database(self):
        """
        Initialize the SQLite database schema if it does not already exist.

        Creates the 'artifacts' table for caching and the 'runs' table for history.
        """
        with sqlite3.connect(self.database_path) as connection:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS artifacts (
                    key TEXT PRIMARY KEY,
                    content TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            connection.execute("""
                CREATE TABLE IF NOT EXISTS runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    key TEXT,
                    content TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            connection.commit()

    def get_artifact(self, key: str) -> Any | None:
        """
        Retrieve an artifact by its deterministic key from the active cache.

        Args:
            key: The unique string identifier for the cached artifact.

        Returns:
            The parsed JSON content of the artifact if found, otherwise None.
        """
        with sqlite3.connect(self.database_path) as connection:
            cursor = connection.execute("SELECT content FROM artifacts WHERE key = ?", (key,))
            row = cursor.fetchone()
            if row:
                logger.trace(f"Artifact cache hit for key: {key}")
                return json.loads(row[0])
            return None

    def save_artifact(self, key: str, content: Any):
        """
        Save a JSON-serializable artifact to the active cache.

        Args:
            key: The unique string identifier for the artifact.
            content: A JSON-serializable object to store.
        """
        with sqlite3.connect(self.database_path) as connection:
            connection.execute(
                "INSERT OR REPLACE INTO artifacts (key, content) VALUES (?, ?)",
                (key, json.dumps(content)),
            )
            connection.commit()
            logger.trace(f"Saved artifact: {key}")

    def save_run(self, key: str, content: Any):
        """
        Save a JSON-serializable artifact to the permanent runs history table.

        Args:
            key: The unique string identifier for the run.
            content: A JSON-serializable object to store in the history.
        """
        with sqlite3.connect(self.database_path) as connection:
            connection.execute(
                "INSERT INTO runs (key, content) VALUES (?, ?)", (key, json.dumps(content))
            )
            connection.commit()
            logger.trace(f"Saved run history for: {key}")

    def delete_artifact(self, key: str):
        """
        Delete an artifact by key from the active cache.

        Args:
            key: The unique string identifier for the artifact to remove.
        """
        with sqlite3.connect(self.database_path) as connection:
            connection.execute("DELETE FROM artifacts WHERE key = ?", (key,))
            connection.commit()

    def clear_stage(self, stage: str):
        """
        Delete all artifacts belonging to a specific pipeline stage from the active cache.

        Args:
            stage: The name of the stage (e.g., 'preprocess', 'assessment', 'diagnostic').
        """
        with sqlite3.connect(self.database_path) as connection:
            connection.execute("DELETE FROM artifacts WHERE key LIKE ?", (f"{stage}%",))
            connection.commit()
            logger.warning(f"Cleared artifacts for stage: {stage}")

    def clear_database(self):
        """
        Wipe all data from both the active cache and the run history.
        """
        with sqlite3.connect(self.database_path) as connection:
            connection.execute("DELETE FROM artifacts")
            connection.execute("DELETE FROM runs")
            connection.commit()
            logger.warning("Database cleared.")

    def get_all_artifacts(self) -> list[tuple[str, Any]]:
        """
        Retrieve all artifacts from the active cache.

        Returns:
            A list of tuples containing (key, content).
        """
        with sqlite3.connect(self.database_path) as connection:
            cursor = connection.execute("SELECT key, content FROM artifacts")
            return [(row[0], json.loads(row[1])) for row in cursor.fetchall()]
