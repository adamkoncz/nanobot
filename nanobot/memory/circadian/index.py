import re
import sqlite3
from pathlib import Path
from typing import Any

from loguru import logger


class CircadianIndex:
    """
    Hybrid search index using SQLite FTS5 and sqlite-vec (if available).
    """

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        
        self.vec_enabled = False
        try:
            import sqlite_vec
            self.conn.enable_load_extension(True)
            sqlite_vec.load(self.conn)
            self.conn.enable_load_extension(False)
            self.vec_enabled = True
            logger.debug("sqlite-vec loaded successfully for Circadian Index.")
        except ImportError:
            logger.debug("sqlite_vec module not found. Falling back to keyword search only.")
        except Exception as e:
            logger.debug(f"Failed to load sqlite-vec: {e}. Falling back to keyword search only.")

        self._init_db()

    def _init_db(self) -> None:
        with self.conn:
            # Main nodes table — includes node_type for efficient filtering
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS nodes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    content TEXT NOT NULL,
                    node_type TEXT DEFAULT 'concept',
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Add node_type column if upgrading from older schema
            try:
                self.conn.execute("ALTER TABLE nodes ADD COLUMN node_type TEXT DEFAULT 'concept'")
            except sqlite3.OperationalError:
                pass  # Column already exists
            
            # FTS5 keyword index
            self.conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS nodes_fts USING fts5(
                    name,
                    content,
                    content='nodes',
                    content_rowid='id'
                )
            """)
            
            # Triggers to keep FTS updated
            self.conn.executescript("""
                CREATE TRIGGER IF NOT EXISTS nodes_ai AFTER INSERT ON nodes BEGIN
                  INSERT INTO nodes_fts(rowid, name, content) VALUES (new.id, new.name, new.content);
                END;
                CREATE TRIGGER IF NOT EXISTS nodes_ad AFTER DELETE ON nodes BEGIN
                  INSERT INTO nodes_fts(nodes_fts, rowid, name, content) VALUES('delete', old.id, old.name, old.content);
                END;
                CREATE TRIGGER IF NOT EXISTS nodes_au AFTER UPDATE ON nodes BEGIN
                  INSERT INTO nodes_fts(nodes_fts, rowid, name, content) VALUES('delete', old.id, old.name, old.content);
                  INSERT INTO nodes_fts(rowid, name, content) VALUES (new.id, new.name, new.content);
                END;
            """)

    def index_node(
        self,
        name: str,
        content: str,
        embedding: list[float] | None = None,
        node_type: str = "concept",
    ) -> None:
        """Upserts a node into the database."""
        with self.conn:
            cursor = self.conn.execute("""
                INSERT INTO nodes (name, content, node_type, updated_at)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(name) DO UPDATE SET
                    content=excluded.content,
                    node_type=excluded.node_type,
                    updated_at=CURRENT_TIMESTAMP
                RETURNING id
            """, (name, content, node_type))
            row_id = cursor.fetchone()[0]

            if self.vec_enabled and embedding:
                dim = len(embedding)
                self.conn.execute(f"""
                    CREATE VIRTUAL TABLE IF NOT EXISTS vec_nodes USING vec0(
                        id INTEGER PRIMARY KEY, 
                        embedding float[{dim}]
                    )
                """)
                import struct
                blob = struct.pack(f"{dim}f", *embedding)
                self.conn.execute("""
                    INSERT OR REPLACE INTO vec_nodes(id, embedding)
                    VALUES (?, ?)
                """, (row_id, blob))

    def remove_node(self, name: str) -> None:
        with self.conn:
            cursor = self.conn.execute("SELECT id FROM nodes WHERE name = ?", (name,))
            row = cursor.fetchone()
            if row:
                row_id = row[0]
                self.conn.execute("DELETE FROM nodes WHERE id = ?", (row_id,))
                if self.vec_enabled:
                    try:
                        self.conn.execute("DELETE FROM vec_nodes WHERE id = ?", (row_id,))
                    except sqlite3.OperationalError:
                        pass

    def list_by_type(self, node_type: str) -> list[str]:
        """Return node names matching a specific type (e.g. 'staging')."""
        cursor = self.conn.execute(
            "SELECT name FROM nodes WHERE node_type = ?", (node_type,)
        )
        return [row[0] for row in cursor.fetchall()]

    @staticmethod
    def _sanitize_fts_query(query: str) -> str:
        """Escape FTS5 special characters by wrapping each token in double quotes."""
        # Split on whitespace, quote each token, rejoin
        tokens = query.split()
        if not tokens:
            return '""'
        return " ".join(f'"{t}"' for t in tokens)

    def search(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        """Keyword search with FTS5 query sanitization."""
        safe_query = self._sanitize_fts_query(query)
        try:
            cursor = self.conn.execute("""
                SELECT name, content 
                FROM nodes_fts 
                WHERE nodes_fts MATCH ? 
                ORDER BY rank 
                LIMIT ?
            """, (safe_query, limit))
            return [dict(row) for row in cursor.fetchall()]
        except sqlite3.OperationalError as e:
            logger.warning(f"FTS5 search failed for query {query!r}: {e}")
            return []

    def semantic_search(self, query_embedding: list[float], limit: int = 5) -> list[dict[str, Any]]:
        if not self.vec_enabled:
            return []
            
        dim = len(query_embedding)
        import struct
        blob = struct.pack(f"{dim}f", *query_embedding)
        
        cursor = self.conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='vec_nodes'")
        if not cursor.fetchone():
            return []

        cursor = self.conn.execute("""
            SELECT n.name, n.content, v.distance
            FROM vec_nodes v
            JOIN nodes n ON v.id = n.id
            WHERE v.embedding MATCH ? AND k = ?
            ORDER BY v.distance
        """, (blob, limit))
        return [dict(row) for row in cursor.fetchall()]
        
    def close(self) -> None:
        self.conn.close()

    def __del__(self) -> None:
        try:
            self.conn.close()
        except Exception:
            pass
