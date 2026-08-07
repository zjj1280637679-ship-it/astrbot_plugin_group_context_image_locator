from __future__ import annotations

import hashlib
import sqlite3
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

IMAGE_TOO_LARGE_ERROR = "图片超过插件允许缓存的单图大小"
IMAGE_FORMAT_UNRECOGNIZED_ERROR = "图片格式无法安全识别"
CAPTURE_ERROR_CATEGORIES = ("超时", "I/O", "数据无效", "内部错误")
GENERIC_CAPTURE_ERROR = "图片捕获失败（内部错误）"
SAFE_UNRESOLVED_ERRORS = frozenset(
    {
        IMAGE_TOO_LARGE_ERROR,
        IMAGE_FORMAT_UNRECOGNIZED_ERROR,
        *(f"图片捕获失败（{category}）" for category in CAPTURE_ERROR_CATEGORIES),
    }
)


@dataclass(frozen=True, slots=True)
class StoredImage:
    locator: str
    blob_hash: str
    mime_type: str
    file_path: Path
    size: int


@dataclass(frozen=True, slots=True)
class ImageLookup:
    locator: str
    message_id: str | None
    image_index: int | None
    source: str
    image: StoredImage | None
    error: str | None


@dataclass(frozen=True, slots=True)
class PruneResult:
    occurrences_removed: int
    blobs_removed: int
    bytes_removed: int


class ImageLocatorStore:
    """Content-addressed image blobs plus scoped occurrence locators."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.blob_root = self.root / "blobs"
        self.db_path = self.root / "index.sqlite3"
        self.blob_root.mkdir(parents=True, exist_ok=True)
        self._initialize_database()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def _initialize_database(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS blobs (
                    blob_hash TEXT PRIMARY KEY,
                    mime_type TEXT NOT NULL,
                    file_name TEXT NOT NULL,
                    size INTEGER NOT NULL,
                    created_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS occurrences (
                    scope TEXT NOT NULL,
                    locator TEXT NOT NULL,
                    message_id TEXT NOT NULL,
                    image_index INTEGER NOT NULL,
                    blob_hash TEXT,
                    source TEXT NOT NULL DEFAULT '',
                    created_at REAL NOT NULL,
                    accessed_at REAL NOT NULL,
                    last_error TEXT,
                    PRIMARY KEY (scope, locator),
                    UNIQUE (scope, message_id, image_index),
                    FOREIGN KEY (blob_hash) REFERENCES blobs(blob_hash)
                );

                CREATE INDEX IF NOT EXISTS idx_occurrences_blob_hash
                    ON occurrences(blob_hash);
                CREATE INDEX IF NOT EXISTS idx_occurrences_accessed_at
                    ON occurrences(accessed_at);
                """
            )
            connection.execute("UPDATE occurrences SET source='' WHERE source<>''")
            connection.execute(
                """
                UPDATE occurrences
                SET last_error='此前图片捕获失败；请等待图片重新出现。'
                WHERE last_error IS NOT NULL AND last_error<>''
                """
            )

    def put(
        self,
        *,
        scope: str,
        locator: str,
        message_id: str,
        image_index: int,
        source: str,
        data: bytes,
        mime_type: str,
    ) -> StoredImage:
        if not data:
            raise ValueError("image data must not be empty")

        blob_hash = hashlib.sha256(data).hexdigest()
        extension = _extension_for_mime(mime_type)
        relative_path = Path(blob_hash[:2]) / f"{blob_hash}{extension}"
        file_path = self.blob_root / relative_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        if not file_path.exists():
            temporary_path = file_path.with_name(
                f".{file_path.name}.{uuid.uuid4().hex}.tmp"
            )
            try:
                temporary_path.write_bytes(data)
                temporary_path.replace(file_path)
            finally:
                temporary_path.unlink(missing_ok=True)

        now = time.time()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                INSERT OR IGNORE INTO blobs(
                    blob_hash, mime_type, file_name, size, created_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    blob_hash,
                    mime_type,
                    relative_path.as_posix(),
                    len(data),
                    now,
                ),
            )
            connection.execute(
                """
                INSERT INTO occurrences(
                    scope, locator, message_id, image_index, blob_hash,
                    source, created_at, accessed_at, last_error
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL)
                ON CONFLICT(scope, locator) DO UPDATE SET
                    message_id=excluded.message_id,
                    image_index=excluded.image_index,
                    blob_hash=excluded.blob_hash,
                    source=excluded.source,
                    accessed_at=excluded.accessed_at,
                    last_error=NULL
                """,
                (
                    scope,
                    locator,
                    message_id,
                    image_index,
                    blob_hash,
                    "",
                    now,
                    now,
                ),
            )

        return StoredImage(
            locator=locator,
            blob_hash=blob_hash,
            mime_type=mime_type,
            file_path=file_path,
            size=len(data),
        )

    def record_unresolved(
        self,
        *,
        scope: str,
        locator: str,
        message_id: str,
        image_index: int,
        source: str,
        error: str,
    ) -> None:
        persisted_error = (
            error if error in SAFE_UNRESOLVED_ERRORS else GENERIC_CAPTURE_ERROR
        )
        now = time.time()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO occurrences(
                    scope, locator, message_id, image_index, blob_hash,
                    source, created_at, accessed_at, last_error
                ) VALUES (?, ?, ?, ?, NULL, ?, ?, ?, ?)
                ON CONFLICT(scope, locator) DO UPDATE SET
                    source=excluded.source,
                    accessed_at=excluded.accessed_at,
                    last_error=CASE
                        WHEN occurrences.blob_hash IS NULL THEN excluded.last_error
                        ELSE NULL
                    END
                """,
                (
                    scope,
                    locator,
                    message_id,
                    image_index,
                    "",
                    now,
                    now,
                    persisted_error,
                ),
            )

    def resolve_many(self, scope: str, locators: list[str]) -> list[ImageLookup]:
        if not locators:
            return []

        placeholders = ",".join("?" for _ in locators)
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT
                    o.locator, o.message_id, o.image_index, o.source,
                    o.last_error, b.blob_hash, b.mime_type, b.file_name, b.size
                FROM occurrences AS o
                LEFT JOIN blobs AS b ON b.blob_hash = o.blob_hash
                WHERE o.scope = ? AND o.locator IN ({placeholders})
                """,
                [scope, *locators],
            ).fetchall()
            now = time.time()
            connection.executemany(
                """
                UPDATE occurrences SET accessed_at = ?
                WHERE scope = ? AND locator = ?
                """,
                [(now, scope, locator) for locator in locators],
            )

        row_by_locator = {str(row["locator"]): row for row in rows}
        lookups: list[ImageLookup] = []
        for locator in locators:
            row = row_by_locator.get(locator)
            if row is None:
                lookups.append(
                    ImageLookup(locator, None, None, "", None, "提取码不存在")
                )
                continue

            image = None
            error = row["last_error"]
            if row["blob_hash"]:
                file_path = self._safe_blob_path(str(row["file_name"]))
                if file_path.is_file():
                    image = StoredImage(
                        locator=locator,
                        blob_hash=str(row["blob_hash"]),
                        mime_type=str(row["mime_type"]),
                        file_path=file_path,
                        size=int(row["size"]),
                    )
                    error = None
                else:
                    error = "原图缓存文件缺失"

            lookups.append(
                ImageLookup(
                    locator=locator,
                    message_id=str(row["message_id"]),
                    image_index=int(row["image_index"]),
                    source=str(row["source"] or ""),
                    image=image,
                    error=str(error) if error else None,
                )
            )
        return lookups

    def prune(self, *, retention_seconds: int, max_bytes: int) -> PruneResult:
        now = time.time()
        files_to_remove: list[Path] = []
        occurrences_removed = 0
        blobs_removed = 0
        bytes_removed = 0

        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            if retention_seconds > 0:
                cursor = connection.execute(
                    "DELETE FROM occurrences WHERE created_at < ?",
                    (now - retention_seconds,),
                )
                occurrences_removed += max(cursor.rowcount, 0)

            orphan_rows = connection.execute(
                """
                SELECT b.blob_hash, b.file_name, b.size
                FROM blobs AS b
                WHERE NOT EXISTS (
                    SELECT 1 FROM occurrences AS o
                    WHERE o.blob_hash = b.blob_hash
                )
                """
            ).fetchall()
            for row in orphan_rows:
                connection.execute(
                    "DELETE FROM blobs WHERE blob_hash = ?",
                    (row["blob_hash"],),
                )
                files_to_remove.append(self._safe_blob_path(str(row["file_name"])))
                blobs_removed += 1
                bytes_removed += int(row["size"])

            total_bytes = int(
                connection.execute(
                    "SELECT COALESCE(SUM(size), 0) FROM blobs"
                ).fetchone()[0]
            )
            if max_bytes > 0 and total_bytes > max_bytes:
                candidates = connection.execute(
                    """
                    SELECT
                        b.blob_hash, b.file_name, b.size,
                        COALESCE(MAX(o.accessed_at), b.created_at) AS last_used
                    FROM blobs AS b
                    LEFT JOIN occurrences AS o ON o.blob_hash = b.blob_hash
                    GROUP BY b.blob_hash
                    ORDER BY last_used ASC, b.created_at ASC
                    """
                ).fetchall()
                for row in candidates:
                    if total_bytes <= max_bytes:
                        break
                    cursor = connection.execute(
                        "DELETE FROM occurrences WHERE blob_hash = ?",
                        (row["blob_hash"],),
                    )
                    occurrences_removed += max(cursor.rowcount, 0)
                    connection.execute(
                        "DELETE FROM blobs WHERE blob_hash = ?",
                        (row["blob_hash"],),
                    )
                    size = int(row["size"])
                    total_bytes -= size
                    files_to_remove.append(self._safe_blob_path(str(row["file_name"])))
                    blobs_removed += 1
                    bytes_removed += size

            # Keep file removal inside the database transaction.  If any unlink
            # fails, sqlite rolls the row deletions back, so a retry can still
            # discover the blob instead of leaving an unindexed file behind.
            for file_path in files_to_remove:
                file_path.unlink(missing_ok=True)

        return PruneResult(
            occurrences_removed=occurrences_removed,
            blobs_removed=blobs_removed,
            bytes_removed=bytes_removed,
        )

    def stats(self) -> dict[str, int]:
        with self._connect() as connection:
            occurrences = int(
                connection.execute("SELECT COUNT(*) FROM occurrences").fetchone()[0]
            )
            blobs, total_bytes = connection.execute(
                "SELECT COUNT(*), COALESCE(SUM(size), 0) FROM blobs"
            ).fetchone()
        return {
            "occurrences": occurrences,
            "blobs": int(blobs),
            "bytes": int(total_bytes),
        }

    def _safe_blob_path(self, file_name: str) -> Path:
        path = (self.blob_root / file_name).resolve()
        if path != self.blob_root and self.blob_root not in path.parents:
            raise ValueError(f"unsafe blob path: {file_name}")
        return path


def _extension_for_mime(mime_type: str) -> str:
    return {
        "image/png": ".png",
        "image/jpeg": ".jpg",
        "image/gif": ".gif",
        "image/webp": ".webp",
        "image/bmp": ".bmp",
    }.get(mime_type.lower(), ".img")
