"""Small, testable PostgreSQL foundation for Memora.

The application does not import this module from its existing routes.  It is
kept as an opt-in data layer so callers can use PostgreSQL incrementally while
unit tests can inject a connection or connection factory without a live DB.
"""

from __future__ import annotations

import os
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, Iterator, Mapping

try:
    import psycopg
    from psycopg.rows import dict_row
except ImportError:  # pragma: no cover - requirements install psycopg in normal use
    psycopg = None  # type: ignore[assignment]
    dict_row = None  # type: ignore[assignment]


DEFAULT_MIGRATIONS_DIR = (
    Path(__file__).resolve().parent / "supabase" / "migrations"
)
MIGRATION_TABLE_DDL = """
create table if not exists public.schema_migrations (
    version text primary key,
    applied_at timestamptz not null default now()
)
"""


class DatabaseConfigurationError(RuntimeError):
    """Raised when a database operation has no configured connection URL."""


def get_database_url(environ: Mapping[str, str] | None = None) -> str | None:
    """Return the configured PostgreSQL URL without exposing its contents.

    ``DATABASE_URL`` is preferred so hosted deployments and local tooling can
    share the same convention.  ``POSTGRES_URL`` is accepted for compatibility
    with Supabase and existing deployment templates.
    """

    values = os.environ if environ is None else environ
    for variable in ("DATABASE_URL", "POSTGRES_URL"):
        value = values.get(variable)
        if value and value.strip():
            return value.strip()
    return None


def _require_psycopg() -> Any:
    if psycopg is None:
        raise RuntimeError(
            "psycopg is required for database access; install requirements.txt"
        )
    return psycopg


@contextmanager
def _cursor(connection: Any) -> Iterator[Any]:
    """Use a cursor without requiring a cursor context manager in test doubles."""

    cursor = connection.cursor()
    try:
        yield cursor
    finally:
        close = getattr(cursor, "close", None)
        if close is not None:
            close()


@contextmanager
def get_connection(
    database_url: str | None = None,
    **connect_kwargs: Any,
) -> Iterator[Any]:
    """Yield a psycopg connection and close it safely.

    A successful scope commits.  Any exception rolls the transaction back and
    is re-raised before the connection is closed.  The URL is never logged or
    included in an exception generated here.
    """

    url = database_url or get_database_url()
    if not url:
        raise DatabaseConfigurationError(
            "Set DATABASE_URL or POSTGRES_URL before opening a database connection."
        )

    driver = _require_psycopg()
    connect_kwargs.setdefault("connect_timeout", 10)
    if "row_factory" not in connect_kwargs and dict_row is not None:
        connect_kwargs["row_factory"] = dict_row

    conn = driver.connect(url, **connect_kwargs)
    try:
        yield conn
    except BaseException:
        try:
            conn.rollback()
        finally:
            conn.close()
        raise
    else:
        try:
            conn.commit()
        except BaseException:
            try:
                conn.rollback()
            finally:
                conn.close()
            raise
        conn.close()


@contextmanager
def connection(
    database_url: str | None = None,
    **connect_kwargs: Any,
) -> Iterator[Any]:
    """Compatibility alias for callers that prefer ``connection()``."""

    with get_connection(database_url, **connect_kwargs) as conn:
        yield conn


def health_check(database_url: str | None = None, **connect_kwargs: Any) -> bool:
    """Return ``True`` when PostgreSQL accepts a simple query.

    Configuration and connectivity failures return ``False`` so this helper is
    safe to use in startup probes and tests that intentionally have no DB.
    """

    try:
        connect_kwargs.setdefault("connect_timeout", 5)
        with get_connection(database_url, **connect_kwargs) as conn:
            with _cursor(conn) as cursor:
                cursor.execute("select 1")
        return True
    except Exception:
        return False


def migration_files(migrations_dir: str | Path | None = None) -> tuple[Path, ...]:
    """Return SQL migrations in deterministic filename order."""

    directory = Path(migrations_dir) if migrations_dir else DEFAULT_MIGRATIONS_DIR
    if not directory.is_dir():
        raise FileNotFoundError(f"Migration directory does not exist: {directory}")
    return tuple(sorted(path for path in directory.glob("*.sql") if path.is_file()))


def _apply_migrations(conn: Any, files: tuple[Path, ...]) -> list[str]:
    applied: list[str] = []
    with _cursor(conn) as cursor:
        # Serialize concurrent startup runners for the duration of this
        # transaction, including the first creation of the migration table.
        cursor.execute(
            "select pg_advisory_xact_lock(hashtext(%s))",
            ("memora:schema_migrations",),
        )
        cursor.execute(MIGRATION_TABLE_DDL)
        for migration in files:
            version = migration.name
            cursor.execute(
                "select 1 from public.schema_migrations where version = %s",
                (version,),
            )
            if cursor.fetchone() is not None:
                continue

            sql = migration.read_text(encoding="utf-8")
            if not sql.strip():
                continue

            # Migrations are local, reviewed SQL files and contain no runtime
            # values.  They are executed as one transaction by get_connection.
            cursor.execute(sql)
            cursor.execute(
                "insert into public.schema_migrations (version) values (%s)",
                (version,),
            )
            applied.append(version)
    return applied


def run_migrations(
    migrations_dir: str | Path | None = None,
    database_url: str | None = None,
    connection: Any | None = None,
    **connect_kwargs: Any,
) -> list[str]:
    """Apply pending SQL migrations and return the filenames applied.

    Passing ``connection`` is useful for tests and for callers already inside a
    larger transaction.  When omitted, the normal safe connection context is
    used and commits only after all migrations succeed.
    """

    files = migration_files(migrations_dir)
    if connection is not None:
        with nullcontext(connection) as conn:
            return _apply_migrations(conn, files)

    # Every migration and its history entry must commit together, even if the
    # caller normally opens autocommit connections for other operations.
    connect_kwargs["autocommit"] = False
    with get_connection(database_url, **connect_kwargs) as conn:
        return _apply_migrations(conn, files)


__all__ = [
    "DatabaseConfigurationError",
    "DEFAULT_MIGRATIONS_DIR",
    "connection",
    "get_connection",
    "get_database_url",
    "health_check",
    "migration_files",
    "run_migrations",
]
