"""Apply reviewed PostgreSQL migrations to the configured Supabase database.

Run this explicitly from a trusted environment or a Render one-off job.  The
web service does not mutate production schema during import or startup.
"""

from __future__ import annotations

import argparse

from database import get_database_url, run_migrations


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database-url",
        help="PostgreSQL URL; defaults to DATABASE_URL or POSTGRES_URL.",
    )
    args = parser.parse_args()

    if not (args.database_url or get_database_url()):
        parser.error("Set DATABASE_URL/POSTGRES_URL or pass --database-url.")

    applied = run_migrations(database_url=args.database_url)
    if applied:
        print("Applied migrations:")
        for migration in applied:
            print(f"- {migration}")
    else:
        print("Database schema is already up to date.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
