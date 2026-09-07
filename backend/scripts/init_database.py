from __future__ import annotations

from app.database.bootstrap import (
    get_database_status,
    initialize_database,
)


def main() -> None:
    print("PostgreSQL bağlantısı kontrol ediliyor.")

    initialize_database()

    status = get_database_status()

    print()
    print("Veritabanı hazır:")
    print(f"Database: {status['database_name']}")
    print(f"PostgreSQL: {status['postgres_version']}")
    print(f"pg_trgm: {status['pg_trgm_version']}")
    print(f"pgvector: {status['vector_version']}")


if __name__ == "__main__":
    main()