from __future__ import annotations

from typing import Any

from sqlalchemy import text

from app.database.base import Base
from app.database.session import engine


def initialize_database() -> None:
    """
    PostgreSQL uzantılarını ve uygulama tablolarını oluşturur.
    """

    with engine.begin() as connection:
        connection.execute(
            text(
                "CREATE EXTENSION IF NOT EXISTS pg_trgm"
            )
        )

        connection.execute(
            text(
                "CREATE EXTENSION IF NOT EXISTS vector"
            )
        )

    Base.metadata.create_all(bind=engine)


def get_database_status() -> dict[str, Any]:
    with engine.connect() as connection:
        database_name = connection.execute(
            text("SELECT current_database()")
        ).scalar_one()

        postgres_version = connection.execute(
            text(
                "SELECT current_setting('server_version')"
            )
        ).scalar_one()

        pg_trgm_version = connection.execute(
            text(
                """
                SELECT extversion
                FROM pg_extension
                WHERE extname = 'pg_trgm'
                """
            )
        ).scalar_one_or_none()

        vector_version = connection.execute(
            text(
                """
                SELECT extversion
                FROM pg_extension
                WHERE extname = 'vector'
                """
            )
        ).scalar_one_or_none()

        return {
            "connected": True,
            "database_name": database_name,
            "postgres_version": postgres_version,
            "pg_trgm_version": pg_trgm_version,
            "vector_version": vector_version,
        }