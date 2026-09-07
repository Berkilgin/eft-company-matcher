from __future__ import annotations

from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.database.models import (
    Company,
    CompanyAlias,
)
from app.retrieval.company_index import (
    AliasRecord,
    CompanyIndex,
    CompanyRecord,
)


def build_company_index_from_database(
    session: Session,
) -> CompanyIndex:
    company_rows = session.scalars(
        select(Company).order_by(
            Company.company_id
        )
    ).all()

    alias_rows = session.scalars(
        select(CompanyAlias).order_by(
            CompanyAlias.alias_id
        )
    ).all()

    if not company_rows:
        raise ValueError(
            "PostgreSQL companies tablosu boş."
        )

    if not alias_rows:
        raise ValueError(
            "PostgreSQL company_aliases tablosu boş."
        )

    companies = {
        int(company.company_id): CompanyRecord(
            company_id=int(company.company_id),
            legal_name=company.legal_name,
            brand_name=company.brand_name,
            city=company.city,
            sector=company.sector,
            tax_number=company.tax_number or "",
        )
        for company in company_rows
    }

    aliases = [
        AliasRecord(
            alias_id=int(alias.alias_id),
            company_id=int(alias.company_id),
            alias_text=alias.alias_text,
            normalized_alias=alias.normalized_alias,
            alias_type=alias.alias_type,
        )
        for alias in alias_rows
    ]

    return CompanyIndex(
        companies=companies,
        aliases=aliases,
    )


def get_database_counts(
    session: Session,
) -> dict[str, int]:
    company_count = session.scalar(
        select(func.count())
        .select_from(Company)
    )

    alias_count = session.scalar(
        select(func.count())
        .select_from(CompanyAlias)
    )

    return {
        "company_count": int(company_count or 0),
        "alias_count": int(alias_count or 0),
    }


def get_company_by_id(
    session: Session,
    company_id: int,
) -> Company | None:
    return session.get(
        Company,
        company_id,
    )


def list_companies(
    session: Session,
    limit: int = 100,
    offset: int = 0,
) -> list[Company]:
    statement = (
        select(Company)
        .order_by(Company.legal_name)
        .offset(offset)
        .limit(limit)
    )

    return list(
        session.scalars(statement).all()
    )


def database_summary(
    session: Session,
) -> dict[str, Any]:
    return {
        **get_database_counts(session),
        "data_source": "postgres",
    }