from __future__ import annotations

import csv
from pathlib import Path

from sqlalchemy.dialects.postgresql import (
    insert as postgresql_insert,
)

from app.config import get_settings
from app.database.bootstrap import (
    initialize_database,
)
from app.database.models import (
    Company,
    CompanyAlias,
)
from app.database.repositories import (
    get_database_counts,
)
from app.database.session import SessionLocal
from app.normalization.normalizer import (
    EFTNormalizer,
)


settings = get_settings()
normalizer = EFTNormalizer()


def normalize_alias(alias_text: str) -> str:
    result = normalizer.normalize(alias_text)

    return (
        result["core_text"]
        or result["ascii_text"]
    ).strip()


def read_csv(
    path: Path,
) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(
            f"CSV dosyası bulunamadı: {path}"
        )

    with path.open(
        mode="r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        return list(csv.DictReader(file))


def upsert_company(
    session,
    row: dict[str, str],
) -> None:
    company_id = int(row["company_id"])

    tax_number = row.get(
        "tax_number",
        "",
    ).strip() or None

    values = {
        "company_id": company_id,
        "legal_name": row["legal_name"].strip(),
        "brand_name": row.get(
            "brand_name",
            "",
        ).strip(),
        "city": row.get(
            "city",
            "",
        ).strip(),
        "sector": row.get(
            "sector",
            "",
        ).strip(),
        "tax_number": tax_number,
    }

    statement = postgresql_insert(
        Company
    ).values(**values)

    statement = statement.on_conflict_do_update(
        index_elements=[
            Company.company_id,
        ],
        set_={
            "legal_name": values["legal_name"],
            "brand_name": values["brand_name"],
            "city": values["city"],
            "sector": values["sector"],
            "tax_number": values["tax_number"],
        },
    )

    session.execute(statement)


def upsert_alias(
    session,
    *,
    company_id: int,
    alias_text: str,
    alias_type: str,
    source: str,
    confidence: float = 1.0,
) -> None:
    alias_text = alias_text.strip()

    if not alias_text:
        return

    normalized_alias = normalize_alias(alias_text)

    if not normalized_alias:
        return

    values = {
        "company_id": company_id,
        "alias_text": alias_text,
        "normalized_alias": normalized_alias,
        "alias_type": alias_type,
        "source": source,
        "confidence": confidence,
    }

    statement = postgresql_insert(
        CompanyAlias
    ).values(**values)

    statement = statement.on_conflict_do_update(
        constraint="uq_company_alias_normalized",
        set_={
            "alias_text": values["alias_text"],
            "alias_type": values["alias_type"],
            "source": values["source"],
            "confidence": values["confidence"],
        },
    )

    session.execute(statement)


def main() -> None:
    initialize_database()

    company_rows = read_csv(
        settings.companies_csv_path
    )

    alias_rows = read_csv(
        settings.aliases_csv_path
    )

    print(
        f"{len(company_rows)} şirket kaydı okunuyor."
    )

    print(
        f"{len(alias_rows)} alias kaydı okunuyor."
    )

    with SessionLocal.begin() as session:
        for company_row in company_rows:
            upsert_company(
                session,
                company_row,
            )

        for alias_row in alias_rows:
            upsert_alias(
                session,
                company_id=int(
                    alias_row["company_id"]
                ),
                alias_text=alias_row["alias_text"],
                alias_type=alias_row["alias_type"],
                source="CSV_EXPLICIT",
            )

        for company_row in company_rows:
            company_id = int(
                company_row["company_id"]
            )

            upsert_alias(
                session,
                company_id=company_id,
                alias_text=company_row["legal_name"],
                alias_type="LEGAL_NAME",
                source="CSV_GENERATED",
            )

            brand_name = company_row.get(
                "brand_name",
                "",
            ).strip()

            if brand_name:
                upsert_alias(
                    session,
                    company_id=company_id,
                    alias_text=brand_name,
                    alias_type="BRAND_NAME",
                    source="CSV_GENERATED",
                )

    with SessionLocal() as session:
        counts = get_database_counts(session)

    print()
    print("CSV aktarımı tamamlandı.")
    print(
        f"Şirket sayısı: "
        f"{counts['company_count']}"
    )
    print(
        f"Alias sayısı: "
        f"{counts['alias_count']}"
    )


if __name__ == "__main__":
    main()