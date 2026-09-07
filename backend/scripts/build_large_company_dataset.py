from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests

from app.normalization.normalizer import EFTNormalizer


GLEIF_API_URL = "https://api.gleif.org/api/v1/lei-records"

PAGE_SIZE = 200

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_DIRECTORY = PROJECT_ROOT / "data" / "raw"
GENERATED_DIRECTORY = PROJECT_ROOT / "data" / "generated"
BACKUP_DIRECTORY = PROJECT_ROOT / "data" / "backups"

RAW_COMPANIES_PATH = RAW_DATA_DIRECTORY / "companies.csv"
RAW_ALIASES_PATH = RAW_DATA_DIRECTORY / "aliases.csv"

GENERATED_COMPANIES_PATH = (
    GENERATED_DIRECTORY / "companies.csv"
)

GENERATED_ALIASES_PATH = (
    GENERATED_DIRECTORY / "aliases.csv"
)

GENERATED_REPORT_PATH = (
    GENERATED_DIRECTORY / "dataset_report.json"
)


# Türkiye öncelikli olmak üzere farklı ülkelerden kayıt toplanır.
COUNTRY_QUOTAS: tuple[tuple[str, int], ...] = (
    ("TR", 150),
    ("US", 120),
    ("GB", 100),
    ("DE", 80),
    ("FR", 70),
    ("NL", 50),
    ("CH", 50),
    ("JP", 40),
    ("CA", 40),
    ("AU", 40),
)


COMPANY_FIELDS = [
    "company_id",
    "legal_name",
    "brand_name",
    "city",
    "sector",
    "tax_number",
    "country",
    "lei",
    "source",
    "entity_status",
]


ALIAS_FIELDS = [
    "alias_id",
    "company_id",
    "alias_text",
    "alias_type",
    "source",
    "confidence",
]


# Yalnızca adın sonundaki tüzel kişilik eklerini kaldırır.
# Resmî unvan ayrıca alias olarak saklanmaya devam eder.
LEGAL_SUFFIX_PATTERN = re.compile(
    r"""
    (?:,\s*|\s+)
    (?:
        incorporated
        | inc\.?
        | corporation
        | corp\.?
        | limited\s+liability\s+company
        | l\.?\s*l\.?\s*c\.?
        | public\s+limited\s+company
        | p\.?\s*l\.?\s*c\.?
        | limited
        | ltd\.?
        | gesellschaft\s+mit\s+beschr[aä]nkter\s+haftung
        | gmbh
        | aktiengesellschaft
        | ag
        | soci[eé]t[eé]\s+anonyme
        | s\.?\s*a\.?
        | s\.?\s*a\.?\s*s\.?
        | sarl
        | n\.?\s*v\.?
        | b\.?\s*v\.?
        | s\.?\s*p\.?\s*a\.?
        | s\.?\s*r\.?\s*l\.?
        | anonim\s+[şs]irketi
        | limited\s+[şs]irketi
        | a\.?\s*[şs]\.?
        | ltd\.?\s*[şs]ti\.?
    )
    \s*$
    """,
    flags=re.IGNORECASE | re.VERBOSE,
)


normalizer = EFTNormalizer()


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "GLEIF API üzerinden gerçek şirket kayıtları "
            "indirerek mevcut şirket veri setini büyütür."
        )
    )

    parser.add_argument(
        "--target",
        type=int,
        default=700,
        help=(
            "Mevcut şirketler dahil hedef toplam şirket sayısı. "
            "Varsayılan: 700"
        ),
    )

    parser.add_argument(
        "--minimum",
        type=int,
        default=500,
        help=(
            "Bu sayıya ulaşılamazsa ham veri dosyaları "
            "değiştirilmez. Varsayılan: 500"
        ),
    )

    parser.add_argument(
        "--apply",
        action="store_true",
        help=(
            "Oluşturulan dosyaları data/raw içindeki aktif "
            "veri setinin üzerine uygular. Önce yedek alınır."
        ),
    )

    return parser.parse_args()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []

    with path.open(
        mode="r",
        encoding="utf-8-sig",
        newline="",
    ) as csv_file:
        return list(csv.DictReader(csv_file))


def write_csv_rows(
    path: Path,
    fieldnames: list[str],
    rows: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    with temporary_path.open(
        mode="w",
        encoding="utf-8-sig",
        newline="",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(
                {
                    field: row.get(field, "")
                    for field in fieldnames
                }
            )

    temporary_path.replace(path)


def normalize_company_name(text: str) -> str:
    cleaned_text = text.strip()

    if not cleaned_text:
        return ""

    result = normalizer.normalize(cleaned_text)

    return (
        result["core_text"]
        or result["ascii_text"]
    ).strip()


def get_name(value: Any) -> str:
    if isinstance(value, dict):
        return str(
            value.get("name", "")
        ).strip()

    if isinstance(value, str):
        return value.strip()

    return ""


def get_address_value(
    address: Any,
    key: str,
) -> str:
    if not isinstance(address, dict):
        return ""

    value = address.get(key)

    if value is None:
        return ""

    return str(value).strip()


def create_short_name(legal_name: str) -> str:
    """
    Resmî unvanın sonundaki şirket türünü çıkararak
    eşleştirmeye yardımcı kısa bir alias oluşturur.
    """
    current_name = legal_name.strip(" ,.-")

    while current_name:
        shortened_name = LEGAL_SUFFIX_PATTERN.sub(
            "",
            current_name,
        ).strip(" ,.-")

        if shortened_name == current_name:
            break

        if len(shortened_name) < 3:
            break

        current_name = shortened_name

    return current_name


def stable_company_id(lei: str) -> int:
    """
    Aynı LEI için her çalıştırmada aynı signed BIGINT
    aralığında company_id üretir.
    """
    digest = hashlib.sha256(
        lei.encode("utf-8")
    ).hexdigest()

    # İlk 15 hexadecimal karakter maksimum yaklaşık 2^60 olur.
    value = int(digest[:15], 16)

    return 1_000_000_000_000 + value


def get_next_link(payload: dict[str, Any]) -> str | None:
    links = payload.get("links", {})

    if not isinstance(links, dict):
        return None

    next_link = links.get("next")

    if isinstance(next_link, str):
        return next_link or None

    if isinstance(next_link, dict):
        href = next_link.get("href")

        if isinstance(href, str):
            return href or None

    return None


def request_json(
    session: requests.Session,
    url: str,
    params: dict[str, Any] | None = None,
    maximum_attempts: int = 5,
) -> dict[str, Any]:
    last_error: Exception | None = None

    for attempt in range(
        1,
        maximum_attempts + 1,
    ):
        try:
            response = session.get(
                url,
                params=params,
                timeout=60,
            )

            if response.status_code == 429:
                retry_after = response.headers.get(
                    "Retry-After",
                    "5",
                )

                try:
                    wait_seconds = float(retry_after)
                except ValueError:
                    wait_seconds = 5.0

                print(
                    "API hız sınırı uyguladı. "
                    f"{wait_seconds:.1f} saniye bekleniyor."
                )

                time.sleep(wait_seconds)
                continue

            if response.status_code >= 500:
                raise requests.HTTPError(
                    f"GLEIF sunucu hatası: "
                    f"HTTP {response.status_code}",
                    response=response,
                )

            response.raise_for_status()

            payload = response.json()

            if not isinstance(payload, dict):
                raise ValueError(
                    "GLEIF API nesne biçiminde JSON döndürmedi."
                )

            return payload

        except (
            requests.RequestException,
            ValueError,
        ) as exc:
            last_error = exc

            if attempt == maximum_attempts:
                break

            wait_seconds = min(
                2 ** (attempt - 1),
                15,
            )

            print(
                f"İstek başarısız: {type(exc).__name__}: {exc}"
            )
            print(
                f"{wait_seconds} saniye sonra "
                "yeniden denenecek."
            )

            time.sleep(wait_seconds)

    raise RuntimeError(
        "GLEIF API isteği tüm denemelerde başarısız oldu."
    ) from last_error


def fetch_country_records(
    session: requests.Session,
    country_code: str,
    quota: int,
) -> list[dict[str, Any]]:
    print()
    print(
        f"{country_code}: {quota} kayıt hedefleniyor."
    )

    records: list[dict[str, Any]] = []

    url: str | None = GLEIF_API_URL

    params: dict[str, Any] | None = {
        "page[size]": PAGE_SIZE,
        "page[number]": 1,
        "filter[entity.legalAddress.country]": (
            country_code
        ),
        "filter[registration.status]": "ISSUED",
    }

    while url and len(records) < quota:
        payload = request_json(
            session=session,
            url=url,
            params=params,
        )

        # Sonraki bağlantı kendi query parametrelerini içerir.
        params = None

        page_records = payload.get("data", [])

        if not isinstance(page_records, list):
            raise ValueError(
                "GLEIF API data alanı liste değil."
            )

        if not page_records:
            break

        for record in page_records:
            if not isinstance(record, dict):
                continue

            records.append(record)

            if len(records) >= quota:
                break

        print(
            f"{country_code}: "
            f"{len(records)}/{quota} kayıt alındı."
        )

        url = get_next_link(payload)

        # API'ye gereksiz şekilde art arda yük bindirmemek için.
        time.sleep(0.25)

    return records[:quota]


def extract_other_names(
    entity: dict[str, Any],
) -> list[tuple[str, str]]:
    results: list[tuple[str, str]] = []

    source_fields = (
        (
            "otherNames",
            "GLEIF_OTHER_NAME",
        ),
        (
            "transliteratedOtherNames",
            "GLEIF_TRANSLITERATED_NAME",
        ),
    )

    for field_name, fallback_type in source_fields:
        values = entity.get(
            field_name,
            [],
        )

        if not isinstance(values, list):
            continue

        for item in values:
            if isinstance(item, dict):
                name = get_name(item)

                item_type = str(
                    item.get(
                        "type",
                        fallback_type,
                    )
                ).strip()

            elif isinstance(item, str):
                name = item.strip()
                item_type = fallback_type

            else:
                continue

            if not name:
                continue

            results.append(
                (
                    name,
                    item_type[:50],
                )
            )

    return results


def parse_gleif_record(
    record: dict[str, Any],
) -> dict[str, Any] | None:
    attributes = record.get(
        "attributes",
        {},
    )

    if not isinstance(attributes, dict):
        return None

    lei = str(
        attributes.get("lei")
        or record.get("id")
        or ""
    ).strip()

    if not lei:
        return None

    entity = attributes.get(
        "entity",
        {},
    )

    registration = attributes.get(
        "registration",
        {},
    )

    if not isinstance(entity, dict):
        return None

    if not isinstance(registration, dict):
        registration = {}

    legal_name = get_name(
        entity.get("legalName")
    )

    if not legal_name:
        return None

    legal_address = entity.get(
        "legalAddress",
        {},
    )

    headquarters_address = entity.get(
        "headquartersAddress",
        {},
    )

    city = (
        get_address_value(
            headquarters_address,
            "city",
        )
        or get_address_value(
            legal_address,
            "city",
        )
    )

    country = (
        get_address_value(
            legal_address,
            "country",
        )
        or get_address_value(
            headquarters_address,
            "country",
        )
    )

    entity_category = str(
        entity.get("category")
        or "GENERAL"
    ).strip()

    entity_status = str(
        entity.get("status")
        or registration.get("status")
        or ""
    ).strip()

    short_name = create_short_name(
        legal_name
    )

    brand_name = (
        short_name
        if short_name
        else legal_name
    )

    aliases: list[tuple[str, str, float]] = [
        (
            legal_name,
            "GLEIF_LEGAL_NAME",
            1.0,
        )
    ]

    if (
        short_name
        and normalize_company_name(short_name)
        != normalize_company_name(legal_name)
    ):
        aliases.append(
            (
                short_name,
                "GENERATED_SHORT_NAME",
                0.90,
            )
        )

    for other_name, alias_type in (
        extract_other_names(entity)
    ):
        aliases.append(
            (
                other_name,
                alias_type,
                0.95,
            )
        )

    return {
        "lei": lei,
        "legal_name": legal_name,
        "brand_name": brand_name,
        "city": city,
        "country": country,
        "sector": f"GLEIF_{entity_category}",
        "entity_status": entity_status,
        "aliases": aliases,
    }


def normalize_existing_company_row(
    row: dict[str, str],
) -> dict[str, Any]:
    return {
        "company_id": row.get(
            "company_id",
            "",
        ),
        "legal_name": row.get(
            "legal_name",
            "",
        ),
        "brand_name": row.get(
            "brand_name",
            "",
        ),
        "city": row.get(
            "city",
            "",
        ),
        "sector": row.get(
            "sector",
            "",
        ),
        "tax_number": row.get(
            "tax_number",
            "",
        ),
        "country": row.get(
            "country",
            "TR",
        ),
        "lei": row.get(
            "lei",
            "",
        ),
        "source": row.get(
            "source",
            "CURATED",
        ),
        "entity_status": row.get(
            "entity_status",
            "ACTIVE",
        ),
    }


def normalize_existing_alias_row(
    row: dict[str, str],
) -> dict[str, Any]:
    return {
        "alias_id": row.get(
            "alias_id",
            "",
        ),
        "company_id": row.get(
            "company_id",
            "",
        ),
        "alias_text": row.get(
            "alias_text",
            "",
        ),
        "alias_type": row.get(
            "alias_type",
            "EXISTING",
        ),
        "source": row.get(
            "source",
            "CURATED",
        ),
        "confidence": row.get(
            "confidence",
            "1.0",
        ),
    }


def make_unique_company_id(
    lei: str,
    used_company_ids: set[int],
) -> int:
    company_id = stable_company_id(lei)

    while company_id in used_company_ids:
        company_id += 1

    used_company_ids.add(company_id)

    return company_id


def create_dataset(
    target_count: int,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    existing_company_rows = [
        normalize_existing_company_row(row)
        for row in read_csv_rows(
            RAW_COMPANIES_PATH
        )
    ]

    existing_alias_rows = [
        normalize_existing_alias_row(row)
        for row in read_csv_rows(
            RAW_ALIASES_PATH
        )
    ]

    companies = list(existing_company_rows)
    aliases = list(existing_alias_rows)

    used_company_ids: set[int] = set()

    for row in companies:
        try:
            used_company_ids.add(
                int(row["company_id"])
            )
        except (
            TypeError,
            ValueError,
        ):
            continue

    current_alias_ids: list[int] = []

    for row in aliases:
        try:
            current_alias_ids.append(
                int(row["alias_id"])
            )
        except (
            TypeError,
            ValueError,
        ):
            continue

    next_alias_id = (
        max(current_alias_ids, default=0)
        + 1
    )

    seen_leis = {
        str(row.get("lei", "")).strip()
        for row in companies
        if str(row.get("lei", "")).strip()
    }

    # Mevcut elle hazırlanmış şirketlerin GLEIF tarafından
    # tekrar eklenmesini önler.
    seen_company_names = {
        normalize_company_name(
            str(row.get("legal_name", ""))
        )
        for row in companies
        if str(row.get("legal_name", "")).strip()
    }

    session = requests.Session()

    session.headers.update(
        {
            "Accept": "application/vnd.api+json",
            "User-Agent": (
                "eft-company-matcher/0.6 "
                "(local entity-resolution project)"
            ),
        }
    )

    country_added_counts: dict[str, int] = {}

    try:
        for country_code, quota in COUNTRY_QUOTAS:
            if len(companies) >= target_count:
                break

            raw_records = fetch_country_records(
                session=session,
                country_code=country_code,
                quota=quota,
            )

            added_count = 0

            for raw_record in raw_records:
                if len(companies) >= target_count:
                    break

                parsed = parse_gleif_record(
                    raw_record
                )

                if parsed is None:
                    continue

                lei = parsed["lei"]

                if lei in seen_leis:
                    continue

                normalized_legal_name = (
                    normalize_company_name(
                        parsed["legal_name"]
                    )
                )

                if not normalized_legal_name:
                    continue

                if (
                    normalized_legal_name
                    in seen_company_names
                ):
                    continue

                company_id = make_unique_company_id(
                    lei=lei,
                    used_company_ids=used_company_ids,
                )

                company_row = {
                    "company_id": company_id,
                    "legal_name": (
                        parsed["legal_name"]
                    ),
                    "brand_name": (
                        parsed["brand_name"]
                    ),
                    "city": parsed["city"],
                    "sector": parsed["sector"],
                    "tax_number": "",
                    "country": (
                        parsed["country"]
                        or country_code
                    ),
                    "lei": lei,
                    "source": "GLEIF",
                    "entity_status": (
                        parsed["entity_status"]
                    ),
                }

                companies.append(
                    company_row
                )

                seen_leis.add(lei)
                seen_company_names.add(
                    normalized_legal_name
                )

                seen_company_aliases: set[str] = set()

                for (
                    alias_text,
                    alias_type,
                    confidence,
                ) in parsed["aliases"]:
                    normalized_alias = (
                        normalize_company_name(
                            alias_text
                        )
                    )

                    if not normalized_alias:
                        continue

                    if (
                        normalized_alias
                        in seen_company_aliases
                    ):
                        continue

                    seen_company_aliases.add(
                        normalized_alias
                    )

                    aliases.append(
                        {
                            "alias_id": next_alias_id,
                            "company_id": company_id,
                            "alias_text": alias_text,
                            "alias_type": alias_type,
                            "source": "GLEIF",
                            "confidence": confidence,
                        }
                    )

                    next_alias_id += 1

                added_count += 1

            country_added_counts[
                country_code
            ] = added_count

            print(
                f"{country_code}: "
                f"{added_count} benzersiz şirket eklendi."
            )

    finally:
        session.close()

    report = {
        "created_at": datetime.now(
            UTC
        ).isoformat(),
        "target_company_count": target_count,
        "existing_company_count": len(
            existing_company_rows
        ),
        "final_company_count": len(companies),
        "existing_alias_count": len(
            existing_alias_rows
        ),
        "final_alias_count": len(aliases),
        "gleif_company_count": sum(
            country_added_counts.values()
        ),
        "country_added_counts": (
            country_added_counts
        ),
        "source": "GLEIF_API",
    }

    return companies, aliases, report


def backup_raw_files() -> Path:
    timestamp = datetime.now(
        UTC
    ).strftime("%Y%m%d_%H%M%S")

    target_directory = (
        BACKUP_DIRECTORY
        / f"before_gleif_{timestamp}"
    )

    target_directory.mkdir(
        parents=True,
        exist_ok=False,
    )

    if RAW_COMPANIES_PATH.exists():
        shutil.copy2(
            RAW_COMPANIES_PATH,
            target_directory / "companies.csv",
        )

    if RAW_ALIASES_PATH.exists():
        shutil.copy2(
            RAW_ALIASES_PATH,
            target_directory / "aliases.csv",
        )

    return target_directory


def apply_generated_dataset() -> Path:
    backup_directory = backup_raw_files()

    RAW_DATA_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    shutil.copy2(
        GENERATED_COMPANIES_PATH,
        RAW_COMPANIES_PATH,
    )

    shutil.copy2(
        GENERATED_ALIASES_PATH,
        RAW_ALIASES_PATH,
    )

    return backup_directory


def main() -> None:
    arguments = parse_arguments()

    if arguments.minimum < 1:
        raise ValueError(
            "--minimum en az 1 olmalıdır."
        )

    if arguments.target < arguments.minimum:
        raise ValueError(
            "--target, --minimum değerinden "
            "küçük olamaz."
        )

    print("Büyük şirket veri seti oluşturuluyor.")
    print(f"Hedef şirket sayısı: {arguments.target}")
    print(f"Minimum kabul: {arguments.minimum}")

    companies, aliases, report = create_dataset(
        target_count=arguments.target
    )

    final_company_count = len(companies)

    if final_company_count < arguments.minimum:
        raise RuntimeError(
            "Minimum şirket sayısına ulaşılamadı. "
            f"Beklenen en az {arguments.minimum}, "
            f"elde edilen {final_company_count}. "
            "Aktif veri dosyaları değiştirilmedi."
        )

    write_csv_rows(
        path=GENERATED_COMPANIES_PATH,
        fieldnames=COMPANY_FIELDS,
        rows=companies,
    )

    write_csv_rows(
        path=GENERATED_ALIASES_PATH,
        fieldnames=ALIAS_FIELDS,
        rows=aliases,
    )

    GENERATED_REPORT_PATH.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("Üretim tamamlandı.")
    print(
        f"Şirket sayısı: {len(companies)}"
    )
    print(
        f"Alias sayısı: {len(aliases)}"
    )

    print()
    print("Oluşturulan dosyalar:")
    print(GENERATED_COMPANIES_PATH)
    print(GENERATED_ALIASES_PATH)
    print(GENERATED_REPORT_PATH)

    if arguments.apply:
        backup_directory = (
            apply_generated_dataset()
        )

        print()
        print("Yeni veri seti aktif hâle getirildi.")
        print(
            "Eski dosyaların yedeği:"
        )
        print(backup_directory)

    else:
        print()
        print(
            "Aktif data/raw dosyaları değiştirilmedi."
        )
        print(
            "Kontrol ettikten sonra aynı komutu "
            "--apply parametresiyle çalıştırın."
        )


if __name__ == "__main__":
    main()