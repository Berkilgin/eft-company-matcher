from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    BigInteger,
    DateTime,
    Float,
    ForeignKey,
    Index,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import (
    Mapped,
    mapped_column,
    relationship,
)

from app.database.base import Base


class Company(Base):
    __tablename__ = "companies"

    company_id: Mapped[int] = mapped_column(
        BigInteger,
        primary_key=True,
        autoincrement=False,
    )

    legal_name: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    brand_name: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        default="",
    )

    city: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        default="",
    )

    sector: Mapped[str] = mapped_column(
        String(200),
        nullable=False,
        default="",
    )

    tax_number: Mapped[str | None] = mapped_column(
        String(20),
        nullable=True,
        index=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    aliases: Mapped[list["CompanyAlias"]] = relationship(
        back_populates="company",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class CompanyAlias(Base):
    __tablename__ = "company_aliases"

    __table_args__ = (
        UniqueConstraint(
            "company_id",
            "normalized_alias",
            name="uq_company_alias_normalized",
        ),
        Index(
            "ix_company_aliases_normalized_trgm",
            "normalized_alias",
            postgresql_using="gin",
            postgresql_ops={
                "normalized_alias": "gin_trgm_ops",
            },
        ),
    )

    alias_id: Mapped[int] = mapped_column(
        BigInteger,
        primary_key=True,
        autoincrement=True,
    )

    company_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey(
            "companies.company_id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )

    alias_text: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    normalized_alias: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    alias_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
    )

    source: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="MANUAL",
    )

    confidence: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        default=1.0,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    company: Mapped[Company] = relationship(
        back_populates="aliases",
    )