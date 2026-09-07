from __future__ import annotations


# EFT açıklamasında şirketi tanımlamaya çoğunlukla katkı sağlamayan
# bankacılık ve işlem kelimeleri.
BANKING_STOPWORDS: frozenset[str] = frozenset(
    {
        "eft",
        "havale",
        "hvl",
        "virman",
        "transfer",
        "gonderim",
        "gonderen",
        "alici",
        "odeme",
        "odemesi",
        "odem",
        "odm",
        "tahsilat",
        "tahsilati",
        "fatura",
        "fat",
        "bedel",
        "bedeli",
        "ref",
        "referans",
        "dekont",
        "aciklama",
        "islem",
        "islemi",
        "masraf",
        "komisyon",
        "iade",
        "tutar",
        "tutari",
        "no",
        "nolu",
        "numarali",
        "icin",
        "ait",
    }
)


# EFT açıklamalarında sık görülen şirket ve sektör kısaltmaları.
# Bunlar silinmez, daha açık biçime dönüştürülür.
ABBREVIATIONS: dict[str, str] = {
    "muh": "muhendislik",
    "muhend": "muhendislik",
    "ins": "insaat",
    "insaat": "insaat",
    "san": "sanayi",
    "tic": "ticaret",
    "paz": "pazarlama",
    "oto": "otomotiv",
    "otom": "otomotiv",
    "loj": "lojistik",
    "tek": "teknoloji",
    "tekn": "teknoloji",
    "elk": "elektrik",
    "elek": "elektrik",
    "elektr": "elektrik",
    "ener": "enerji",
    "dag": "dagitim",
    "dagit": "dagitim",
    "pet": "petrol",
    "raf": "rafineri",
    "tur": "turizm",
    "gyd": "gayrimenkul",
    "gm": "gayrimenkul",
    "dan": "danismanlik",
    "danis": "danismanlik",
    "bil": "bilisim",
    "bilis": "bilisim",
    "med": "medikal",
    "sag": "saglik",
    "kim": "kimya",
    "mak": "makina",
    "makine": "makina",
    "mob": "mobilya",
    "tar": "tarim",
    "nak": "nakliyat",
    "tas": "tasimacilik",
    "ith": "ithalat",
    "ihr": "ihracat",
}


# Patternlar ASCII'ye çevrilmiş ve küçük harf yapılmış metin
# üzerinde çalıştırılacaktır.
#
# Sıralama önemlidir: önce daha uzun ve özel biçimler işlenir.
LEGAL_SUFFIX_PATTERNS: tuple[tuple[str, str], ...] = (
    (
        "LIMITED_SIRKETI",
        r"\blimited\s+sirketi\b",
    ),
    (
        "LIMITED_SIRKETI",
        r"\bltd\s*\.?\s*sti\s*\.?\b",
    ),
    (
        "ANONIM_SIRKETI",
        r"\banonim\s+sirketi\b",
    ),
    (
        "ANONIM_SIRKETI",
        r"\ba\s*\.?\s*s\s*\.?\b",
    ),
    (
        "KOLLEKTIF_SIRKET",
        r"\bkollektif\s+sirketi?\b",
    ),
    (
        "KOMANDIT_SIRKET",
        r"\bkomandit\s+sirketi?\b",
    ),
    (
        "KOOPERATIF",
        r"\bkooperatifi?\b",
    ),
    (
        "LIMITED_SIRKETI",
        r"\bltd\b",
    ),
)