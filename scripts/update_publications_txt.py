#!/usr/bin/env python3
"""Regenerate titles.txt and links.txt from publications parquet."""

from __future__ import annotations

import argparse
import tempfile
import urllib.request
from pathlib import Path

import pyarrow.parquet as pq

DEFAULT_PARQUET_URL = (
    "https://huggingface.co/datasets/neurovlm/neuro_image_papers/resolve/main/"
    "publications.parquet"
)


def normalize_title(value: object) -> str:
    if value is None:
        return ""
    text = str(value).replace("\n", " ").replace("\r", " ")
    return " ".join(text.split())


def normalize_abstract(value: object) -> str:
    if value is None:
        return ""
    text = str(value).replace("\n", " ").replace("\r", " ")
    text = " ".join(text.split())
    if text.lower() in {"", "na", "n/a", "none"}:
        return ""
    return text


def normalize_link(value: object) -> str:
    if value is None:
        return ""
    doi = str(value).strip()
    if not doi:
        return ""
    if doi.startswith("http://") or doi.startswith("https://"):
        return doi
    return f"https://doi.org/{doi}"


def download_parquet(url: str, out_path: Path) -> None:
    with urllib.request.urlopen(url) as response:  # nosec - trusted URL input by user
        data = response.read()
    out_path.write_bytes(data)


def resolve_title_column(columns: set[str]) -> str:
    for candidate in ("titles", "title", "name"):
        if candidate in columns:
            return candidate
    raise ValueError(
        "Could not find a title column. Expected one of: titles, title, name."
    )


def resolve_abstract_column(columns: set[str]) -> str | None:
    for candidate in ("abstract", "abstracts", "abstract_text", "summary"):
        if candidate in columns:
            return candidate
    return None


def write_lines(path: Path, values: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(values) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build PMID-sorted links.txt and titles.txt from publications parquet."
    )
    parser.add_argument(
        "--parquet",
        default=DEFAULT_PARQUET_URL,
        help=(
            "Parquet source path or URL. Default: neuro_image_papers/publications.parquet"
        ),
    )
    parser.add_argument(
        "--out-dir",
        default="static/models",
        help="Output directory for links.txt and titles.txt (default: static/models)",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)

    with tempfile.TemporaryDirectory() as tmpdir:
        parquet_path = Path(args.parquet)
        if args.parquet.startswith("http://") or args.parquet.startswith("https://"):
            parquet_path = Path(tmpdir) / "publications.parquet"
            download_parquet(args.parquet, parquet_path)

        table = pq.read_table(parquet_path)
        columns = set(table.column_names)
        if "doi" not in columns:
            raise ValueError("Could not find required column: doi")
        if "pmid" not in columns:
            raise ValueError("Could not find required column: pmid")

        title_col = resolve_title_column(columns)
        abstract_col = resolve_abstract_column(columns)
        pmid_values = table["pmid"].to_pylist()
        doi_values = table["doi"].to_pylist()
        title_values = table[title_col].to_pylist()
        if abstract_col is None:
            abstract_values = [""] * len(pmid_values)
        else:
            abstract_values = table[abstract_col].to_pylist()

        rows = list(
            zip(pmid_values, doi_values, title_values, abstract_values, strict=True)
        )
        rows.sort(key=lambda row: int(row[0]))

        titles = [normalize_title(title) for _, _, title, _ in rows]
        links = [normalize_link(doi) for _, doi, _, _ in rows]
        pmids = [str(int(pmid)) for pmid, _, _, _ in rows]
        abstracts = [normalize_abstract(abstract) for _, _, _, abstract in rows]

    write_lines(out_dir / "titles.txt", titles)
    write_lines(out_dir / "links.txt", links)
    write_lines(out_dir / "pmids.txt", pmids)
    write_lines(out_dir / "abstracts.txt", abstracts)

    print(
        f"Wrote {len(titles)} rows to {out_dir / 'titles.txt'} and {out_dir / 'links.txt'}"
    )


if __name__ == "__main__":
    main()
