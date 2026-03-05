#!/usr/bin/env python3
"""Sort publication metadata and latent embeddings by PMID.

This keeps publication list indices aligned with the embedding row indices used by the app.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pyarrow.parquet as pq
import torch
from safetensors.torch import save_file


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


def resolve_title_column(columns: set[str]) -> str:
    for candidate in ("titles", "title", "name"):
        if candidate in columns:
            return candidate
    raise ValueError("Missing title column. Expected one of: titles, title, name")


def resolve_abstract_column(columns: set[str]) -> str | None:
    for candidate in ("abstract", "abstracts", "abstract_text", "summary"):
        if candidate in columns:
            return candidate
    return None


def write_lines(path: Path, values: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(values) + "\n", encoding="utf-8")


def pmid_to_int_list(values: object) -> list[int]:
    if isinstance(values, torch.Tensor):
        return [int(x) for x in values.detach().cpu().flatten().tolist()]
    return [int(x) for x in values]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sort metadata and latent embeddings using PMID as key."
    )
    parser.add_argument(
        "--publications",
        default="/tmp/publications.parquet",
        help="Path to publications.parquet",
    )
    parser.add_argument(
        "--latent-pt",
        default="static/models/latent_text_specter2_adhoc_query.pt",
        help="Path to latent .pt with keys 'latent' and 'pmid'",
    )
    parser.add_argument(
        "--out-dir",
        default="static/models",
        help="Output directory for titles/links/pmids txt files",
    )
    parser.add_argument(
        "--latent-safetensors-out",
        default="static/models/latent_text_specter2_adhoc_query.safetensors",
        help="Output safetensors path for sorted latent embeddings",
    )
    parser.add_argument(
        "--latent-pt-out",
        default="",
        help="Optional output path for sorted latent .pt",
    )
    args = parser.parse_args()

    table = pq.read_table(Path(args.publications))
    columns = set(table.column_names)
    required = {"pmid", "doi"}
    if not required.issubset(columns):
        missing = sorted(required - columns)
        raise ValueError(f"Missing required columns in publications parquet: {missing}")

    title_col = resolve_title_column(columns)

    abstract_col = resolve_abstract_column(columns)

    pub_pmids = [int(x) for x in table["pmid"].to_pylist()]
    pub_doi = table["doi"].to_pylist()
    pub_title = table[title_col].to_pylist()
    if abstract_col is None:
        pub_abstract = [""] * len(pub_pmids)
    else:
        pub_abstract = table[abstract_col].to_pylist()

    if len(pub_pmids) != len(set(pub_pmids)):
        raise ValueError("Publications parquet contains duplicate PMID values")

    pub_rows = list(zip(pub_pmids, pub_doi, pub_title, pub_abstract, strict=True))
    pub_rows.sort(key=lambda row: row[0])

    latent_obj = torch.load(args.latent_pt, map_location="cpu", weights_only=False)
    if not isinstance(latent_obj, dict) or "latent" not in latent_obj or "pmid" not in latent_obj:
        raise ValueError("latent .pt must be a dict with keys: latent, pmid")

    latent_tensor = latent_obj["latent"].detach().cpu()
    latent_pmids = pmid_to_int_list(latent_obj["pmid"])

    if len(latent_pmids) != latent_tensor.shape[0]:
        raise ValueError(
            "Latent pmid length does not match embedding row count: "
            f"{len(latent_pmids)} vs {latent_tensor.shape[0]}"
        )
    if len(latent_pmids) != len(set(latent_pmids)):
        raise ValueError("Latent checkpoint contains duplicate PMID values")

    latent_row_for_pmid = {pmid: idx for idx, pmid in enumerate(latent_pmids)}

    sorted_pmids = [pmid for pmid, _, _, _ in pub_rows]
    missing_in_latent = [pmid for pmid in sorted_pmids if pmid not in latent_row_for_pmid]
    if missing_in_latent:
        raise ValueError(f"PMIDs missing in latent checkpoint: {missing_in_latent[:10]}")

    sorted_indices = [latent_row_for_pmid[pmid] for pmid in sorted_pmids]
    sorted_tensor = latent_tensor[torch.tensor(sorted_indices, dtype=torch.long)]

    titles = [normalize_title(title) for _, _, title, _ in pub_rows]
    links = [normalize_link(doi) for _, doi, _, _ in pub_rows]
    abstracts = [normalize_abstract(abstract) for _, _, _, abstract in pub_rows]
    pmids_txt = [str(pmid) for pmid in sorted_pmids]

    out_dir = Path(args.out_dir)
    write_lines(out_dir / "titles.txt", titles)
    write_lines(out_dir / "links.txt", links)
    write_lines(out_dir / "pmids.txt", pmids_txt)
    write_lines(out_dir / "abstracts.txt", abstracts)

    save_file({"latent": sorted_tensor}, args.latent_safetensors_out)

    if args.latent_pt_out:
        torch.save({"latent": sorted_tensor, "pmid": sorted_pmids}, args.latent_pt_out)

    print(f"Wrote sorted metadata rows: {len(sorted_pmids)}")
    print(f"Wrote sorted latent safetensors: {args.latent_safetensors_out}")


if __name__ == "__main__":
    main()
