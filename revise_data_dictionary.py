#!/usr/bin/env python3
"""
Revise the auto-generated data dictionary using metadata from FAF5_metadata.xlsx.

This script:
- Loads the previously generated FAF5_Data_Dictionary.csv
- Reads all sheets from FAF5_metadata.xlsx and auto-detects columns describing variables
- Merges authoritative descriptions/types/units into the generated dictionary
- Writes FAF5_Data_Dictionary_REVISED.csv and FAF5_Data_Dictionary_REVISED.md
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
GEN_DICT_CSV = PROJECT_ROOT / "FAF5_Data_Dictionary.csv"
METADATA_XLSX = PROJECT_ROOT / "FAF5_metadata.xlsx"
OUT_REVISED_CSV = PROJECT_ROOT / "FAF5_Data_Dictionary_REVISED.csv"
OUT_REVISED_MD = PROJECT_ROOT / "FAF5_Data_Dictionary_REVISED.md"


def _standardize_column_names(columns: List[str]) -> List[str]:
    return [c.strip().lower().replace(" ", "_") for c in columns]


def _pick_first_present(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = set(_standardize_column_names(df.columns.tolist()))
    for cand in candidates:
        if cand in cols:
            return cand
    return None


def _standardize_df_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = _standardize_column_names(df.columns.tolist())
    return df


def _extract_metadata_from_sheet(df_raw: pd.DataFrame, known_column_keys: Optional[Set[str]] = None) -> Optional[pd.DataFrame]:
    """Attempt to extract a standardized metadata frame from a sheet.

    Returns a DataFrame with columns:
    - column_name
    - meta_description
    - meta_data_type
    - meta_category (optional)
    - units (optional)
    - allowed_values (optional)
    - source (optional)
    - notes (optional)
    """
    if df_raw is None or df_raw.empty:
        return None

    df = _standardize_df_columns(df_raw)

    # Likely variable name columns
    name_col = _pick_first_present(
        df,
        [
            "column_name",
            "column",
            "field_name",
            "field",
            "variable_name",
            "variable",
            "name",
        ],
    )
    if name_col is None and known_column_keys:
        # Heuristic: choose the column with the highest overlap with known dataset columns
        best_col = None
        best_overlap = 0
        for candidate in df.columns:
            series = df[candidate].astype(str).str.strip().str.lower()
            overlap = series.isin(list(known_column_keys)).sum()
            if overlap > best_overlap:
                best_overlap = overlap
                best_col = candidate
        # Require a minimal overlap threshold (at least 3 matches)
        if best_col is not None and best_overlap >= 3:
            name_col = best_col
    if name_col is None:
        return None

    # Likely description column
    desc_col = _pick_first_present(df, ["description", "desc", "label", "definition"]) 
    if desc_col is None:
        # Heuristic: pick the non-name text column with the longest average string length
        text_candidates = [c for c in df.columns if c != name_col]
        best_col = None
        best_len = 0
        for c in text_candidates:
            vals = df[c].dropna().astype(str)
            # Skip columns that look numeric-only
            if vals.empty:
                continue
            try:
                # If most values can be converted to float, skip as it's numeric
                numeric_like = (vals.str.match(r"^[\-\+]?\d+(?:\.\d+)?$", na=False)).mean()
                if numeric_like > 0.7:
                    continue
            except Exception:
                pass
            avg_len = vals.map(len).mean()
            if avg_len > best_len:
                best_len = avg_len
                best_col = c
        desc_col = best_col
    # Likely data type column
    dtype_col = _pick_first_present(df, ["data_type", "type", "dtype", "format"])
    # Optional supportive columns
    category_col = _pick_first_present(df, ["category", "group", "domain"])
    units_col = _pick_first_present(df, ["units", "unit"])
    allowed_values_col = _pick_first_present(df, ["allowed_values", "values", "valid_values", "range"])
    source_col = _pick_first_present(df, ["source", "origin"])
    notes_col = _pick_first_present(df, ["notes", "comments", "remark", "remarks"])

    keep_cols = {
        "column_name": name_col,
        "meta_description": desc_col,
        "meta_data_type": dtype_col,
        "meta_category": category_col,
        "units": units_col,
        "allowed_values": allowed_values_col,
        "source": source_col,
        "notes": notes_col,
    }

    # Build standardized frame
    out_cols: Dict[str, List] = {k: [] for k in keep_cols.keys()}
    for _, row in df.iterrows():
        name_val = row.get(name_col)
        if pd.isna(name_val):
            continue
        # Normalize name to exact dataset column naming conventions (lowercase)
        col_name = str(name_val).strip()
        # Preserve original case if provided in Excel, but match later case-insensitively
        out_cols["column_name"].append(col_name)
        out_cols["meta_description"].append(row.get(desc_col) if desc_col else None)
        out_cols["meta_data_type"].append(row.get(dtype_col) if dtype_col else None)
        out_cols["meta_category"].append(row.get(category_col) if category_col else None)
        out_cols["units"].append(row.get(units_col) if units_col else None)
        out_cols["allowed_values"].append(row.get(allowed_values_col) if allowed_values_col else None)
        out_cols["source"].append(row.get(source_col) if source_col else None)
        out_cols["notes"].append(row.get(notes_col) if notes_col else None)

    meta_df = pd.DataFrame(out_cols)
    if meta_df.empty:
        return None

    # Standardize column_name for matching
    meta_df["_column_key"] = meta_df["column_name"].astype(str).str.strip().str.lower()
    meta_df = meta_df.drop_duplicates(subset=["_column_key"])
    return meta_df


def load_combined_metadata(xlsx_path: Path, known_column_keys: Set[str]) -> pd.DataFrame:
    xls = pd.ExcelFile(xlsx_path)
    meta_frames: List[pd.DataFrame] = []
    # Prefer a sheet literally named "Data Dictionary" if present
    preferred_order = [s for s in xls.sheet_names if str(s).strip().lower() == "data dictionary"]
    remaining = [s for s in xls.sheet_names if s not in preferred_order]
    sheet_order = preferred_order + remaining
    for sheet in sheet_order:
        try:
            df_sheet = pd.read_excel(xls, sheet_name=sheet, engine="openpyxl")
        except (ValueError, FileNotFoundError, ImportError):
            continue
        extracted = _extract_metadata_from_sheet(df_sheet, known_column_keys)
        if extracted is not None and not extracted.empty:
            meta_frames.append(extracted)

    if not meta_frames:
        raise RuntimeError("No usable metadata sheets found in FAF5_metadata.xlsx")

    # Combine; prefer first occurrence
    combined = pd.concat(meta_frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["_column_key"], keep="first")
    return combined


def _detect_code_label_columns(df: pd.DataFrame) -> Optional[Tuple[str, str]]:
    """Heuristically detect (code, label) columns for lookup sheets.

    Returns (code_col, label_col) or None if cannot be detected.
    """
    cand_code = [
        "code", "id", "fips", "state_fips", "sctg", "sctg2", "mode", "value", "trade_type",
        "dist_band", "distance_band", "faf_zone", "faf_zone_code"
    ]
    cand_label = [
        "name", "state", "state_name", "label", "description", "desc", "title", "mode_name",
        "commodity", "trade", "band", "region"
    ]

    df_std = _standardize_df_columns(df)

    # Direct matches first
    code_col = _pick_first_present(df_std, cand_code)
    label_col = _pick_first_present(df_std, cand_label)
    if code_col and label_col:
        return code_col, label_col

    # Fallback: choose one mostly numeric column as code, and one texty column as label
    code_guess = None
    label_guess = None
    best_text_len = 0.0
    for c in df_std.columns:
        vals = df_std[c].dropna().astype(str)
        if vals.empty:
            continue
        numeric_like = (vals.str.match(r"^[\-\+]?\d+(?:\.\d+)?$", na=False)).mean()
        if numeric_like > 0.7 and code_guess is None:
            code_guess = c
        else:
            avg_len = vals.map(len).mean()
            if avg_len > best_text_len:
                best_text_len = avg_len
                label_guess = c
    if code_guess and label_guess:
        return code_guess, label_guess
    return None


def _read_lookup_map(xls: pd.ExcelFile, sheet_name: str) -> Optional[Dict[str, str]]:
    try:
        df = pd.read_excel(xls, sheet_name=sheet_name, engine="openpyxl")
    except (ValueError, OSError, ImportError):
        return None
    if df is None or df.empty:
        return None
    cols = _detect_code_label_columns(df)
    if cols is None:
        return None
    code_col, label_col = cols
    df_std = _standardize_df_columns(df)
    mapping: Dict[str, str] = {}
    for _, row in df_std.iterrows():
        k = row.get(code_col)
        v = row.get(label_col)
        if pd.isna(k) or pd.isna(v):
            continue
        mapping[str(k).strip()] = str(v).strip()
    return mapping if mapping else None


def enrich_with_code_mappings(revised: pd.DataFrame, xlsx_path: Path) -> pd.DataFrame:
    """Attach human-readable allowed values for coded fields from known sheets."""
    xls = pd.ExcelFile(xlsx_path)
    # Known sheets from the screenshot
    sheet_names = {s.strip().lower(): s for s in xls.sheet_names}

    lookups: Dict[str, Dict[str, str]] = {}
    # Build mappings if sheets exist
    def add_map(key: str, sheet_key: str):
        sheet = sheet_names.get(sheet_key)
        if sheet:
            m = _read_lookup_map(xls, sheet)
            if m:
                lookups[key] = m

    add_map("state", "state")
    add_map("faf_zone_domestic", "faf zone (domestic)")
    add_map("faf_zone_foreign", "faf zone (foreign)")
    add_map("sctg2", "commodity (sctg2)")
    add_map("mode", "mode")
    add_map("trade_type", "trade type")
    add_map("dist_band", "distance band")

    def map_to_string(mapping: Dict[str, str]) -> str:
        items = sorted(mapping.items(), key=lambda x: str(x[0]))
        preview = "; ".join([f"{k}={v}" for k, v in items])
        return preview

    # Apply to specific columns in the dictionary
    col_to_lookup_key = {
        "dms_origst": "state",
        "dms_destst": "state",
        "fr_orig": "faf_zone_foreign",
        "fr_dest": "faf_zone_foreign",
        "sctg2": "sctg2",
        "dms_mode": "mode",
        "fr_inmode": "mode",
        "fr_outmode": "mode",
        "trade_type": "trade_type",
        "dist_band": "dist_band",
    }

    revised = revised.copy()
    revised["_column_key"] = revised["Column_Name"].astype(str).str.strip().str.lower()

    for dataset_col, lookup_key in col_to_lookup_key.items():
        if lookup_key not in lookups:
            continue
        mapping = lookups[lookup_key]
        allowed_val_str = map_to_string(mapping)
        mask = revised["_column_key"] == dataset_col
        if mask.any():
            # Fill allowed_values and enhance description with mapping preview
            existing_desc = revised.loc[mask, "Description"].astype(str).fillna("")
            revised.loc[mask, "allowed_values"] = allowed_val_str
            # Only append mapping preview if not already present
            revised.loc[mask, "Description"] = [
                (d + (" " if d and not d.endswith(".") else "") + f"Codes: {allowed_val_str}").strip()
                if allowed_val_str and ("Codes:" not in d) else d
                for d in existing_desc
            ]

    # Drop helper key
    revised = revised.drop(columns=["_column_key"], errors="ignore")
    return revised


def revise_dictionary(gen_csv_path: Path, meta_xlsx_path: Path) -> pd.DataFrame:
    if not gen_csv_path.exists():
        raise FileNotFoundError(f"Generated dictionary not found: {gen_csv_path}")
    if not meta_xlsx_path.exists():
        raise FileNotFoundError(f"Metadata workbook not found: {meta_xlsx_path}")

    gen = pd.read_csv(gen_csv_path)
    if "Column_Name" not in gen.columns:
        raise ValueError("Generated dictionary missing 'Column_Name' column")

    # Key for case-insensitive merge
    gen["_column_key"] = gen["Column_Name"].astype(str).str.strip().str.lower()

    known_keys = set(gen["_column_key"].astype(str))
    meta = load_combined_metadata(meta_xlsx_path, known_keys)

    merged = gen.merge(meta, on="_column_key", how="left", suffixes=("", "_meta"))

    # Prefer metadata description and data type if provided
    def prefer(left, right):
        return right if pd.notna(right) and str(right).strip() != "" else left

    merged["Description"] = [
        prefer(l, r) for l, r in zip(merged.get("Description"), merged.get("meta_description"))
    ]

    if "Data_Type" in merged.columns and "meta_data_type" in merged.columns:
        merged["Data_Type"] = [
            prefer(l, r) for l, r in zip(merged.get("Data_Type"), merged.get("meta_data_type"))
        ]

    # Add optional metadata fields to output
    optional_cols = [
        "meta_category",
        "units",
        "allowed_values",
        "source",
        "notes",
    ]
    for col in optional_cols:
        if col not in merged.columns:
            merged[col] = None

    # Reorder columns for readability
    base_order = [
        "Column_Name",
        "Data_Type",
        "Description",
        "Category" if "Category" in merged.columns else None,
        "Missing_Count" if "Missing_Count" in merged.columns else None,
        "Missing_Percent" if "Missing_Percent" in merged.columns else None,
        "Unique_Values" if "Unique_Values" in merged.columns else None,
        "Mean" if "Mean" in merged.columns else None,
        "Median" if "Median" in merged.columns else None,
        "Std_Dev" if "Std_Dev" in merged.columns else None,
        "Min" if "Min" in merged.columns else None,
        "Max" if "Max" in merged.columns else None,
        "Zero_Count" if "Zero_Count" in merged.columns else None,
        # Metadata extras
        "meta_category",
        "units",
        "allowed_values",
        "source",
        "notes",
    ]
    base_order = [c for c in base_order if c is not None and c in merged.columns]
    other_cols = [c for c in merged.columns if c not in base_order and not c.startswith("_")]
    final_cols = base_order + other_cols
    revised = merged[final_cols]

    return revised


def write_markdown(doc: pd.DataFrame, out_path: Path) -> None:
    lines: List[str] = []
    lines.append("# FAF5.7 Dataset Data Dictionary (Revised)\n")
    lines.append("This document merges the auto-generated dictionary with authoritative metadata from `FAF5_metadata.xlsx`.\n\n")

    # High-level summary
    lines.append("## Dataset Overview\n\n")
    lines.append(f"- **Total Variables**: {len(doc):,}\n")
    if "Category" in doc.columns:
        lines.append(f"- **Categories**: {doc['Category'].nunique()}\n")
    lines.append("\n")

    # Table header
    lines.append("## Variable Descriptions\n\n")
    lines.append("| Column Name | Data Type | Category | Description | Units |")
    lines.append("\n|---|---|---|---|---|\n")

    for _, row in doc.iterrows():
        col = str(row.get("Column_Name", "")).strip()
        dtype = str(row.get("Data_Type", "")).strip()
        cat = str(row.get("Category", row.get("meta_category", ""))).strip()
        desc = str(row.get("Description", "")).strip().replace("\n", " ")
        units = str(row.get("units", "")).strip()
        lines.append(f"| {col} | {dtype} | {cat} | {desc} | {units} |\n")

    out_path.write_text("".join(lines), encoding="utf-8")


def main() -> int:
    try:
        revised = revise_dictionary(GEN_DICT_CSV, METADATA_XLSX)
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"Error: {e}")
        return 1

    # Enrich with code mappings from the known sheets
    try:
        revised = enrich_with_code_mappings(revised, METADATA_XLSX)
    except Exception:
        # Non-fatal: proceed without enrichment if lookups fail
        pass

    revised.to_csv(OUT_REVISED_CSV, index=False)
    write_markdown(revised, OUT_REVISED_MD)
    print(f"✅ Wrote revised data dictionary: {OUT_REVISED_CSV.name}")
    print(f"✅ Wrote revised markdown: {OUT_REVISED_MD.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())


