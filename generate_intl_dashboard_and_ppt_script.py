#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages


ROOT = Path(__file__).resolve().parent
DATA_CSV = ROOT / "outputs" / "FAF5.7_State_fr_orig_ge_800.csv"
META_XLSX = ROOT / "FAF5_metadata.xlsx"
OUT_DIR = ROOT / "outputs"
MANIFEST = OUT_DIR / "dashboard_manifest.json"
PPT_SCRIPT = OUT_DIR / "powerpoint_content_script.md"
DASHBOARD_PDF = OUT_DIR / "intl_dashboard_full.pdf"
FIELDS_DOC = OUT_DIR / "data_fields_used.md"


def ensure_outputs_dir() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(DATA_CSV)
    # Base condition: imports/exports if available
    cond_trade = df["trade_type"].isin([2, 3]) if "trade_type" in df.columns else pd.Series(True, index=df.index)
    # International origin/destination constraint: FR regions 801–808
    cond_fr = pd.Series(False, index=df.index)
    if "fr_orig" in df.columns:
        cond_fr = cond_fr | (df["fr_orig"].between(801, 808, inclusive="both"))
    if "fr_dest" in df.columns:
        cond_fr = cond_fr | (df["fr_dest"].between(801, 808, inclusive="both"))
    # Fallback if no FR columns
    if cond_fr.sum() == 0:
        cond_fr = (df.get("fr_orig").notna()) | (df.get("fr_dest").notna()) if ("fr_orig" in df.columns or "fr_dest" in df.columns) else pd.Series(True, index=df.index)
    df_intl = df[cond_trade & cond_fr].copy()
    return df, df_intl


def load_codebooks() -> Dict[str, Dict[str, str]]:
    maps: Dict[str, Dict[str, str]] = {}
    try:
        xls = pd.ExcelFile(META_XLSX)
        # Modes
        if "Mode" in xls.sheet_names:
            df = pd.read_excel(xls, sheet_name="Mode", engine="openpyxl")
            cols = {c.strip().lower(): c for c in df.columns}
            if "numeric label" in cols and "description" in cols:
                maps["mode"] = {str(k): str(v) for k, v in zip(df[cols["numeric label"]], df[cols["description"]]) if pd.notna(k) and pd.notna(v)}
        # Foreign FAF regions
        if "FAF Zone (Foreign)" in xls.sheet_names:
            df = pd.read_excel(xls, sheet_name="FAF Zone (Foreign)", engine="openpyxl")
            cols = {c.strip().lower(): c for c in df.columns}
            if "numeric label" in cols and "description" in cols:
                maps["fr"] = {str(k): str(v) for k, v in zip(df[cols["numeric label"]], df[cols["description"]]) if pd.notna(k) and pd.notna(v)}
        # Commodity SCTG2
        if "Commodity (SCTG2)" in xls.sheet_names:
            df = pd.read_excel(xls, sheet_name="Commodity (SCTG2)", engine="openpyxl")
            cols = {c.strip().lower(): c for c in df.columns}
            if "numeric label" in cols and "description" in cols:
                maps["sctg2"] = {str(int(k)): str(v) for k, v in zip(df[cols["numeric label"]], df[cols["description"]]) if pd.notna(k) and pd.notna(v)}
    except (ValueError, OSError, ImportError):
        # Codebooks are optional
        pass
    return maps


def read_data_dictionary() -> Dict[str, Dict[str, str]]:
    """Load the data dictionary (revised preferred) into a mapping of column -> {description, category}."""
    paths = [ROOT / "FAF5_Data_Dictionary_REVISED.csv", ROOT / "FAF5_Data_Dictionary.csv"]
    for p in paths:
        if p.exists():
            try:
                dd = pd.read_csv(p)
                name_col = "Column_Name" if "Column_Name" in dd.columns else ("column_name" if "column_name" in dd.columns else None)
                if name_col is None:
                    continue
                desc_col = "Description" if "Description" in dd.columns else ("description" if "description" in dd.columns else None)
                cat_col = "Category" if "Category" in dd.columns else ("category" if "category" in dd.columns else None)
                mapping: Dict[str, Dict[str, str]] = {}
                for _, row in dd.iterrows():
                    col = str(row.get(name_col, "")).strip()
                    if not col:
                        continue
                    mapping[col] = {
                        "description": str(row.get(desc_col, "")).strip() if desc_col else "",
                        "category": str(row.get(cat_col, "")).strip() if cat_col else "",
                    }
                return mapping
            except Exception:
                continue
    return {}


def compute_derived(df_intl: pd.DataFrame) -> pd.DataFrame:
    df = df_intl.copy()
    tons_years = [c for c in df.columns if c.startswith("tons_") and c.split("_")[-1].isdigit()]
    year_nums = [int(c.split("_")[-1]) for c in tons_years]
    y_2017_2023 = [c for c, y in zip(tons_years, year_nums) if 2017 <= y <= 2023]
    if y_2017_2023:
        df["tons_volatility"] = df[y_2017_2023].std(axis=1) / (df[y_2017_2023].mean(axis=1) + 1e-6)
    else:
        df["tons_volatility"] = np.nan
    if {"tons_2017", "tons_2023"}.issubset(df.columns):
        df["tons_growth_17_23"] = (df["tons_2023"] - df["tons_2017"]) / (df["tons_2017"] + 1e-6)
    else:
        df["tons_growth_17_23"] = np.nan
    if {"value_2023", "tons_2023"}.issubset(df.columns):
        df["value_density_2023"] = df["value_2023"] / (df["tons_2023"] + 1e-6)
    else:
        df["value_density_2023"] = np.nan
    # Concentration: HHI by foreign region over destinations (tons_2023)
    if set(["fr_orig", "dms_destst", "tons_2023"]).issubset(df.columns):
        hhi = (
            df.groupby(["fr_orig", "dms_destst"], dropna=True)["tons_2023"].sum()
            .groupby(level=0)
            .apply(lambda s: (s / (s.sum() + 1e-9)) ** 2)
            .groupby(level=0)
            .sum()
            .rename("hhi")
            .reset_index()
        )
        df = df.merge(hhi, on=["fr_orig"], how="left")
    else:
        df["hhi"] = np.nan
    return df


def fips_state_map() -> Dict[str, str]:
    # FIPS 1..56 (incl. DC and territories present in FAF)
    return {
        "1": "Alabama", "2": "Alaska", "4": "Arizona", "5": "Arkansas", "6": "California",
        "8": "Colorado", "9": "Connecticut", "10": "Delaware", "11": "District of Columbia",
        "12": "Florida", "13": "Georgia", "15": "Hawaii", "16": "Idaho", "17": "Illinois",
        "18": "Indiana", "19": "Iowa", "20": "Kansas", "21": "Kentucky", "22": "Louisiana",
        "23": "Maine", "24": "Maryland", "25": "Massachusetts", "26": "Michigan",
        "27": "Minnesota", "28": "Mississippi", "29": "Missouri", "30": "Montana",
        "31": "Nebraska", "32": "Nevada", "33": "New Hampshire", "34": "New Jersey",
        "35": "New Mexico", "36": "New York", "37": "North Carolina", "38": "North Dakota",
        "39": "Ohio", "40": "Oklahoma", "41": "Oregon", "42": "Pennsylvania",
        "44": "Rhode Island", "45": "South Carolina", "46": "South Dakota", "47": "Tennessee",
        "48": "Texas", "49": "Utah", "50": "Vermont", "51": "Virginia", "53": "Washington",
        "54": "West Virginia", "55": "Wisconsin", "56": "Wyoming"
    }


def winsorize(series: pd.Series, low_q: float = 0.01, high_q: float = 0.99) -> pd.Series:
    s = series.replace([np.inf, -np.inf], np.nan).dropna()
    if s.empty:
        return series
    lo, hi = s.quantile(low_q), s.quantile(high_q)
    return series.clip(lo, hi)


def save_chart(fig: plt.Figure, filename: str, pdf: PdfPages) -> Path:
    fig.tight_layout()
    path = OUT_DIR / filename
    fig.savefig(path, dpi=180)
    pdf.savefig(fig)
    plt.close(fig)
    return path


def build_dashboard(df: pd.DataFrame, df_intl: pd.DataFrame, maps: Dict[str, Dict[str, str]]) -> Tuple[Dict[str, str], Path]:
    sns.set_style("whitegrid")
    manifest: Dict[str, str] = {}
    # Choose a writable PDF path (avoid overwrite when file is open)
    pdf_path = DASHBOARD_PDF
    if pdf_path.exists():
        i = 1
        while True:
            candidate = pdf_path.with_name(pdf_path.stem + f"_{i}" + pdf_path.suffix)
            if not candidate.exists():
                pdf_path = candidate
                break
            i += 1

    with PdfPages(pdf_path) as pdf:
        # 1. Tons trend (intl)
        tons_cols = sorted([c for c in df_intl.columns if c.startswith("tons_") and c.split("_")[-1].isdigit()], key=lambda c: int(c.split("_")[-1]))
        tons_series = pd.DataFrame({"year": [int(c.split("_")[-1]) for c in tons_cols], "tons": [df_intl[c].sum() for c in tons_cols]})
        fig, ax = plt.subplots(figsize=(7,4))
        y = tons_series["tons"]/1e6
        y_clip = winsorize(y, 0.01, 0.995)
        ax.plot(tons_series["year"], y_clip, marker="o")
        ax.set_xlabel("Year"); ax.set_ylabel("Million Tons"); ax.set_title("International Tons Trend")
        ax.set_ylim(bottom=0)
        manifest[str(save_chart(fig, "01_tons_trend.png", pdf))] = "International tons from 2017-2050."

        # 2. Value trend (intl)
        val_cols = sorted([c for c in df_intl.columns if c.startswith("value_") and c.split("_")[-1].isdigit()], key=lambda c: int(c.split("_")[-1]))
        val_series = pd.DataFrame({"year": [int(c.split("_")[-1]) for c in val_cols], "value_thousands": [df_intl[c].sum() for c in val_cols]})
        fig, ax = plt.subplots(figsize=(7,4))
        y = val_series["value_thousands"]/1e6
        y_clip = winsorize(y, 0.01, 0.995)
        ax.plot(val_series["year"], y_clip, marker="o", color="darkgreen")
        ax.set_xlabel("Year"); ax.set_ylabel("Billions (Thousand USD)"); ax.set_title("International Value Trend")
        ax.set_ylim(bottom=0)
        manifest[str(save_chart(fig, "02_value_trend.png", pdf))] = "International value from 2017-2050 (thousands USD)."

        # 3. Transport miles trend (intl)
        tm_cols = sorted([c for c in df_intl.columns if c.startswith("tmiles_") and c.split("_")[-1].isdigit()], key=lambda c: int(c.split("_")[-1]))
        if tm_cols:
            tm_series = pd.DataFrame({"year": [int(c.split("_")[-1]) for c in tm_cols], "tmiles": [df_intl[c].sum() for c in tm_cols]})
            fig, ax = plt.subplots(figsize=(7,4))
            y = tm_series["tmiles"]/1e6
            y_clip = winsorize(y, 0.01, 0.995)
            ax.plot(tm_series["year"], y_clip, marker="o", color="#9467bd")
            ax.set_xlabel("Year"); ax.set_ylabel("M Miles"); ax.set_title("International Transport Miles")
            ax.set_ylim(bottom=0)
            manifest[str(save_chart(fig, "03_tmiles_trend.png", pdf))] = "International transport miles by year."

        # Aggregations for 2023
        if {"tons_2023", "value_2023"}.issubset(df_intl.columns):
            # 4. Mode share by tons 2023
            if "dms_mode" in df_intl.columns:
                mode_tons = df_intl.groupby("dms_mode")["tons_2023"].sum().reset_index().sort_values("tons_2023", ascending=False)
                mode_tons["label"] = mode_tons["dms_mode"].astype(str).map(maps.get("mode", {})).fillna(mode_tons["dms_mode"].astype(str))
                fig, ax = plt.subplots(figsize=(7,4))
                sns.barplot(data=mode_tons, x="label", y="tons_2023", ax=ax, color="#1f77b4"); ax.set_xlabel("Mode"); ax.set_ylabel("Tons 2023"); ax.set_title("Mode Share by Tons (2023)"); ax.tick_params(axis='x', rotation=45)
                manifest[str(save_chart(fig, "04_mode_tons_2023.png", pdf))] = "Mode share by tons in 2023."

            # 5. Mode share by value 2023
            if "dms_mode" in df_intl.columns:
                mode_val = df_intl.groupby("dms_mode")["value_2023"].sum().reset_index().sort_values("value_2023", ascending=False)
                mode_val["label"] = mode_val["dms_mode"].astype(str).map(maps.get("mode", {})).fillna(mode_val["dms_mode"].astype(str))
                fig, ax = plt.subplots(figsize=(7,4))
                sns.barplot(data=mode_val, x="label", y="value_2023", ax=ax, color="#2ca02c"); ax.set_xlabel("Mode"); ax.set_ylabel("Value 2023 (kUSD)"); ax.set_title("Mode Share by Value (2023)"); ax.tick_params(axis='x', rotation=45)
                manifest[str(save_chart(fig, "05_mode_value_2023.png", pdf))] = "Mode share by value in 2023."

            # 6. Trade type distribution (records)
            if "trade_type" in df_intl.columns:
                trade_map = {1: "Domestic", 2: "Import", 3: "Export"}
                trade_counts = df_intl["trade_type"].map(trade_map).value_counts().reset_index()
                trade_counts.columns = ["trade_type", "records"]
                fig, ax = plt.subplots(figsize=(7,4))
                sns.barplot(data=trade_counts, x="trade_type", y="records", ax=ax, color="#ff7f0e"); ax.set_xlabel(""); ax.set_ylabel("Records"); ax.set_title("Trade Type Mix")
                manifest[str(save_chart(fig, "06_trade_mix.png", pdf))] = "Record counts by trade type."

            # 7. Top foreign regions by tons 2023
            if "fr_orig" in df_intl.columns:
                fr_tons = df_intl.groupby("fr_orig")["tons_2023"].sum().reset_index().sort_values("tons_2023", ascending=False).head(10)
                fr_tons["region"] = fr_tons["fr_orig"].astype(str).map(maps.get("fr", {})).fillna(fr_tons["fr_orig"].astype(str))
                fig, ax = plt.subplots(figsize=(7,5))
                sns.barplot(data=fr_tons, y="region", x="tons_2023", ax=ax, color="#4e79a7")
                ax.set_xlabel("Tons 2023"); ax.set_ylabel(""); ax.set_title("Top Foreign Regions by Tons (2023)")
                manifest[str(save_chart(fig, "07_fr_tons_top10.png", pdf))] = "Top 10 foreign regions by 2023 tons."

            # 8. Top commodities by tons 2023
            if "sctg2" in df_intl.columns:
                com_tons = df_intl.groupby("sctg2")["tons_2023"].sum().reset_index().sort_values("tons_2023", ascending=False).head(10)
                com_tons["label"] = com_tons["sctg2"].astype(int).astype(str).map(maps.get("sctg2", {})).fillna(com_tons["sctg2"].astype(str))
                fig, ax = plt.subplots(figsize=(7,5))
                sns.barplot(data=com_tons, y="label", x="tons_2023", ax=ax, color="#59a14f"); ax.set_xlabel("Tons 2023"); ax.set_ylabel(""); ax.set_title("Top Commodities by Tons (2023)")
                manifest[str(save_chart(fig, "08_commodities_top10.png", pdf))] = "Top 10 commodities by 2023 tons."

            # 9. Top origin states by tons 2023
            if "dms_origst" in df_intl.columns:
                os_tons = df_intl.groupby("dms_origst")["tons_2023"].sum().reset_index().sort_values("tons_2023", ascending=False).head(10)
                fmap = fips_state_map()
                os_tons["state"] = os_tons["dms_origst"].astype(str).map(fmap).fillna(os_tons["dms_origst"].astype(str))
                fig, ax = plt.subplots(figsize=(7,5))
                sns.barplot(data=os_tons, x="state", y="tons_2023", ax=ax, color="#e15759")
                ax.set_xlabel("Origin State"); ax.set_ylabel("Tons 2023"); ax.set_title("Top Origin States (Tons 2023)"); ax.tick_params(axis='x', rotation=45)
                manifest[str(save_chart(fig, "09_top_origin_states.png", pdf))] = "Top 10 origin states by 2023 tons."

            # 10. Top destination states by tons 2023
            if "dms_destst" in df_intl.columns:
                ds_tons = df_intl.groupby("dms_destst")["tons_2023"].sum().reset_index().sort_values("tons_2023", ascending=False).head(10)
                fmap = fips_state_map()
                ds_tons["state"] = ds_tons["dms_destst"].astype(str).map(fmap).fillna(ds_tons["dms_destst"].astype(str))
                fig, ax = plt.subplots(figsize=(7,5))
                sns.barplot(data=ds_tons, x="state", y="tons_2023", ax=ax, color="#f28e2b")
                ax.set_xlabel("Destination State"); ax.set_ylabel("Tons 2023"); ax.set_title("Top Destination States (Tons 2023)"); ax.tick_params(axis='x', rotation=45)
                manifest[str(save_chart(fig, "10_top_dest_states.png", pdf))] = "Top 10 destination states by 2023 tons."

        # 11. Volatility distribution
        if "tons_volatility" in df.columns:
            fig, ax = plt.subplots(figsize=(7,4))
            vals = df["tons_volatility"].replace([np.inf, -np.inf], np.nan).dropna()
            vals = winsorize(vals, 0.01, 0.99)
            sns.histplot(vals, bins=40, ax=ax, color="#b07aa1"); ax.set_xlabel("Tons Volatility (σ/μ, 2017-2023)"); ax.set_title("Volatility Distribution")
            manifest[str(save_chart(fig, "11_volatility_hist.png", pdf))] = "Distribution of corridor volatility."

        # 12. Growth distribution 2017-2023
        if "tons_growth_17_23" in df.columns:
            fig, ax = plt.subplots(figsize=(7,4))
            vals = df["tons_growth_17_23"].replace([np.inf, -np.inf], np.nan).dropna()
            vals = winsorize(vals, 0.01, 0.99)
            sns.histplot(vals, bins=40, ax=ax, color="#76b7b2"); ax.set_xlabel("Tons Growth 2017-2023"); ax.set_title("Growth Distribution (Clipped 1%-99%)")
            manifest[str(save_chart(fig, "12_growth_hist.png", pdf))] = "Distribution of 2017-2023 growth."

        # 13. Value density distribution
        if "value_density_2023" in df.columns:
            fig, ax = plt.subplots(figsize=(7,4))
            vals = df["value_density_2023"].replace([np.inf, -np.inf], np.nan).dropna()
            vals = winsorize(vals, 0.01, 0.99)
            sns.histplot(vals, bins=40, ax=ax, color="#59a14f"); ax.set_xlabel("Value Density (2023)"); ax.set_title("Value Density Distribution")
            manifest[str(save_chart(fig, "13_value_density_hist.png", pdf))] = "Distribution of value per ton in 2023."

        # 14. Volatility vs value density scatter
        if {"tons_volatility", "value_density_2023"}.issubset(df.columns):
            fig, ax = plt.subplots(figsize=(7,5))
            sample = df[["tons_volatility", "value_density_2023"]].replace([np.inf, -np.inf], np.nan).dropna()
            # Winsorize both axes to reduce outliers
            sample["tons_volatility"] = winsorize(sample["tons_volatility"], 0.01, 0.99)
            sample["value_density_2023"] = winsorize(sample["value_density_2023"], 0.01, 0.99)
            sample = sample.sample(min(80000, len(sample)), random_state=42)
            ax.scatter(sample["tons_volatility"], sample["value_density_2023"], s=4, alpha=0.25, color="#4e79a7")
            ax.set_xlabel("Tons Volatility"); ax.set_ylabel("Value Density (2023)"); ax.set_title("Volatility vs Value Density")
            manifest[str(save_chart(fig, "14_vol_vs_valden.png", pdf))] = "Scatter of volatility vs value density."

        # 15. HHI concentration distribution
        if "hhi" in df.columns:
            fig, ax = plt.subplots(figsize=(7,4))
            vals = df["hhi"].replace([np.inf, -np.inf], np.nan).dropna()
            vals = winsorize(vals, 0.01, 0.99)
            sns.histplot(vals, bins=40, ax=ax, color="#e15759"); ax.set_xlabel("HHI by Foreign Region"); ax.set_title("Concentration (HHI) Distribution")
            manifest[str(save_chart(fig, "15_hhi_hist.png", pdf))] = "Concentration across destination states per foreign region."

        # 16. Correlation heatmap (sample)
        numeric_cols = [c for c in ["tons_2017", "tons_2023", "value_2017", "value_2023", "tmiles_2017", "tmiles_2023", "tons_volatility", "tons_growth_17_23", "value_density_2023", "hhi"] if c in df.columns]
        if len(numeric_cols) >= 4:
            sample = df[numeric_cols].replace([np.inf, -np.inf], np.nan).dropna().sample(min(20000, df.shape[0]), random_state=42)
            corr = sample.corr(numeric_only=True)
            fig, ax = plt.subplots(figsize=(7,6))
            sns.heatmap(corr, annot=False, cmap="coolwarm", center=0, ax=ax)
            ax.set_title("Correlation Heatmap (Sample)")
            manifest[str(save_chart(fig, "16_corr_heatmap.png", pdf))] = "Correlation among key metrics."

    return manifest, pdf_path


def write_powerpoint_script(df: pd.DataFrame, df_intl: pd.DataFrame, _manifest: Dict[str, str]) -> None:
    total_rows, total_cols = df.shape
    intl_rows = len(df_intl)
    bullets: List[str] = []
    bullets.append(f"Dataset overview: {total_rows:,} rows × {total_cols} vars. International subset: {intl_rows:,} rows.")
    bullets.append("")
    content = []
    content.append("# PowerPoint Content Script (Text Only)\n")
    content.append("\n")
    content.append("Slide 1 — Title\n")
    content.append("- Title: International Supply Chain Volatility\n")
    content.append("- Subtitle: FAF5.7 — Key trends, exposures, and actions\n")
    content.append("- Notes: Set context and objectives; keep slide minimal.\n\n")

    content.append("Slide 2 — Trends\n")
    content.append("- Visuals: 01_tons_trend.png (left), 02_value_trend.png (right)\n")
    content.append("- Text (minimal): 'International volumes and values over time'\n")
    content.append("- Notes: Call out inflection years; align value scale (thousands USD).\n\n")

    content.append("Slide 3 — Network Exposure\n")
    content.append("- Visuals: 04_mode_tons_2023.png (left), 05_mode_value_2023.png (right)\n")
    content.append("- Text (minimal): 'Mode mix by tons and value (2023)'\n")
    content.append("- Notes: Discuss reliance on top modes and sensitivity to disruptions.\n\n")

    content.append("Slide 4 — Geography & Commodities\n")
    content.append("- Visuals: 07_fr_tons_top10.png (left), 08_commodities_top10.png (right)\n")
    content.append("- Text (minimal): 'Where volume concentrates'\n")
    content.append("- Notes: Highlight top foreign regions and commodity exposures.\n\n")

    content.append("Slide 5 — Origins & Destinations\n")
    content.append("- Visuals: 09_top_origin_states.png (left), 10_top_dest_states.png (right)\n")
    content.append("- Text (minimal): 'Top US endpoints'\n")
    content.append("- Notes: Link to corridor planning implications.\n\n")

    content.append("Slide 6 — Risk Profile\n")
    content.append("- Visuals: 11_volatility_hist.png (left), 12_growth_hist.png (right)\n")
    content.append("- Text (minimal): 'Volatility and growth distribution'\n")
    content.append("- Notes: Emphasize tails; identify where mitigation is needed.\n\n")

    content.append("Slide 7 — Fragility Map\n")
    content.append("- Visuals: 14_vol_vs_valden.png (single, centered)\n")
    content.append("- Text (minimal): 'Fragile high-value flows'\n")
    content.append("- Notes: Upper-right quadrant indicates high-value, high-volatility risk.\n\n")

    content.append("Slide 8 — Concentration & Correlations\n")
    content.append("- Visuals: 15_hhi_hist.png (left), 16_corr_heatmap.png (right)\n")
    content.append("- Text (minimal): 'Concentration and relationships'\n")
    content.append("- Notes: Tie HHI to resilience strategy; interpret key correlations.\n\n")

    content.append("Slide 9 — Actions\n")
    content.append("- Text (minimal): 'Target high-volatility lanes; diversify; enhance mode optionality'\n")
    content.append("- Notes: Summarize 3-5 decisions and owners.\n\n")

    PPT_SCRIPT.write_text("".join(content), encoding="utf-8")


def write_fields_doc(dict_map: Dict[str, Dict[str, str]]) -> None:
    used_fields: List[str] = [
        "trade_type", "fr_orig", "fr_dest", "dms_mode", "sctg2", "dms_origst", "dms_destst",
        "tons_2017", "tons_2023", "value_2017", "value_2023", "tmiles_2017", "tmiles_2023",
        "tons_volatility", "tons_growth_17_23", "value_density_2023", "hhi",
    ]
    lines: List[str] = []
    lines.append("# Fields Used (with Data Dictionary References)\n\n")
    lines.append("International subset filtered to FR regions 801–808 and trade types Import/Export when available.\n\n")
    lines.append("| Field | Category | Description |\n")
    lines.append("|---|---|---|\n")
    for f in used_fields:
        meta = dict_map.get(f, {"category": "", "description": ""})
        lines.append(f"| {f} | {meta.get('category','')} | {meta.get('description','')} |\n")
    FIELDS_DOC.write_text("".join(lines), encoding="utf-8")


def main() -> int:
    ensure_outputs_dir()
    df, df_intl = load_data()
    maps = load_codebooks()
    df_features = compute_derived(df_intl)
    manifest, pdf_path = build_dashboard(df_features, df_intl, maps)
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_powerpoint_script(df, df_intl, manifest)
    write_fields_doc(read_data_dictionary())
    print(f"✅ Dashboard charts saved to: {OUT_DIR}")
    print(f"📄 Manifest: {MANIFEST}")
    print(f"📝 PowerPoint content script: {PPT_SCRIPT}")
    print(f"📘 Multi-page PDF dashboard: {pdf_path}")
    print(f"📑 Fields doc: {FIELDS_DOC}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


