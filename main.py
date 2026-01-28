import pandas as pd
import streamlit as st
from io import BytesIO

st.set_page_config(page_title="Maersk → AAA COCOA (Copy/Paste)", layout="wide")


# ---------- Helpers ----------
def detect_header_row(file_bytes: bytes, sheet_name: str, max_scan_rows: int = 80) -> int:
    preview = pd.read_excel(BytesIO(file_bytes), sheet_name=sheet_name, header=None, nrows=max_scan_rows)
    required = {"Load Port", "To (City, Country/Region)", "Charge Code"}
    for r in range(len(preview)):
        values = set(str(x).strip() for x in preview.iloc[r].tolist() if pd.notna(x))
        if required.issubset(values):
            return r
    raise ValueError(f"Could not detect the header row in sheet '{sheet_name}'.")


def flatten_two_row_header(cols: pd.MultiIndex) -> list[str]:
    flat = []
    for top, bottom in cols:
        top = "" if pd.isna(top) else str(top).strip()
        bottom = "" if pd.isna(bottom) else str(bottom).strip()
        if top.startswith("Unnamed") or top == "":
            flat.append(bottom)
        else:
            flat.append(f"{top}_{bottom}")
    return flat


def to_number(x):
    if pd.isna(x):
        return pd.NA
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip().replace(",", "")
    try:
        return float(s)
    except Exception:
        return pd.NA


def city_only(x: str) -> str:
    if not isinstance(x, str) or not x.strip():
        return ""
    return x.split(",")[0].strip().replace(" ", "")


def is_ams_or_batam(to_value: str) -> bool:
    return city_only(to_value).lower() in {"amsterdam", "batam"}


def pick_first_existing(df: pd.DataFrame, candidates: list[str]):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def norm_cur(x):
    if pd.isna(x):
        return ""
    return str(x).strip().upper()


# ---------- Core ----------
def process_maersk_cocoa(file_bytes: bytes, sheet_name: str = "QuoteOutput") -> pd.DataFrame:
    header_row = detect_header_row(file_bytes, sheet_name=sheet_name)
    if header_row == 0:
        raise ValueError("Header row detected at row 0; expected a 2-row header layout.")

    df = pd.read_excel(BytesIO(file_bytes), sheet_name=sheet_name, header=[header_row - 1, header_row])
    if not isinstance(df.columns, pd.MultiIndex):
        raise ValueError("Expected a 2-row header (MultiIndex columns), but did not get one.")
    df.columns = flatten_two_row_header(df.columns)

    required = [
        "Load Port",
        "To (City, Country/Region)",
        "Transit Time",
        "Charge Code",
        "Charge Name",
        "Rate Basis",
    ]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    # Container rate/currency columns
    container_sources = {
        20: [("20DRY_Rate", "20DRY_Currency")],
        40: [("40DRY_Rate", "40DRY_Currency"), ("40HDRY_Rate", "40HDRY_Currency")],
    }

    base_cols = [
        "Load Port",
        "To (City, Country/Region)",
        "Transit Time",
        "Charge Code",
        "Charge Name",
        "Rate Basis",
    ]

    # Build long table by container
    long_parts = []
    for cont, sources in container_sources.items():
        rate_series = None
        cur_series = None
        for rcol, ccol in sources:
            if rcol in df.columns and ccol in df.columns:
                r = df[rcol].map(to_number)
                c = df[ccol]
                if rate_series is None:
                    rate_series, cur_series = r, c
                else:
                    rate_series = rate_series.fillna(r)
                    cur_series = cur_series.fillna(c)

        if rate_series is None:
            continue

        part = df[base_cols].copy()
        part["CONTAINER"] = cont
        part["Charge Rate"] = rate_series
        part["Charge Currency"] = cur_series
        part = part[part["Charge Rate"].notna()].copy()
        long_parts.append(part)

    if not long_parts:
        return pd.DataFrame()

    long_df = pd.concat(long_parts, ignore_index=True)

    # keep only PER_CONTAINER
    long_df["Rate Basis"] = long_df["Rate Basis"].astype(str).str.upper().str.strip()
    long_df = long_df[long_df["Rate Basis"].eq("PER_CONTAINER")].copy()

    idx = ["Load Port", "To (City, Country/Region)", "Transit Time", "CONTAINER"]

    # Normalize currency strings
    long_df["Charge Currency"] = long_df["Charge Currency"].map(norm_cur)

    # Identify key charges
    # Codes (adjust if your Maersk file uses different codes)
    FREIGHT_CODE = "BAS"
    SURCHARGE_CODE = "FFF"
    INLAND_IMPORT_CODE = "IHI"  # inland haulage import (your earlier logic used IHI)
    # Free Time Extension: sometimes a code, sometimes only in Charge Name.
    # We'll detect by name containing "FREE TIME" and "EXT" (robust-ish).
    long_df["is_free_time_ext"] = long_df["Charge Name"].astype(str).str.upper().str.contains("FREE TIME") & \
                                 long_df["Charge Name"].astype(str).str.upper().str.contains("EXT")

    # Lane currency = BAS currency (preferred), else first non-empty currency
    def pick_lane_currency(group: pd.DataFrame):
        bas_cur = group.loc[group["Charge Code"].eq(FREIGHT_CODE), "Charge Currency"]
        bas_cur = bas_cur[bas_cur.ne("")]
        if len(bas_cur):
            return bas_cur.iloc[0]
        first = group["Charge Currency"]
        first = first[first.ne("")]
        return first.iloc[0] if len(first) else ""

    lane_currency = long_df.groupby(idx).apply(pick_lane_currency).reset_index(name="Currency")

    # Merge currency and keep only charges in same currency as lane currency
    long_df = long_df.merge(lane_currency, on=idx, how="left")
    same_cur = long_df["Charge Currency"].eq(long_df["Currency"])
    long_df_same = long_df[same_cur].copy()

    # Helper to get first value by code within same currency
    def first_by_code(code: str) -> pd.DataFrame:
        return (
            long_df_same[long_df_same["Charge Code"].eq(code)]
            .groupby(idx, as_index=False)["Charge Rate"]
            .first()
            .rename(columns={"Charge Rate": code})
        )

    bas = first_by_code(FREIGHT_CODE)
    fff = first_by_code(SURCHARGE_CODE)
    ihi = first_by_code(INLAND_IMPORT_CODE)

    # Free Time Extension amount (first match by name)
    fte = (
        long_df_same[long_df_same["is_free_time_ext"]]
        .groupby(idx, as_index=False)["Charge Rate"]
        .first()
        .rename(columns={"Charge Rate": "Free Time Extension"})
    )

    lanes = long_df_same[idx].drop_duplicates().copy()
    out = lanes.merge(lane_currency, on=idx, how="left")
    out = out.merge(bas, on=idx, how="left")
    out = out.merge(fff, on=idx, how="left")
    out = out.merge(ihi, on=idx, how="left")
    out = out.merge(fte, on=idx, how="left")

    # Freight rule: FREIGHT = BAS (+ IHI if Amsterdam/Batam)
    def calc_freight(row):
        if pd.isna(row.get(FREIGHT_CODE, pd.NA)):
            return pd.NA
        bas_val = row[FREIGHT_CODE]
        if is_ams_or_batam(row["To (City, Country/Region)"]):
            inland = row.get(INLAND_IMPORT_CODE, 0)
            inland = 0 if pd.isna(inland) else inland
            return bas_val + inland
        return bas_val

    out["FREIGHT"] = out.apply(calc_freight, axis=1)

    # Surcharge rule: Surcharge = FFF (not sum of everything)
    out["Surcharge"] = out.get(SURCHARGE_CODE, pd.NA)

    # Valid date (optional)
    valid_col = pick_first_existing(df, ["Valid To", "Valid Until", "Expiry Date", "Expiration Date", "Quote Expiry"])
    valid_series = df[valid_col] if valid_col else ""

    # Currency mismatch flag (useful for debugging why something is blank)
    # Example: FFF exists but in different currency than BAS -> it will be filtered out above
    # We'll detect that case from the unfiltered long_df.
    def has_code_in_other_currency(group: pd.DataFrame, code: str, lane_cur: str) -> bool:
        g = group[group["Charge Code"].eq(code)]
        if g.empty:
            return False
        return any(g["Charge Currency"].ne(lane_cur))

    # Build mismatch table for FFF and Free Time Ext
    mismatch_rows = []
    grouped_all = long_df.groupby(idx)
    cur_map = {(tuple(r[idx])): r["Currency"] for _, r in lane_currency.iterrows()}
    for key, g in grouped_all:
        lane_cur = cur_map.get(tuple(key), "")
        fff_mismatch = has_code_in_other_currency(g, SURCHARGE_CODE, lane_cur)
        ihi_mismatch = has_code_in_other_currency(g, INLAND_IMPORT_CODE, lane_cur)
        fte_g = g[g["is_free_time_ext"]]
        fte_mismatch = (not fte_g.empty) and any(fte_g["Charge Currency"].ne(lane_cur))
        mismatch_rows.append((*key, fff_mismatch, ihi_mismatch, fte_mismatch))

    mismatch_df = pd.DataFrame(
        mismatch_rows,
        columns=idx + ["FFF_other_currency", "IHI_other_currency", "FTE_other_currency"],
    )

    out = out.merge(mismatch_df, on=idx, how="left")

    final = pd.DataFrame({
        "POL": out["Load Port"],
        "POD": out["To (City, Country/Region)"],
        "CONTAINER": out["CONTAINER"],
        "FREIGHT": out["FREIGHT"],
        "Currency": out["Currency"],
        "Surcharge": out["Surcharge"],  # = FFF
        "TT": out["Transit Time"],
        "Additional charge (per container) included in all-in": out["Free Time Extension"],
        "Valid": valid_series if isinstance(valid_series, pd.Series) else "",
        # Debug flags so you see why values might be blank:
        "FFF_other_currency": out["FFF_other_currency"],
        "IHI_other_currency": out["IHI_other_currency"],
        "FTE_other_currency": out["FTE_other_currency"],
    }).sort_values(["POL", "POD", "CONTAINER"], ignore_index=True)

    return final


# ---------- UI ----------
st.title("Maersk → AAA COCOA (Copy/Paste output)")

st.write(
    """
**Rules**
- Output only: POL, POD, CONTAINER, FREIGHT, Currency, Surcharge(=FFF), TT, Additional(=Free Time Extension)
- **FREIGHT = BAS** + (**IHI** only if POD city is **Amsterdam** or **Batam**)
- **No mixing currencies:** lane currency is the BAS currency; other charges are taken only if they match that currency
"""
)

uploaded = st.file_uploader("Upload Maersk Excel (.xlsx)", type=["xlsx"])
sheet_name = st.text_input("Maersk sheet name", value="QuoteOutput")

if uploaded:
    try:
        final_df = process_maersk_cocoa(uploaded.getvalue(), sheet_name=sheet_name)

        if final_df.empty:
            st.warning("No rows generated. Check the file contains 20DRY/40DRY rates and PER_CONTAINER charges.")
        else:
            st.subheader("AAA-ready (copy/paste)")
            st.dataframe(final_df, use_container_width=True)

            # Copy/paste helper (TSV works great in Excel)
            tsv = final_df.to_csv(sep="\t", index=False)
            st.text_area("Copy/paste into AAA_freight COCOA_ACTUAL (TSV):", tsv, height=220)

            buffer = BytesIO()
            with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
                final_df.to_excel(writer, index=False, sheet_name="AAA_PASTE")

            st.download_button(
                "Download AAA_PASTE.xlsx",
                data=buffer.getvalue(),
                file_name="AAA_PASTE.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

            if final_df[["FFF_other_currency", "IHI_other_currency", "FTE_other_currency"]].any().any():
                st.warning(
                    "Some lanes have charges in a different currency than BAS. "
                    "Those charges were NOT added (to avoid EUR/USD mixing). "
                    "See the *_other_currency columns."
                )

    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.caption("Upload a Maersk file to get started.")
