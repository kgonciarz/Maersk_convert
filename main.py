import pandas as pd
import streamlit as st
from io import BytesIO

st.set_page_config(page_title="Maersk → AAA Formatter", layout="wide")


# ---------- Helpers ----------
def detect_header_row(file_bytes: bytes, sheet_name: str, max_scan_rows: int = 80) -> int:
    """
    Detect the main header row (the row that contains: Load Port, To..., Charge Code).
    We'll then read with a TWO-row header: [header_row-1, header_row]
    because Maersk puts container types (20DRY/40DRY/40HDRY) one row above.
    """
    preview = pd.read_excel(BytesIO(file_bytes), sheet_name=sheet_name, header=None, nrows=max_scan_rows)

    required = {"Load Port", "To (City, Country/Region)", "Charge Code"}
    for r in range(len(preview)):
        values = set(str(x).strip() for x in preview.iloc[r].tolist() if pd.notna(x))
        if required.issubset(values):
            return r

    # fallback if To column differs slightly
    for r in range(len(preview)):
        values = [str(x).strip() for x in preview.iloc[r].tolist() if pd.notna(x)]
        if "Load Port" in values and "Charge Code" in values:
            return r

    raise ValueError(f"Could not detect the header row in sheet '{sheet_name}'.")


def flatten_two_row_header(cols: pd.MultiIndex) -> list[str]:
    """
    Turn MultiIndex columns like:
      ('20DRY', 'Rate') -> '20DRY_Rate'
      ('Unnamed: 1_level_0', 'Load Port') -> 'Load Port'
    Also handles Maersk quirks where 'Rfq Ids'/'Comments' appear under a container group.
    """
    flat = []
    for top, bottom in cols:
        top = "" if pd.isna(top) else str(top)
        bottom = "" if pd.isna(bottom) else str(bottom)

        if top.startswith("Unnamed") or bottom in ("Rfq Ids", "Comments"):
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


def city_from_to_field(to_value: str) -> str:
    if not isinstance(to_value, str) or not to_value.strip():
        return ""
    return to_value.split(",")[0].strip()


def is_ams_or_batam(to_value: str) -> bool:
    city = city_from_to_field(to_value).lower()
    return city in {"amsterdam", "batam"}


def process_maersk(file_bytes: bytes, sheet_name: str = "QuoteOutput") -> pd.DataFrame:
    """
    Output columns for AAA:
      POL, POD, CONTAINER, FREIGHT, Currency, Surcharge, TT, Additional Charge

    Rules:
      - FREIGHT = sum of ALL Charge Type == 'FREIGHT' (per container) + IHI if AMS/Batam
      - Surcharge = FFF
      - Additional Charge = DTI
      - 20 from 20DRY
      - 40 from 40DRY (fallback 40HDRY)
    """

    header_row = detect_header_row(file_bytes, sheet_name=sheet_name)
    if header_row == 0:
        raise ValueError("Header row detected at 0 — file format unexpected (needs 2-row header).")

    # Read with 2-row header: container row + field row
    df = pd.read_excel(BytesIO(file_bytes), sheet_name=sheet_name, header=[header_row - 1, header_row])
    if not isinstance(df.columns, pd.MultiIndex):
        raise ValueError("Expected a 2-row header (MultiIndex columns) but did not get one.")

    df.columns = flatten_two_row_header(df.columns)

    required_cols = [
        "Load Port",
        "To (City, Country/Region)",
        "Transit Time",
        "Charge Type",
        "Charge Code",
        "Charge Name",
        "Rate Basis",
    ]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    # Container rate/currency sources (prefer DRY, fallback to HDRY for 40)
    container_sources = {
        20: [("20DRY_Rate", "20DRY_Currency")],
        40: [("40DRY_Rate", "40DRY_Currency"), ("40HDRY_Rate", "40HDRY_Currency")],
    }

    # Build long format rows: one row per (lane, charge, container)
    base_cols = [
        "Load Port",
        "To (City, Country/Region)",
        "Transit Time",
        "Charge Type",
        "Charge Code",
        "Charge Name",
        "Rate Basis",
    ]

    long_parts = []
    for cont, sources in container_sources.items():
        # pick FIRST non-null rate among sources (for 40: DRY then HDRY)
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

    # Keep only per-container charges (avoid per document etc.)
    long_df = long_df[long_df["Rate Basis"].astype(str).str.upper().eq("PER_CONTAINER")].copy()

    idx = ["Load Port", "To (City, Country/Region)", "Transit Time", "CONTAINER"]

    # Currency pick: prefer BAS currency, else first available
    def pick_currency(group: pd.DataFrame):
        bas_cur = group.loc[group["Charge Code"].eq("BAS"), "Charge Currency"]
        if len(bas_cur) and pd.notna(bas_cur.iloc[0]):
            return bas_cur.iloc[0]
        first = group["Charge Currency"].dropna()
        return first.iloc[0] if len(first) else pd.NA

    # Freight sum: ALL Charge Type == FREIGHT (this includes BAS + surcharges like FFF, etc.)
    freight_sum = (
        long_df[long_df["Charge Type"].astype(str).str.upper().eq("FREIGHT")]
        .groupby(idx, as_index=False)["Charge Rate"]
        .sum()
        .rename(columns={"Charge Rate": "FREIGHT_BASE"})
    )

    # Extract specific codes we need
    def code_value(code: str):
        return (
            long_df[long_df["Charge Code"].eq(code)]
            .groupby(idx, as_index=False)["Charge Rate"]
            .first()
            .rename(columns={"Charge Rate": code})
        )

    bas = code_value("BAS")
    fff = code_value("FFF")
    dti = code_value("DTI")
    ihi = code_value("IHI")  # inland haulage import (DESTINATION)

    # Build a lane/container base set from long_df (unique lanes + TT + container)
    lanes = long_df[idx].drop_duplicates().copy()

    # Merge everything
    out = lanes.merge(freight_sum, on=idx, how="left")
    out = out.merge(bas, on=idx, how="left")
    out = out.merge(fff, on=idx, how="left")
    out = out.merge(dti, on=idx, how="left")
    out = out.merge(ihi, on=idx, how="left")

    # Add currency
    currency = long_df.groupby(idx).apply(pick_currency).reset_index(name="Currency")
    out = out.merge(currency, on=idx, how="left")

    # Add inland haulage import ONLY for Amsterdam/Batam
    out["IHI"] = out["IHI"].fillna(0)

    def calc_freight(row):
        base = row["FREIGHT_BASE"]
        if pd.isna(base):
            return pd.NA
        if is_ams_or_batam(row["To (City, Country/Region)"]):
            return base + (row["IHI"] if pd.notna(row["IHI"]) else 0)
        return base

    out["FREIGHT"] = out.apply(calc_freight, axis=1)

    final = pd.DataFrame({
        "POL": out["Load Port"],
        "POD": out["To (City, Country/Region)"],
        "CONTAINER": out["CONTAINER"],
        "FREIGHT": out["FREIGHT"],
        "Currency": out["Currency"],
        "Surcharge": out["FFF"],                 # Fossil Fuel Fee
        "TT": out["Transit Time"],
        "Additional Charge": out["DTI"],         # Free Time Extension (Contracts)
    }).sort_values(["POL", "POD", "CONTAINER"], ignore_index=True)

    return final


# ---------- UI ----------
st.title("Maersk Quote → AAA Freight Format")

st.write(
    """
Upload the Maersk Excel quote (QuoteOutput).
The app will output AAA-ready rows for 20 and 40.
"""
)

uploaded = st.file_uploader("Upload Maersk Excel (.xlsx)", type=["xlsx"])
sheet_name = st.text_input("Maersk sheet name", value="QuoteOutput")

if uploaded:
    try:
        final_df = process_maersk(uploaded.getvalue(), sheet_name=sheet_name)

        if final_df.empty:
            st.warning("No rows generated. Check that the file contains rates for 20DRY/40DRY and per-container charges.")
        else:
            st.subheader("AAA-ready output")
            st.dataframe(final_df, use_container_width=True)

            buffer = BytesIO()
            with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
                final_df.to_excel(writer, index=False, sheet_name="AAA_Output")

            st.download_button(
                "Download AAA_Output.xlsx",
                data=buffer.getvalue(),
                file_name="AAA_Output.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

            st.info("You can also copy-paste directly from the table into your AAA file.")
    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.caption("Upload a Maersk file to get started.")
