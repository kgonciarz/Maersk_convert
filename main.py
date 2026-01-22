import pandas as pd
import streamlit as st
from io import BytesIO

st.set_page_config(page_title="Maersk → AAA Formatter", layout="wide")


# ---------- Helpers ----------
def detect_header_row(file_bytes: bytes, sheet_name: str, max_scan_rows: int = 80) -> int:
    """
    Detect the row that contains the real field headers, e.g.
    Load Port | To (City, Country/Region) | Charge Code | ...
    Maersk often has a 2-row header where container types (20DRY/40DRY/40HDRY)
    are on the row ABOVE the field header row.
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
    MultiIndex columns like ('20DRY','Rate') become '20DRY_Rate'
    Columns like ('Unnamed: ...', 'Load Port') become 'Load Port'
    """
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
    """
    "Amsterdam, Netherlands" -> "Amsterdam"
    Removes spaces for ID-building: "New York" -> "NewYork"
    """
    if not isinstance(x, str) or not x.strip():
        return ""
    city = x.split(",")[0].strip()
    return city.replace(" ", "")


def is_ams_or_batam(to_value: str) -> bool:
    return city_only(to_value).lower() in {"amsterdam", "batam"}


def country_from_pol(pol: str) -> str:
    """
    Best-effort:
      "Lome, TOG" -> "TOG"
      "Lome" -> ""
    """
    if not isinstance(pol, str):
        return ""
    parts = [p.strip() for p in pol.split(",")]
    return parts[-1] if len(parts) > 1 else ""


def pick_first_existing(df: pd.DataFrame, candidates: list[str]):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def process_maersk(file_bytes: bytes, sheet_name: str = "QuoteOutput") -> pd.DataFrame:
    """
    Creates AAA-shaped output with these columns:
    Shipping Line, ID, country, POL, POD, CONTAINER, FREIGHT, LINER, Currency,
    Surcharge, ALL_IN, TT, Valid, LinerIncluded, ... , Additional charge (per container) included in all-in

    Rules:
      - FREIGHT = BAS (+ IHI only if destination is Amsterdam or Batam)
      - Surcharge = FFF
      - Additional charge included in all-in = DTI (Free Time Extension)
      - ALL_IN = sum of ALL PER_CONTAINER charges (+ IHI only for AMS/Batam),
                but to avoid double count we exclude IHI from the sum then add it conditionally.
    """

    header_row = detect_header_row(file_bytes, sheet_name=sheet_name)
    if header_row == 0:
        raise ValueError("Header row detected at row 0; expected a 2-row header layout.")

    # Read with 2-row header
    df = pd.read_excel(BytesIO(file_bytes), sheet_name=sheet_name, header=[header_row - 1, header_row])
    if not isinstance(df.columns, pd.MultiIndex):
        raise ValueError("Expected a 2-row header (MultiIndex columns), but did not get one.")

    df.columns = flatten_two_row_header(df.columns)

    # Required fields
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

    # Charge Type is sometimes available; if not, we can still compute ALL_IN by summing all PER_CONTAINER
    has_charge_type = "Charge Type" in df.columns

    # Rate/currency columns for containers
    # Prefer 40DRY, fallback 40HDRY for 40'
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
    if has_charge_type:
        base_cols.insert(3, "Charge Type")  # keep it with the charge fields

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

    # Only PER_CONTAINER
    long_df["Rate Basis"] = long_df["Rate Basis"].astype(str).str.upper().str.strip()
    long_df = long_df[long_df["Rate Basis"].eq("PER_CONTAINER")].copy()

    idx = ["Load Port", "To (City, Country/Region)", "Transit Time", "CONTAINER"]

    # Currency: prefer BAS currency else first available
    def pick_currency(group: pd.DataFrame):
        bas_cur = group.loc[group["Charge Code"].eq("BAS"), "Charge Currency"]
        if len(bas_cur) and pd.notna(bas_cur.iloc[0]):
            return bas_cur.iloc[0]
        first = group["Charge Currency"].dropna()
        return first.iloc[0] if len(first) else pd.NA

    currency = long_df.groupby(idx).apply(pick_currency).reset_index(name="Currency")

    # Extract specific codes (first value per lane/container)
    def code_value(code: str):
        return (
            long_df[long_df["Charge Code"].eq(code)]
            .groupby(idx, as_index=False)["Charge Rate"]
            .first()
            .rename(columns={"Charge Rate": code})
        )

    bas = code_value("BAS")   # base ocean freight
    fff = code_value("FFF")   # surcharge (fuel, etc.)
    dti = code_value("DTI")   # free time extension
    ihi = code_value("IHI")   # inland haulage import (to be added to freight for AMS/Batam)

    lanes = long_df[idx].drop_duplicates().copy()

    # ALL_IN base: sum of all per-container charges EXCEPT IHI (we'll add it conditionally)
    allin_sum = (
        long_df[~long_df["Charge Code"].eq("IHI")]
        .groupby(idx, as_index=False)["Charge Rate"]
        .sum()
        .rename(columns={"Charge Rate": "ALL_IN_BASE"})
    )

    out = lanes.merge(allin_sum, on=idx, how="left")
    out = out.merge(currency, on=idx, how="left")
    out = out.merge(bas, on=idx, how="left")
    out = out.merge(fff, on=idx, how="left")
    out = out.merge(dti, on=idx, how="left")
    out = out.merge(ihi, on=idx, how="left")

    out["IHI"] = out["IHI"].fillna(0)

    def calc_freight(row):
        bas_val = row.get("BAS", pd.NA)
        if pd.isna(bas_val):
            return pd.NA
        if is_ams_or_batam(row["To (City, Country/Region)"]):
            return bas_val + row["IHI"]
        return bas_val

    def calc_all_in(row):
        base = row.get("ALL_IN_BASE", pd.NA)
        if pd.isna(base):
            return pd.NA
        if is_ams_or_batam(row["To (City, Country/Region)"]):
            return base + row["IHI"]
        return base

    out["FREIGHT"] = out.apply(calc_freight, axis=1)
    out["ALL_IN"] = out.apply(calc_all_in, axis=1)

    # Shipping Line ID like "LomeAmsterdam40"
    out["POL_city"] = out["Load Port"].map(city_only)
    out["POD_city"] = out["To (City, Country/Region)"].map(city_only)
    out["ID"] = out["POL_city"] + out["POD_city"] + out["CONTAINER"].astype(str)

    # country (best effort)
    out["country"] = out["Load Port"].map(country_from_pol)

    # Valid date: try common Maersk columns if they exist; else blank
    valid_col = pick_first_existing(df, ["Valid To", "Valid Until", "Expiry Date", "Expiration Date", "Quote Expiry"])
    if valid_col:
        # If valid_col came from multiheader flattening, it might be like "Something_Valid To"
        # We try to pull it directly from df
        valid_series = df[valid_col]
    else:
        valid_series = ""

    # AAA-shaped output (keep headers exactly as you typed, including spaces if you need them)
    final = pd.DataFrame({
        "Shipping Line": "MAERSK",
        "ID": out["ID"],
        "country": out["country"],
        "POL": out["Load Port"],
        "POD": out["To (City, Country/Region)"],
        "CONTAINER": out["CONTAINER"],
        "FREIGHT": out["FREIGHT"],
        "LINER": "not included",
        "Currency": out["Currency"],
        "Surcharge": out.get("FFF", pd.NA),
        "ALL_IN": out["ALL_IN"],
        "TT": out["Transit Time"],
        "Valid": valid_series if isinstance(valid_series, pd.Series) else "",
        "LinerIncluded": 0,
        "Shipping Line S": "",
        "DEMDET": "",
        "Contrat Ref": "",
        "Detention": "",
        "Demurrage": "",
        "Additional detention": "",
        "Additional demurrage": "",
        "Additional charge (per container) included in all-in": out.get("DTI", pd.NA),
    }).sort_values(["POL", "POD", "CONTAINER"], ignore_index=True)

    return final


# ---------- UI ----------
st.title("Maersk Quote → AAA Freight Format (COCOA)")

st.write(
    """
Upload the Maersk Excel quote (usually sheet **QuoteOutput**).
The app outputs an AAA-ready table (with Shipping Line defaults + computed FREIGHT/ALL_IN).
"""
)

uploaded = st.file_uploader("Upload Maersk Excel (.xlsx)", type=["xlsx"])
sheet_name = st.text_input("Maersk sheet name", value="QuoteOutput")

if uploaded:
    try:
        final_df = process_maersk(uploaded.getvalue(), sheet_name=sheet_name)

        if final_df.empty:
            st.warning("No rows generated. Check the file contains 20DRY/40DRY rates and PER_CONTAINER charges.")
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
