import pandas as pd
import streamlit as st
from io import BytesIO
from datetime import datetime, date

st.set_page_config(page_title="Maersk → AAA COCOA Formatter", layout="wide")

# Optional: yfinance for FX conversion
try:
    import yfinance as yf
    YF_AVAILABLE = True
except Exception:
    YF_AVAILABLE = False


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


def country_from_pol(pol: str) -> str:
    if not isinstance(pol, str):
        return ""
    parts = [p.strip() for p in pol.split(",")]
    return parts[-1] if len(parts) > 1 else ""


def pick_first_existing(df: pd.DataFrame, candidates: list[str]):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def norm_cur(x) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip().upper()
    return "" if s in {"NA", "N/A", "NONE"} else s


def safe_upper(s):
    return "" if pd.isna(s) else str(s).strip().upper()


def parse_any_date(x):
    """Best-effort parse for Valid/Expiry columns; returns date or None."""
    if pd.isna(x):
        return None
    if isinstance(x, (datetime, pd.Timestamp)):
        return x.date()
    try:
        return pd.to_datetime(x).date()
    except Exception:
        return None


# ---------- FX ----------
@st.cache_data(show_spinner=False)
def _yf_close_on_or_after(ticker: str, start: date, end: date):
    # yfinance end is exclusive-ish; extend by 1 day
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if df is None or df.empty:
        return None
    # Use Close; first available row
    if "Close" in df.columns:
        return float(df["Close"].dropna().iloc[0])
    # Sometimes series differently shaped
    try:
        return float(df.dropna().iloc[0])
    except Exception:
        return None


def fx_ticker(from_cur: str, to_cur: str) -> str | None:
    from_cur = safe_upper(from_cur)
    to_cur = safe_upper(to_cur)
    if not from_cur or not to_cur or from_cur == to_cur:
        return None
    # Yahoo FX format: "EURUSD=X" means 1 EUR in USD
    return f"{from_cur}{to_cur}=X"


def get_fx_rate_yfinance(from_cur: str, to_cur: str, fx_date: date) -> float:
    """
    Returns multiplier such that: amount_in_to = amount_in_from * rate
    Example: from USD to EUR returns USDEUR rate.
    """
    from_cur = safe_upper(from_cur)
    to_cur = safe_upper(to_cur)
    if from_cur == to_cur:
        return 1.0

    if not YF_AVAILABLE:
        raise RuntimeError("yfinance not installed/available in this environment.")

    # Pull rate near fx_date (use a small window)
    start = fx_date
    end = fx_date.replace(day=fx_date.day)  # placeholder; we’ll just add days safely below
    # Use pandas to add day safely
    start_dt = pd.Timestamp(fx_date)
    end_dt = start_dt + pd.Timedelta(days=5)

    t = fx_ticker(from_cur, to_cur)
    r = _yf_close_on_or_after(t, start_dt.date(), end_dt.date())
    if r is not None:
        return float(r)

    # If direct pair not available, try via USD pivot:
    # from->USD and USD->to => from->to = (fromUSD) * (USDTo)
    if from_cur != "USD" and to_cur != "USD":
        r1 = _yf_close_on_or_after(fx_ticker(from_cur, "USD"), start_dt.date(), end_dt.date())
        r2 = _yf_close_on_or_after(fx_ticker("USD", to_cur), start_dt.date(), end_dt.date())
        if r1 is not None and r2 is not None:
            return float(r1) * float(r2)

    raise RuntimeError(f"Could not fetch FX rate for {from_cur}->{to_cur} on/near {fx_date}.")


def get_fx_rate(from_cur: str, to_cur: str, fx_date: date, manual_rates: dict[tuple[str, str], float]) -> float:
    from_cur = safe_upper(from_cur)
    to_cur = safe_upper(to_cur)
    if from_cur == to_cur:
        return 1.0
    if (from_cur, to_cur) in manual_rates:
        return float(manual_rates[(from_cur, to_cur)])

    # allow inverse manual
    if (to_cur, from_cur) in manual_rates and manual_rates[(to_cur, from_cur)] != 0:
        return 1.0 / float(manual_rates[(to_cur, from_cur)])

    # fall back to yfinance
    return get_fx_rate_yfinance(from_cur, to_cur, fx_date)


# ---------- Core ----------
def process_maersk(
    file_bytes: bytes,
    sheet_name: str,
    fx_date: date,
    manual_rates: dict[tuple[str, str], float],
    inland_import_code: str = "IHI",
) -> pd.DataFrame:
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
        "Charge Type",
    ]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

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
        "Charge Type",
        "Rate Basis",
    ]

    long_parts = []
    for cont, sources in container_sources.items():
        rate_series = None
        cur_series = None
        for rcol, ccol in sources:
            if rcol in df.columns and ccol in df.columns:
                r = df[rcol].map(to_number)
                c = df[ccol].map(norm_cur)
                if rate_series is None:
                    rate_series, cur_series = r, c
                else:
                    rate_series = rate_series.fillna(r)
                    cur_series = cur_series.where(cur_series.ne(""), "").fillna(c)

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

    # PER_CONTAINER only
    long_df["Rate Basis"] = long_df["Rate Basis"].astype(str).str.upper().str.strip()
    long_df = long_df[long_df["Rate Basis"].eq("PER_CONTAINER")].copy()

    # Normalize
    long_df["Charge Currency"] = long_df["Charge Currency"].map(norm_cur)
    long_df["Charge Type"] = long_df["Charge Type"].astype(str).str.upper().str.strip()

    idx = ["Load Port", "To (City, Country/Region)", "Transit Time", "CONTAINER"]

    # Lane currency = BAS currency (preferred)
    def pick_lane_currency(group: pd.DataFrame):
        bas_cur = group.loc[group["Charge Code"].eq("BAS"), "Charge Currency"]
        bas_cur = bas_cur[bas_cur.ne("")]
        if len(bas_cur):
            return bas_cur.iloc[0]
        first = group["Charge Currency"]
        first = first[first.ne("")]
        return first.iloc[0] if len(first) else ""

    lane_currency = long_df.groupby(idx).apply(pick_lane_currency).reset_index(name="Currency")

    # Merge lane currency onto ALL rows (we will CONVERT, not filter)
    long_df = long_df.merge(lane_currency, on=idx, how="left")

    # Convert every row to lane currency when needed
    def convert_row(row):
        amt = row["Charge Rate"]
        from_cur = row["Charge Currency"]
        to_cur = row["Currency"]
        if pd.isna(amt) or not from_cur or not to_cur:
            return pd.NA
        try:
            r = get_fx_rate(from_cur, to_cur, fx_date, manual_rates)
            return float(amt) * float(r)
        except Exception:
            # If FX cannot be fetched, leave NA so it won’t corrupt totals
            return pd.NA

    long_df["Charge Rate (lane cur)"] = long_df.apply(convert_row, axis=1)

    # Build lanes output
    lanes = long_df[idx].drop_duplicates().copy()
    out = lanes.merge(lane_currency, on=idx, how="left")

    # BAS (in lane currency)
    bas = (
        long_df[long_df["Charge Code"].eq("BAS")]
        .groupby(idx, as_index=False)["Charge Rate (lane cur)"]
        .first()
        .rename(columns={"Charge Rate (lane cur)": "BAS"})
    )
    out = out.merge(bas, on=idx, how="left")

    # IHI (in lane currency) for AMS/Batam freight add
    ihi = (
        long_df[long_df["Charge Code"].eq(inland_import_code)]
        .groupby(idx, as_index=False)["Charge Rate (lane cur)"]
        .first()
        .rename(columns={"Charge Rate (lane cur)": "IHI"})
    )
    out = out.merge(ihi, on=idx, how="left")

    # Free Time Extension (by name, lane currency)
    long_df["Charge Name U"] = long_df["Charge Name"].astype(str).str.upper()
    fte_df = long_df[
        long_df["Charge Name U"].str.contains("FREE TIME", na=False)
        & long_df["Charge Name U"].str.contains("EXT", na=False)
    ].copy()
    fte = (
        fte_df.groupby(idx, as_index=False)["Charge Rate (lane cur)"]
        .first()
        .rename(columns={"Charge Rate (lane cur)": "Free Time Extension"})
    )
    out = out.merge(fte, on=idx, how="left")

    # FREIGHT = BAS (+ IHI only for Amsterdam/Batam)
    def calc_freight(row):
        if pd.isna(row.get("BAS", pd.NA)):
            return pd.NA
        bas_val = row["BAS"]
        if is_ams_or_batam(row["To (City, Country/Region)"]):
            inland = row.get("IHI", 0)
            inland = 0 if pd.isna(inland) else inland
            return bas_val + inland
        return bas_val

    out["FREIGHT"] = out.apply(calc_freight, axis=1)

    # SURCHARGE = ALL Charge Type Freight EXCEPT BAS (converted to lane currency)
    is_freight_type = long_df["Charge Type"].str.contains("FREIGHT", na=False)

    surcharge_base = long_df[
        is_freight_type & (~long_df["Charge Code"].eq("BAS"))
    ].copy()

    # If AMS/Batam, exclude inland import from surcharge to avoid double counting (since added to freight)
    lane_special = out[idx].copy()
    lane_special["is_special"] = lane_special["To (City, Country/Region)"].apply(is_ams_or_batam)
    surcharge_base = surcharge_base.merge(lane_special, on=idx, how="left")
    surcharge_base = surcharge_base[
        ~((surcharge_base["is_special"] == True) & (surcharge_base["Charge Code"].eq(inland_import_code)))
    ]

    surcharge_sum = (
        surcharge_base.groupby(idx, as_index=False)["Charge Rate (lane cur)"]
        .sum(min_count=1)
        .rename(columns={"Charge Rate (lane cur)": "Surcharge"})
    )
    out = out.merge(surcharge_sum, on=idx, how="left")
    out["Surcharge"] = out["Surcharge"].fillna(0)

    # Optional: explicit FFF audit in lane currency
    fff = (
        long_df[long_df["Charge Code"].eq("FFF")]
        .groupby(idx, as_index=False)["Charge Rate (lane cur)"]
        .first()
        .rename(columns={"Charge Rate (lane cur)": "FFF (lane cur)"})
    )
    out = out.merge(fff, on=idx, how="left")

    out["ALL_IN"] = out["FREIGHT"] + out["Surcharge"]

    out["POL_city"] = out["Load Port"].map(city_only)
    out["POD_city"] = out["To (City, Country/Region)"].map(city_only)
    out["ID"] = out["POL_city"] + out["POD_city"] + out["CONTAINER"].astype(str)
    out["country"] = out["Load Port"].map(country_from_pol)

    # Valid date (optional)
    valid_col = pick_first_existing(df, ["Valid To", "Valid Until", "Expiry Date", "Expiration Date", "Quote Expiry"])
    valid_series = df[valid_col] if valid_col else ""

    final = pd.DataFrame({
        "Shipping Line": "MAERSK",
        "ID": out["ID"],
        "country": out["country"],
        "POL": out["Load Port"],
        "POD": out["To (City, Country/Region)"],
        "CONTAINER": out["CONTAINER"],
        "FREIGHT": out["FREIGHT"],
        "Currency": out["Currency"],  # ALWAYS BAS currency
        "Surcharge": out["Surcharge"],  # Freight-type charges except BAS, converted
        "ALL_IN": out["ALL_IN"],
        "TT": out["Transit Time"],
        "Valid": valid_series if isinstance(valid_series, pd.Series) else "",
        "Additional charge (per container) included in all-in": out["Free Time Extension"],
        # audits
        "BAS (lane cur)": out["BAS"],
        "IHI (lane cur)": out["IHI"],
        "FFF (lane cur)": out["FFF (lane cur)"],
    }).sort_values(["POL", "POD", "CONTAINER"], ignore_index=True)

    return final


# ---------- UI ----------
st.title("Maersk Quote → AAA Freight Format (COCOA)")

st.write(
    """
**Rules**
- Lane currency is **BAS currency**
- Any charge in another currency is **converted into BAS currency** (so no mixing)
- **FREIGHT = BAS + inland haulage import (IHI) if Amsterdam/Batam**
- **SURCHARGE = sum of ALL Charge Type = Freight EXCEPT BAS**, converted into BAS currency
- Free Time Extension → Additional charge (converted into BAS currency if needed)
"""
)

uploaded = st.file_uploader("Upload Maersk Excel (.xlsx)", type=["xlsx"])
sheet_name = st.text_input("Maersk sheet name", value="QuoteOutput")
inland_code = st.text_input("Inland haulage import charge code", value="IHI")

st.subheader("FX conversion settings")

fx_basis = st.selectbox(
    "Which FX date should be used for conversion?",
    ["Use today's date", "Use a specific date (manual)"],
    index=0,
)

if fx_basis == "Use today's date":
    fx_date = date.today()
else:
    fx_date = st.date_input("FX date", value=date.today())

st.caption("If your Streamlit environment has no internet, yfinance FX will fail. You can provide manual FX below.")

manual_fx_text = st.text_area(
    "Manual FX rates (optional). One per line: FROM,TO,RATE (amount_in_TO = amount_in_FROM * RATE). Example: USD,EUR,0.92",
    value="",
    height=120,
)

manual_rates = {}
for line in manual_fx_text.splitlines():
    line = line.strip()
    if not line:
        continue
    try:
        f, t, r = [x.strip().upper() for x in line.split(",")]
        manual_rates[(f, t)] = float(r)
    except Exception:
        st.warning(f"Could not parse manual FX line: {line}")

if uploaded:
    try:
        if not YF_AVAILABLE and not manual_rates:
            st.error("yfinance is not available AND no manual FX rates provided. Install yfinance or provide manual FX rates.")
        else:
            final_df = process_maersk(
                uploaded.getvalue(),
                sheet_name=sheet_name,
                fx_date=fx_date,
                manual_rates=manual_rates,
                inland_import_code=inland_code.strip().upper(),
            )

            if final_df.empty:
                st.warning("No rows generated. Check the file contains 20DRY/40DRY rates and PER_CONTAINER charges.")
            else:
                st.subheader("AAA-ready output")
                st.dataframe(final_df, use_container_width=True)

                st.text_area("Copy/paste into AAA (TSV):", final_df.to_csv(sep="\t", index=False), height=220)

                buffer = BytesIO()
                with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
                    final_df.to_excel(writer, index=False, sheet_name="AAA_Output")

                st.download_button(
                    "Download AAA_Output.xlsx",
                    data=buffer.getvalue(),
                    file_name="AAA_Output.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )

    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.caption("Upload a Maersk file to get started.")
