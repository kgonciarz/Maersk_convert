import pandas as pd
import streamlit as st
from io import BytesIO
from datetime import date, datetime
import re
st.set_page_config(page_title="Maersk → AAA COCOA Formatter", layout="wide")

# Optional FX
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


def norm_cur(x) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip().upper()
    s = s.replace(".", "")
    mapping = {
        "US$": "USD",
        "USDOLLAR": "USD",
        "EURO": "EUR",
        "€": "EUR",
    }
    return mapping.get(s, s)

def city_key(x: str) -> str:
    """Normalize city name for matching (Batam Island -> batam)."""
    if not isinstance(x, str) or not x.strip():
        return ""
    first = x.split(",")[0].strip().lower()

    # keep only letters/spaces
    first = re.sub(r"[^a-z\s]", " ", first)
    first = re.sub(r"\s+", " ", first).strip()

    # handle common variants
    if first.startswith("batam"):
        return "batam"
    if first.startswith("amsterdam"):
        return "amsterdam"

    return first  # fallback

def is_ams_or_batam(to_value: str) -> bool:
    return city_key(to_value) in {"amsterdam", "batam"}


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


def safe_upper(x) -> str:
    return "" if pd.isna(x) else str(x).strip().upper()


def parse_any_date(x):
    if pd.isna(x):
        return None
    if isinstance(x, (datetime, pd.Timestamp)):
        return x.date()
    try:
        return pd.to_datetime(x).date()
    except Exception:
        return None


# ---------- FX (yfinance + manual fallback) ----------
@st.cache_data(show_spinner=False)
def _yf_close_on_or_after(ticker: str, start: date, end: date):
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if df is None or df.empty:
        return None
    if "Close" in df.columns:
        s = df["Close"].dropna()
        return float(s.iloc[0]) if len(s) else None
    try:
        return float(df.dropna().iloc[0])
    except Exception:
        return None


def fx_ticker(from_cur: str, to_cur: str) -> str:
    return f"{from_cur}{to_cur}=X"  # 1 FROM in TO


def get_fx_rate_yfinance(from_cur: str, to_cur: str, fx_date: date) -> float:
    from_cur = safe_upper(from_cur)
    to_cur = safe_upper(to_cur)
    if from_cur == to_cur:
        return 1.0
    if not YF_AVAILABLE:
        raise RuntimeError("yfinance not available")

    start = pd.Timestamp(fx_date).date()
    end = (pd.Timestamp(fx_date) + pd.Timedelta(days=7)).date()

    # 1) direct: 1 FROM in TO
    direct_ticker = f"{from_cur}{to_cur}=X"
    direct = _yf_close_on_or_after(direct_ticker, start, end)
    if direct is not None and direct > 0:
        return float(direct)

    # 2) inverse available? use 1 / (1 TO in FROM)
    inv_ticker = f"{to_cur}{from_cur}=X"
    inv = _yf_close_on_or_after(inv_ticker, start, end)
    if inv is not None and inv > 0:
        return 1.0 / float(inv)

    # 3) USD pivot (with inverse safety)
    if from_cur != "USD" and to_cur != "USD":
        r1 = _yf_close_on_or_after(f"{from_cur}USD=X", start, end)
        if r1 is None:
            inv1 = _yf_close_on_or_after(f"USD{from_cur}=X", start, end)
            r1 = (1.0 / inv1) if inv1 else None

        r2 = _yf_close_on_or_after(f"USD{to_cur}=X", start, end)
        if r2 is None:
            inv2 = _yf_close_on_or_after(f"{to_cur}USD=X", start, end)
            r2 = (1.0 / inv2) if inv2 else None

        if r1 is not None and r2 is not None:
            return float(r1) * float(r2)

    raise RuntimeError(f"Could not fetch FX rate for {from_cur}->{to_cur}")


def get_fx_rate(from_cur: str, to_cur: str, fx_date: date, manual_rates: dict[tuple[str, str], float]) -> float:
    from_cur = safe_upper(from_cur)
    to_cur = safe_upper(to_cur)
    if from_cur == to_cur:
        return 1.0

    if (from_cur, to_cur) in manual_rates:
        return float(manual_rates[(from_cur, to_cur)])

    # inverse manual supported
    if (to_cur, from_cur) in manual_rates and manual_rates[(to_cur, from_cur)] != 0:
        return 1.0 / float(manual_rates[(to_cur, from_cur)])

    return get_fx_rate_yfinance(from_cur, to_cur, fx_date)


# ---------- Core ----------
SURCHARGE_CODES = {"CFD", "CFO", "DTI", "EBS", "EMS", "FFF", "PSS"}  # from your screenshot


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

    # Build long df by container (so ALL codes come from 20/40 columns correctly)
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
    long_df["Charge Code"] = long_df["Charge Code"].astype(str).str.upper().str.strip()
    long_df["Charge Currency"] = long_df["Charge Currency"].map(norm_cur)
    long_df["Charge Type"] = long_df["Charge Type"].astype(str).str.upper().str.strip()

    idx = ["Load Port", "To (City, Country/Region)", "Transit Time", "CONTAINER"]

    # Lane currency = BAS currency (required for your rule "final currency must be same as BAS")
    def pick_bas_currency(group: pd.DataFrame):
        bas_cur = group.loc[group["Charge Code"].eq("BAS"), "Charge Currency"]
        bas_cur = bas_cur[bas_cur.ne("")]
        if len(bas_cur):
            return bas_cur.iloc[0]
        # If BAS missing, we still pick first non-empty to avoid crash, but the lane will be incomplete
        first = group["Charge Currency"]
        first = first[first.ne("")]
        return first.iloc[0] if len(first) else ""

    lane_currency = long_df.groupby(idx).apply(pick_bas_currency).reset_index(name="Currency")
    long_df = long_df.merge(lane_currency, on=idx, how="left")

    # Convert each charge to lane (BAS) currency
    def convert_to_lane(row):
        amt = row["Charge Rate"]
        from_cur = row["Charge Currency"]
        to_cur = row["Currency"]
        if pd.isna(amt) or not from_cur or not to_cur:
            return pd.NA
        try:
            r = get_fx_rate(from_cur, to_cur, fx_date, manual_rates)
            return float(amt) * float(r)
        except Exception:
            return pd.NA  # do NOT silently mix/guess

    long_df["Rate_in_BAS_CCY"] = long_df.apply(convert_to_lane, axis=1)

    # Build lane list
    lanes = long_df[idx].drop_duplicates().copy()
    out = lanes.merge(lane_currency, on=idx, how="left")

    # BAS in lane currency
    bas = (
        long_df[long_df["Charge Code"].eq("BAS")]
        .groupby(idx, as_index=False)["Rate_in_BAS_CCY"]
        .first()
        .rename(columns={"Rate_in_BAS_CCY": "BAS"})
    )
    out = out.merge(bas, on=idx, how="left")

    # Inland haulage import (IHI) in lane currency
    ihi = (
        long_df[long_df["Charge Code"].eq(safe_upper(inland_import_code))]
        .groupby(idx, as_index=False)["Rate_in_BAS_CCY"]
        .first()
        .rename(columns={"Rate_in_BAS_CCY": "IHI"})
    )
    out = out.merge(ihi, on=idx, how="left")

    # FREIGHT = BAS (+ IHI for AMS/Batam)
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

    # Additional charge = DTI (converted into BAS currency)
    dti = (
        long_df[long_df["Charge Code"].eq("DTI")]
        .groupby(idx, as_index=False)["Rate_in_BAS_CCY"]
        .first()
        .rename(columns={"Rate_in_BAS_CCY": "DTI"})
    )
    out = out.merge(dti, on=idx, how="left")

    # SURCHARGE = sum of the specific freight codes in SURCHARGE_CODES (converted), excluding BAS
    # IMPORTANT: If AMS/Batam and IHI is added into FREIGHT, ensure IHI is not part of surcharge even if code list changes
    surcharge_base = long_df[long_df["Charge Code"].isin(SURCHARGE_CODES)].copy()

    # (Optional safety: If someone later adds BAS into the list, remove it anyway)
    surcharge_base = surcharge_base[~surcharge_base["Charge Code"].eq("BAS")]

    surcharge_sum = (
        surcharge_base.groupby(idx, as_index=False)["Rate_in_BAS_CCY"]
        .sum(min_count=1)
        .rename(columns={"Rate_in_BAS_CCY": "Surcharge"})
    )
    out = out.merge(surcharge_sum, on=idx, how="left")
    out["Surcharge"] = out["Surcharge"].fillna(0)

    out["ALL_IN"] = out["FREIGHT"] + out["Surcharge"]

    # Audits for each code (in BAS currency) so you can verify the surcharge breakdown
    audit_wide = (
        surcharge_base.pivot_table(
            index=idx,
            columns="Charge Code",
            values="Rate_in_BAS_CCY",
            aggfunc="first",
        )
        .reset_index()
    )

    # ✅ Rename pivoted code columns to avoid collisions with real columns like "DTI"
    rename_map = {c: f"{c}_audit" for c in audit_wide.columns if c not in idx}
    audit_wide = audit_wide.rename(columns=rename_map)

    out = out.merge(audit_wide, on=idx, how="left")


    # IDs / country
    out["POL_city"] = out["Load Port"].map(city_only)
    out["POD_city"] = out["To (City, Country/Region)"].map(city_only)
    out["ID"] = out["POL_city"] + out["POD_city"] + out["CONTAINER"].astype(str)
    out["country"] = out["Load Port"].map(country_from_pol)

    # Valid date (optional)
    valid_col = pick_first_existing(df, ["Valid To", "Valid Until", "Expiry Date", "Expiration Date", "Quote Expiry"])
    valid_series = df[valid_col] if valid_col else ""

    final_cols = {
        "Shipping Line": "MAERSK",
        "ID": out["ID"],
        "country": out["country"],
        "POL": out["Load Port"],
        "POD": out["To (City, Country/Region)"],
        "CONTAINER": out["CONTAINER"],
        "FREIGHT": out["FREIGHT"],
        "Currency": out["Currency"],  # ALWAYS BAS currency
        "Surcharge": out["Surcharge"],
        "ALL_IN": out["ALL_IN"],
        "TT": out["Transit Time"],
        "Valid": valid_series if isinstance(valid_series, pd.Series) else "",
        "Additional charge (per container) included in all-in": out["DTI"],  # DTI per your list
        # audits:
        "BAS (audit)": out["BAS"],
        "IHI (audit)": out["IHI"],
    }

    # Add audit columns for each surcharge code if present
    for code in sorted(SURCHARGE_CODES):
        col = f"{code}_audit"
        if col in out.columns:
            final_cols[f"{code} (audit)"] = out[col]


    final = pd.DataFrame(final_cols).sort_values(["POL", "POD", "CONTAINER"], ignore_index=True)
    return final


# ---------- UI ----------
st.title("Maersk Quote → AAA Freight Format (COCOA)")

st.write(
    """
**Rules implemented**
- Pulls 20/40 rates from Maersk (two-row header)
- **Final currency = BAS currency**
- If a surcharge is in USD/EUR/etc, it is **converted into BAS currency** before summing
- **Surcharge = CFD + CFO + DTI + EBS + EMS + FFF + PSS** (all converted), **BAS excluded**
- **FREIGHT = BAS** (+ **IHI** only if destination is Amsterdam or Batam; converted if needed)
- Additional charge (included in all-in) = **DTI**
"""
)

uploaded = st.file_uploader("Upload Maersk Excel (.xlsx)", type=["xlsx"])
sheet_name = st.text_input("Maersk sheet name", value="QuoteOutput")

st.subheader("FX conversion")
fx_basis = st.selectbox("FX date basis", ["Use today's date", "Use a specific date"], index=0)
fx_date = date.today() if fx_basis == "Use today's date" else st.date_input("FX date", value=date.today())

inland_code = st.text_input("Inland haulage import code (for Amsterdam/Batam add to freight)", value="IHI")

manual_fx_text = st.text_area(
    "Manual FX rates (optional). One per line: FROM,TO,RATE. Example: USD,EUR,0.92",
    value="",
    height=100,
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
        st.warning(f"Could not parse FX line: {line}")

if uploaded:
    try:
        if not YF_AVAILABLE and not manual_rates:
            st.warning("yfinance not available and no manual FX provided. Install yfinance or provide manual FX rates.")
        final_df = process_maersk(
            uploaded.getvalue(),
            sheet_name=sheet_name,
            fx_date=fx_date,
            manual_rates=manual_rates,
            inland_import_code=inland_code,
        )

        if final_df.empty:
            st.warning("No rows generated. Check your sheet name and that it contains 20DRY/40DRY rates.")
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
