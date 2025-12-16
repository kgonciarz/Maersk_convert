import pandas as pd
import streamlit as st
from io import BytesIO

st.set_page_config(page_title="Maersk → AAA Formatter", layout="wide")


# ---------- Helpers ----------
def detect_header_row(file_bytes: bytes, sheet_name: str, max_scan_rows: int = 60) -> int:
    """
    Detect the actual header row by scanning for required column names.
    """
    preview = pd.read_excel(BytesIO(file_bytes), sheet_name=sheet_name, header=None, nrows=max_scan_rows)
    required = {"Load Port", "To (City, Country/Region)", "Charge Code"}

    for r in range(len(preview)):
        values = set(str(x).strip() for x in preview.iloc[r].tolist() if pd.notna(x))
        if required.issubset(values):
            return r

    # Fallback if "To (City, Country/Region)" slightly differs
    for r in range(len(preview)):
        values = [str(x).strip() for x in preview.iloc[r].tolist() if pd.notna(x)]
        if "Load Port" in values and "Charge Code" in values:
            return r

    raise ValueError(
        f"Could not detect the header row in sheet '{sheet_name}'. "
        "The file layout may be different than expected."
    )


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
    Returns a DataFrame with AAA-ready columns:
    POL, POD, CONTAINER, FREIGHT, Currency, Surcharge, TT, Additional Charge
    """

    header_row = detect_header_row(file_bytes, sheet_name=sheet_name)
    df = pd.read_excel(BytesIO(file_bytes), sheet_name=sheet_name, header=header_row)

    base_cols = ["Load Port", "To (City, Country/Region)", "Transit Time", "Charge Code", "Charge Name"]
    for c in base_cols:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    rate_cols = [c for c in df.columns if str(c).startswith("Rate")]
    cur_cols = [c for c in df.columns if str(c).startswith("Currency")]

    if not rate_cols or not cur_cols:
        raise ValueError("Rate / Currency columns not found in the Maersk file.")

    def suffix(name: str) -> str:
        return str(name).replace("Rate", "").replace("Currency", "")

    rate_by = {suffix(c): c for c in rate_cols}
    cur_by = {suffix(c): c for c in cur_cols}
    suffixes = sorted(
        [s for s in rate_by.keys() if s in cur_by.keys()],
        key=lambda s: (len(s), s)
    )

    # First rate column = 20, remaining = 40
    container_map = {}
    for i, s in enumerate(suffixes[:3]):
        container_map[s] = 20 if i == 0 else 40

    wanted_codes = {"BAS", "FFF", "DTI", "IHI"}
    df = df[df["Charge Code"].isin(wanted_codes)].copy()

    long_parts = []
    for sfx, cont in container_map.items():
        rcol = rate_by[sfx]
        ccol = cur_by[sfx]

        part = df[base_cols + [rcol, ccol]].copy()
        part.rename(columns={rcol: "Charge Rate", ccol: "Charge Currency"}, inplace=True)
        part["CONTAINER"] = cont
        part["Charge Rate"] = part["Charge Rate"].map(to_number)
        part = part[part["Charge Rate"].notna()]
        long_parts.append(part)

    if not long_parts:
        return pd.DataFrame()

    long_df = pd.concat(long_parts, ignore_index=True)

    idx = ["Load Port", "To (City, Country/Region)", "Transit Time", "CONTAINER"]

    pivot_rate = (
        long_df.pivot_table(index=idx, columns="Charge Code", values="Charge Rate", aggfunc="first")
        .reset_index()
    )
    pivot_cur = (
        long_df.pivot_table(index=idx, columns="Charge Code", values="Charge Currency", aggfunc="first")
        .reset_index()
    )

    out = pivot_rate.merge(pivot_cur, on=idx, suffixes=("", "_cur"))

    def pick_currency(row):
        if pd.notna(row.get("BAS_cur", pd.NA)):
            return row["BAS_cur"]
        for c in ["FFF_cur", "DTI_cur", "IHI_cur"]:
            if pd.notna(row.get(c, pd.NA)):
                return row[c]
        return pd.NA

    out["Currency"] = out.apply(pick_currency, axis=1)

    if "IHI" not in out.columns:
        out["IHI"] = 0

    def calc_freight(row):
        bas = row.get("BAS", pd.NA)
        if pd.isna(bas):
            return pd.NA
        inland = row.get("IHI", 0)
        if pd.isna(inland):
            inland = 0
        return bas + inland if is_ams_or_batam(row["To (City, Country/Region)"]) else bas

    out["FREIGHT"] = out.apply(calc_freight, axis=1)

    final = pd.DataFrame({
        "POL": out["Load Port"],
        "POD": out["To (City, Country/Region)"],
        "CONTAINER": out["CONTAINER"],
        "FREIGHT": out["FREIGHT"],
        "Currency": out["Currency"],
        "Surcharge": out.get("FFF", pd.NA),
        "TT": out["Transit Time"],
        "Additional Charge": out.get("DTI", pd.NA),
    }).sort_values(["POL", "POD", "CONTAINER"], ignore_index=True)

    return final


# ---------- UI ----------
st.title("Maersk Quote → AAA Freight Format")

st.write(
    """
**How it works**
1. Upload the Maersk Excel file (QuoteOutput).
2. The app converts it to AAA format.
3. Copy & paste the table into AAA, or download the Excel output.
"""
)

uploaded = st.file_uploader("Upload Maersk Excel (.xlsx)", type=["xlsx"])
sheet_name = st.text_input("Maersk sheet name", value="QuoteOutput")

if uploaded:
    try:
        final_df = process_maersk(uploaded.getvalue(), sheet_name=sheet_name)

        if final_df.empty:
            st.warning(
                "No data was generated. This usually means the required charge codes "
                "(BAS, FFF, DTI, IHI) were not found or rate columns are empty."
            )
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

            st.info("You can also select the table and copy-paste it directly into your AAA file.")
    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.caption("Upload a Maersk file to get started.")
