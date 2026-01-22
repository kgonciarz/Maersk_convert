import pandas as pd
import streamlit as st
from io import BytesIO

st.set_page_config(page_title="Maersk → AAA Formatter", layout="wide")


# ---------- Helpers ----------
def detect_header_row(file_bytes: bytes, sheet_name: str, max_scan_rows: int = 80) -> int:
    preview = pd.read_excel(BytesIO(file_bytes), sheet_name=sheet_name, header=None, nrows=max_scan_rows)
    required = {"Load Port", "To (City, Country/Region)", "Charge Code"}

    for r in range(len(preview)):
        values = set(str(x).strip() for x in preview.iloc[r].tolist() if pd.notna(x))
        if required.issubset(values):
            return r

    raise ValueError("Could not detect header row.")


def flatten_two_row_header(cols):
    flat = []
    for top, bottom in cols:
        top = "" if pd.isna(top) else str(top)
        bottom = "" if pd.isna(bottom) else str(bottom)
        if top.startswith("Unnamed") or top == "":
            flat.append(bottom)
        else:
            flat.append(f"{top}_{bottom}")
    return flat


def to_number(x):
    try:
        return float(str(x).replace(",", ""))
    except:
        return pd.NA


def city_only(x):
    return x.split(",")[0].strip().replace(" ", "") if isinstance(x, str) else ""


def is_ams_or_batam(x):
    return city_only(x).lower() in {"amsterdam", "batam"}


def process_maersk(file_bytes, sheet_name="QuoteOutput"):
    header_row = detect_header_row(file_bytes, sheet_name)
    df = pd.read_excel(BytesIO(file_bytes), sheet_name=sheet_name, header=[header_row - 1, header_row])
    df.columns = flatten_two_row_header(df.columns)

    container_sources = {
        20: [("20DRY_Rate", "20DRY_Currency")],
        40: [("40DRY_Rate", "40DRY_Currency"), ("40HDRY_Rate", "40HDRY_Currency")],
    }

    base_cols = ["Load Port", "To (City, Country/Region)", "Transit Time", "Charge Code", "Rate Basis"]

    long = []
    for cont, sources in container_sources.items():
        rate, cur = None, None
        for r, c in sources:
            if r in df.columns:
                rate = df[r].map(to_number)
                cur = df[c]
                break
        if rate is None:
            continue

        part = df[base_cols].copy()
        part["CONTAINER"] = cont
        part["Rate"] = rate
        part["Currency"] = cur
        part = part[part["Rate"].notna()]
        long.append(part)

    df = pd.concat(long, ignore_index=True)
    df = df[df["Rate Basis"].astype(str).str.upper() == "PER_CONTAINER"]

    idx = ["Load Port", "To (City, Country/Region)", "Transit Time", "CONTAINER"]

    bas = df[df["Charge Code"] == "BAS"].groupby(idx)["Rate"].first()
    ihi = df[df["Charge Code"] == "IHI"].groupby(idx)["Rate"].first()

    surcharge = (
        df[~df["Charge Code"].isin(["BAS", "IHI"])]
        .groupby(idx)["Rate"]
        .sum()
    )

    out = (
        bas.rename("BAS")
        .to_frame()
        .join(ihi.rename("IHI"))
        .join(surcharge.rename("SURCHARGE"))
        .reset_index()
    )

    out["IHI"] = out["IHI"].fillna(0)
    out["SURCHARGE"] = out["SURCHARGE"].fillna(0)

    out["FREIGHT"] = out["BAS"]
    mask = out["To (City, Country/Region)"].apply(is_ams_or_batam)
    out.loc[mask, "FREIGHT"] += out.loc[mask, "IHI"]

    out["ALL_IN"] = out["FREIGHT"] + out["SURCHARGE"]

    out["ID"] = (
        out["Load Port"].map(city_only)
        + out["To (City, Country/Region)"].map(city_only)
        + out["CONTAINER"].astype(str)
    )

    final = pd.DataFrame({
        "Shipping Line": "MAERSK",
        "ID": out["ID"],
        "POL": out["Load Port"],
        "POD": out["To (City, Country/Region)"],
        "CONTAINER": out["CONTAINER"],
        "FREIGHT": out["FREIGHT"],
        "Surcharge": out["SURCHARGE"],
        "ALL_IN": out["ALL_IN"],
        "TT": out["Transit Time"],
    })

    return final.sort_values(["POL", "POD", "CONTAINER"])


# ---------- UI ----------
st.title("Maersk → AAA Freight Converter")

uploaded = st.file_uploader("Upload Maersk Excel (.xlsx)", type="xlsx")
sheet = st.text_input("Sheet name", "QuoteOutput")

if uploaded:
    df = process_maersk(uploaded.getvalue(), sheet)
    st.dataframe(df, use_container_width=True)

    buf = BytesIO()
    df.to_excel(buf, index=False)
    st.download_button("Download AAA_Output.xlsx", buf.getvalue(), "AAA_Output.xlsx")
