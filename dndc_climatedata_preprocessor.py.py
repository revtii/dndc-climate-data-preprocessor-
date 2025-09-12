import streamlit as st
import pandas as pd
import numpy as np

## structure of this project: upload -> map columns -> validate structure -> convert units -> validate again -> download 

# title of the app
st.set_page_config(page_title="DNDC Climate Pre-processor", layout="wide")
st.title("Meterological Data Pre-processing for DNDC")

# Ordered DNDC formats and required fields
FORMATS = [
    "Jday, MeanT, Prec",
    "Jday, MaxT, MinT, Prec",
    "Jday, MaxT, MinT, Prec, Radiation",
    "Jday, MaxT, MinT, Prec, WindSpeed",
    "Jday, MaxT, MinT, Prec, WindSpeed, Radiation, Humidity",
    "Jday, MaxT, MinT, Prec, WindSpeed, Humidity",
    "Jday, MaxT, MinT, Prec, Humidity",
    "Prec (cm), Radiation (MJ/m2/day), WindSpeed (m/s), Humidity (%)",
]
FORMATS_TO_FIELDS = {
    "Jday, MeanT, Prec": ["Jday", "MeanT", "Prec"],
    "Jday, MaxT, MinT, Prec": ["Jday", "MaxT", "MinT", "Prec"],
    "Jday, MaxT, MinT, Prec, Radiation": ["Jday", "MaxT", "MinT", "Prec", "Radiation"],
    "Jday, MaxT, MinT, Prec, WindSpeed": ["Jday", "MaxT", "MinT", "Prec", "WindSpeed"],
    "Jday, MaxT, MinT, Prec, WindSpeed, Radiation, Humidity": ["Jday", "MaxT", "MinT", "Prec", "WindSpeed", "Radiation", "Humidity"],
    "Jday, MaxT, MinT, Prec, WindSpeed, Humidity": ["Jday", "MaxT", "MinT", "Prec", "WindSpeed", "Humidity"],
    "Jday, MaxT, MinT, Prec, Humidity": ["Jday", "MaxT", "MinT", "Prec", "Humidity"],
    "Prec (cm), Radiation (MJ/m2/day), WindSpeed (m/s), Humidity (%)": ["Prec", "Radiation", "WindSpeed", "Humidity"],
}

# DNDC format selector
st.header("DNDC climate file format")
selected_format = st.selectbox("Select the DNDC climate data format you want to preprocess your data for:", FORMATS)
st.caption(f"Selected: {selected_format}")
required_fields = FORMATS_TO_FIELDS[selected_format]

# Upload
st.header("Upload your meteorological data")
st.write("Upload a `.csv` file with meteorological data. **First row must be column headers.**")
uploaded_file = st.file_uploader("Drag and drop or click to upload a file", type=["csv"])
if uploaded_file is None:
    st.info("Waiting for file to be uploaded")
    st.stop()

# Read
try:
    df_raw = pd.read_csv(uploaded_file)
except Exception as e:
    st.error(f"Error reading the file: {e}")
    st.stop()

st.success("File uploaded successfully!")
st.subheader("Preview of your data: ")
st.dataframe(df_raw.head(), use_container_width=True)

# Mapping
st.subheader("Mapping data columns")
st.text("Map your data columns to DNDC format. Others will be discarded")
placeholder = "— Select column —"
options = [placeholder] + list(df_raw.columns)
mapping = {}
cols = st.columns(len(required_fields)) if len(required_fields) <= 6 else None
for i, field in enumerate(required_fields):
    widget = cols[i] if cols is not None else st
    mapping[field] = widget.selectbox(
        f"{field}",
        options=options,
        key=f"map_{field}"
    )

# Validate mapping
if placeholder in mapping.values():
    st.warning("Please map all required fields.")
    st.stop()

mapped_cols = list(mapping.values())
if len(set(mapped_cols)) != len(mapped_cols):
    st.error("Each DNDC field must map to a different source column. You have duplicates.")
    st.stop()

# Build DNDC-ordered dataframe
ordered_source_cols = [mapping[f] for f in required_fields]
dndc_df = df_raw[ordered_source_cols].copy()
dndc_df.columns = required_fields

st.text("Mapped data preview:")
st.dataframe(dndc_df.head(), use_container_width=True)

# Structural validation: missing & non-numeric
st.subheader("Validating data for missing/invalid values")
df = dndc_df.copy()

missing_mask = df.isna()
numeric_df = df.apply(pd.to_numeric, errors="coerce")
non_numeric_mask = numeric_df.isna() & ~missing_mask

def row_positions(mask_col) -> list[int]:
    return (np.where(mask_col.to_numpy())[0] + 1).tolist()

def preview(lst, k=8) -> str:
    if not lst:
        return ""
    return ", ".join(map(str, lst[:k])) + (f" … (+{len(lst)-k})" if len(lst) > k else "")

missing_rows = {col: row_positions(missing_mask[col]) for col in df.columns}
non_numeric_rows = {col: row_positions(non_numeric_mask[col]) for col in df.columns}

summary = pd.DataFrame({
    "missing_count": {col: len(missing_rows[col]) for col in df.columns},
    "missing_rows":  {col: preview(missing_rows[col]) for col in df.columns},
    "non_numeric_count": {col: len(non_numeric_rows[col]) for col in df.columns},
    "non_numeric_rows":  {col: preview(non_numeric_rows[col]) for col in df.columns},
}).reindex(df.columns)

st.caption("Issue summary by column (row locations shown as 1-based positions)")
st.dataframe(summary, use_container_width=True)

with st.expander("See full row locations for each column"):
    st.write("**Missing rows (1-based):**")
    st.json(missing_rows)
    st.write("**Non-numeric rows (1-based):**")
    st.json(non_numeric_rows)

total_issues = summary["missing_count"].sum() + summary["non_numeric_count"].sum()
if total_issues > 0:
    st.error("Missing or non-numeric values found. DNDC requires complete numeric data; runs will fail/crash if gaps or invalid entries exist. Please fix your data or restart with a different dataset")
    st.stop()
else:
    st.success("No missing or non-numeric values found. Data looks good!")

# Unit options
EXPECTED_UNITS = {
    "Jday": "day index (1–365/366)",
    "MeanT": "°C",
    "MaxT": "°C",
    "MinT": "°C",
    "Prec": "cm/day",
    "Radiation": "MJ/m²/day",
    "WindSpeed": "m/s",
    "Humidity": "%"
}
UNIT_OPTIONS = {
    "Jday": ["Jday (1–365/366)"],
    "MeanT": ["°C (DNDC)", "°F", "K"],
    "MaxT":  ["°C (DNDC)", "°F", "K"],
    "MinT":  ["°C (DNDC)", "°F", "K"],
    "Prec":  ["cm/day (DNDC)", "mm/day", "inch/day"],
    "WindSpeed": ["m/s (DNDC)", "km/h", "mph"],
    "Radiation": ["MJ/m²/day (DNDC)", "kWh/m²/day", "W/m² (daily mean)"],
    "Humidity": ["% (DNDC)", "fraction (0–1)"]
}

# Only include columns that actually exist in the current DF
exp_cols = [c for c in dndc_df.columns if c in EXPECTED_UNITS]

st.markdown("**Confirm the current units in your uploaded dataset:**")
selected_units = {}

if exp_cols:
    unit_cols = st.columns(min(4, len(exp_cols)))
    for i, col in enumerate(exp_cols):
        with unit_cols[i % len(unit_cols)]:
            selected_units[col] = st.selectbox(
                f"{col} — current unit",
                UNIT_OPTIONS[col],
                index=0,
                key=f"unit_{col}"
            )
else:
    st.info("No DNDC-recognized columns found to set units for.")
    selected_units = {}

# Converters
def to_celsius(x, from_unit):
    if from_unit == "°C (DNDC)": return x
    if from_unit == "°F":        return (x - 32.0) / 1.8
    if from_unit == "K":         return x - 273.15
    return x

def to_cm_per_day(x, from_unit):
    if from_unit == "cm/day (DNDC)": return x
    if from_unit == "mm/day":        return x / 10.0
    if from_unit == "inch/day":      return x * 2.54
    return x

def to_m_per_s(x, from_unit):
    if from_unit == "m/s (DNDC)": return x
    if from_unit == "km/h":       return x / 3.6
    if from_unit == "mph":        return x * 0.44704
    return x

def to_MJ_per_m2_per_day(x, from_unit):
    if from_unit == "MJ/m²/day (DNDC)": return x
    if from_unit == "kWh/m²/day":       return x * 3.6
    if from_unit == "W/m² (daily mean)": return x * 86.4  # 86,400 s/day ÷ 1e6 J/MJ
    return x

def to_percent(x, from_unit):
    if from_unit == "% (DNDC)":       return x
    if from_unit == "fraction (0–1)": return x * 100.0
    return x

CONVERTERS = {
    "MeanT": to_celsius,
    "MaxT": to_celsius,
    "MinT": to_celsius,
    "Prec": to_cm_per_day,
    "WindSpeed": to_m_per_s,
    "Radiation": to_MJ_per_m2_per_day,
    "Humidity": to_percent,
}

# Apply conversions ONLY to columns that exist
dndc_converted = dndc_df.copy()
for col in exp_cols:
    from_unit = selected_units.get(col, UNIT_OPTIONS[col][0])  # default to DNDC if missing
    if col in CONVERTERS:
        s = pd.to_numeric(dndc_converted[col], errors="coerce")
        dndc_converted[col] = CONVERTERS[col](s, from_unit)

# Preview (only for existing columns)
st.markdown("**Conversion preview (first 8 rows):**")
preview_cols = []
for col in exp_cols:
    preview_cols.append(dndc_df[col].rename(f"{col} (input: {selected_units[col]})"))
    preview_cols.append(dndc_converted[col].rename(f"{col} → {EXPECTED_UNITS[col]}"))
if preview_cols:
    st.dataframe(pd.concat(preview_cols, axis=1).head(8), use_container_width=True)
else:
    st.caption("No columns to preview.")

# Compact summary
change_summary = []
for col in exp_cols:
    src = selected_units[col]
    tgt = EXPECTED_UNITS[col]
    changed = "Yes" if "(DNDC)" not in src else "Maybe"
    change_summary.append({"Field": col, "From": src, "To (DNDC)": tgt, "Changed?": changed})
if change_summary:
    st.caption("Unit conversion summary")
    st.dataframe(pd.DataFrame(change_summary), use_container_width=True)

# Final validation
st.subheader("Final validation of converted data")
df = dndc_converted.copy()
EXPECTED_RANGES = {
    "Jday": (1, 366),
    "Humidity": (0, 100),
    "Prec": (0, np.inf),
    "WindSpeed": (0, np.inf),
    "Radiation": (0, np.inf),
    "MeanT": (-90, 60),
    "MaxT": (-50, 60),
    "MinT": (-90, 50),
}
fields = [c for c in df.columns if c in EXPECTED_RANGES]
below_masks = {}
above_masks = {}
for col in fields:
    lo, hi = EXPECTED_RANGES[col]
    s = pd.to_numeric(df[col], errors="coerce")
    below_masks[col] = s < lo
    above_masks[col] = s > hi

def row_positions(mask_col):
    return (np.where(mask_col.to_numpy())[0] + 1).tolist()

def preview(lst, k=8):
    return "" if not lst else ", ".join(map(str, lst[:k])) + (f" … (+{len(lst)-k})" if len(lst) > k else "")

summary = []
full_below = {}
full_above = {}
for col in fields:
    below_rows = row_positions(below_masks[col])
    above_rows = row_positions(above_masks[col])
    full_below[col] = below_rows
    full_above[col] = above_rows
    lo, hi = EXPECTED_RANGES[col]
    summary.append({
        "Field": col,
        "Allowed range": f"[{lo}, {hi}]",
        "below_count": len(below_rows),
        "below_rows": preview(below_rows),
        "above_count": len(above_rows),
        "above_rows": preview(above_rows),
    })

summary_df = pd.DataFrame(summary).set_index("Field")
st.caption("Range violations by column (row locations shown as 1-based positions)")
st.dataframe(summary_df, use_container_width=True)
with st.expander("See full row locations for each column"):
    st.write("**Below-range rows (1-based):**")
    st.json(full_below)
    st.write("**Above-range rows (1-based):**")
    st.json(full_above)

total_violations = summary_df["below_count"].sum() + summary_df["above_count"].sum()
if total_violations > 0:
    st.error("Out of range values found. DNDC may fail/crash if values are outside expected ranges. Please fix your data or restart with a different dataset")
    st.stop()
else:
    st.success("No out-of-range values found. Data looks good!")

# Download
st.subheader("Download converted DNDC climate data")
st.download_button(
    "⬇️ Download CSV",
    data=dndc_converted.to_csv(index=False),
    file_name="dndc_climate.csv",
    mime="text/csv",
)
