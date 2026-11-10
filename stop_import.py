"""Reading delivery stops from an uploaded CSV or Excel file.

Kept separate from the Streamlit app so it can be tested on its own.
"""
import io
import re

import pandas as pd

COLUMNS = ["Location", "Priority", "Volume (m³)"]
PRIORITIES = {"high": "High", "medium": "Medium", "low": "Low"}
DEFAULT_PRIORITY = "Medium"
DEFAULT_VOLUME = 1.0
MAX_STOPS = 200

# Accepted spellings for each column, after lower-casing and dropping units/punctuation.
HEADER_ALIASES = {
    "Location": {"location", "address", "stop", "place", "name", "destination"},
    "Priority": {"priority", "urgency"},
    "Volume (m³)": {"volume", "volumem3", "volumem³", "size", "load", "m3", "m³"},
}


def _normalise_header(header):
    return re.sub(r"[\s_()\-.]", "", str(header).strip().lower())


def template_csv():
    """A ready-to-fill example file."""
    example = pd.DataFrame({
        "Location": ["Guindy, Chennai", "Adyar, Chennai 600020", "Anna Nagar, Chennai"],
        "Priority": ["High", "Medium", "Low"],
        "Volume (m³)": [1.5, 1.0, 2.0],
    })
    return example.to_csv(index=False).encode("utf-8")


def _read(file_name, data):
    name = file_name.lower()
    if name.endswith((".xlsx", ".xlsm")):
        return pd.read_excel(io.BytesIO(data), dtype=object)
    if name.endswith(".csv"):
        try:
            return pd.read_csv(io.BytesIO(data), dtype=object, encoding="utf-8-sig")
        except UnicodeDecodeError:  # e.g. saved from Excel in a Windows code page
            return pd.read_csv(io.BytesIO(data), dtype=object, encoding="cp1252")
    raise ValueError("Upload a .csv or .xlsx file.")


def parse_stops_file(file_name, data):
    """Returns (stops DataFrame with COLUMNS, list of warnings). Raises ValueError if unusable."""
    try:
        raw = _read(file_name, data)
    except ValueError:
        raise
    except Exception as e:  # malformed file
        raise ValueError(f"Couldn't read {file_name}: {e}") from e

    mapping = {}
    for column in raw.columns:
        key = _normalise_header(column)
        for target, aliases in HEADER_ALIASES.items():
            if key in aliases and target not in mapping.values():
                mapping[column] = target
                break
    if "Location" not in mapping.values():
        raise ValueError("The file needs a 'Location' column (or 'Address'). Download the template to see the format.")
    df = raw[list(mapping)].rename(columns=mapping)

    warnings = []
    df["Location"] = df["Location"].map(lambda v: "" if pd.isna(v) else str(v).strip())
    df = df[df["Location"] != ""].reset_index(drop=True)
    if df.empty:
        raise ValueError("No stops found in the file: every Location is empty.")
    if len(df) > MAX_STOPS:
        warnings.append(f"Only the first {MAX_STOPS} of {len(df)} stops were imported.")
        df = df.head(MAX_STOPS)

    if "Priority" in df:
        given = df["Priority"].map(lambda v: "" if pd.isna(v) else str(v).strip().lower())
        bad = sorted({g for g in given if g and g not in PRIORITIES})
        if bad:
            warnings.append(f"Unknown priority {', '.join(repr(b) for b in bad)}: set to {DEFAULT_PRIORITY}.")
        df["Priority"] = given.map(lambda g: PRIORITIES.get(g, DEFAULT_PRIORITY))
    else:
        df["Priority"] = DEFAULT_PRIORITY

    if "Volume (m³)" in df:
        volumes = pd.to_numeric(df["Volume (m³)"], errors="coerce")
        invalid = df["Location"][volumes.isna() | (volumes <= 0)].tolist()
        if invalid:
            warnings.append(f"Missing or invalid volume for {', '.join(invalid[:5])}"
                            f"{' and others' if len(invalid) > 5 else ''}: set to {DEFAULT_VOLUME:g} m³.")
        df["Volume (m³)"] = volumes.where(volumes > 0, DEFAULT_VOLUME).astype(float)
    else:
        df["Volume (m³)"] = DEFAULT_VOLUME

    return df[COLUMNS], warnings
