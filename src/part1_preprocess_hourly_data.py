#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Part 1: Convert raw THEMIS measurements to an aligned hourly dataset.

This script reads the four raw CSV files created by Part 0, averages each
measurement stream into common one-hour windows, and merges the THB and THC
features into the single dataset consumed by Part 2.

Input
-----
data/part1/raw_data/
    FULL_thb_fgm_raw_data_df.csv
    FULL_thb_mom_raw_data_df.csv
    FULL_thc_fgm_raw_data_df.csv
    FULL_thc_mom_raw_data_df.csv

Outputs
-------
data/part1/hourly_data/
    FULL_thb_fgm_hourly_data.csv
    FULL_thb_mom_hourly_data.csv
    FULL_thc_fgm_hourly_data.csv
    FULL_thc_mom_hourly_data.csv

data/part1/
    FULL_merged_df.csv

Reproducibility note
--------------------
The original MSc analysis labelled each hourly window one second after the
clock hour (e.g. 01:00:01 rather than 01:00:00). That convention is preserved
here because Parts 2 and 4 have been validated against the resulting dataset.
Downstream code normalises timestamps when exact clock-hour matching is needed.
"""

from __future__ import annotations

from functools import reduce
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Project paths and configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PART_1_DATA_DIR = PROJECT_ROOT / "data" / "part1"
RAW_DATA_DIR = PART_1_DATA_DIR / "raw_data"
HOURLY_DATA_DIR = PART_1_DATA_DIR / "hourly_data"

MERGED_OUTPUT_PATH = PART_1_DATA_DIR / "FULL_merged_df.csv"

SECONDS_PER_HOUR = 3600
HOURLY_LABEL_OFFSET_SECONDS = 1

RAW_PATHS = {
    "thb_fgm": RAW_DATA_DIR / "FULL_thb_fgm_raw_data_df.csv",
    "thc_fgm": RAW_DATA_DIR / "FULL_thc_fgm_raw_data_df.csv",
    "thb_mom": RAW_DATA_DIR / "FULL_thb_mom_raw_data_df.csv",
    "thc_mom": RAW_DATA_DIR / "FULL_thc_mom_raw_data_df.csv",
}

HOURLY_PATHS = {
    "thb_fgm": HOURLY_DATA_DIR / "FULL_thb_fgm_hourly_data.csv",
    "thc_fgm": HOURLY_DATA_DIR / "FULL_thc_fgm_hourly_data.csv",
    "thb_mom": HOURLY_DATA_DIR / "FULL_thb_mom_hourly_data.csv",
    "thc_mom": HOURLY_DATA_DIR / "FULL_thc_mom_hourly_data.csv",
}


# ---------------------------------------------------------------------------
# Loading and validation
# ---------------------------------------------------------------------------

def load_raw_dataframe(path: Path) -> pd.DataFrame:
    """Load and validate one raw measurement stream."""
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find raw input:\n{path}\n\n"
            "Run Part 0 first or place the raw CSV in data/part1/raw_data."
        )

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError(f"{path.name} contains no rows.")

    time_column = df.columns[0]

    df[time_column] = pd.to_numeric(
        df[time_column],
        errors="coerce",
    )

    if df[time_column].isna().any():
        raise ValueError(
            f"{path.name}: first column '{time_column}' contains "
            "non-numeric or missing timestamps."
        )

    if not df[time_column].is_monotonic_increasing:
        df = df.sort_values(time_column).reset_index(drop=True)

    return df


def load_all_raw_data() -> dict[str, pd.DataFrame]:
    """Load all four Part 0 output files."""
    return {
        key: load_raw_dataframe(path)
        for key, path in RAW_PATHS.items()
    }


# ---------------------------------------------------------------------------
# Common hourly grid
# ---------------------------------------------------------------------------

def next_hour_plus_offset(timestamp: float) -> float:
    """
    Reproduce the historical hourly-start convention.

    Example
    -------
    2011-03-01 00:06:46 -> 2011-03-01 01:00:01
    """
    current_hour = np.floor(
        timestamp / SECONDS_PER_HOUR
    ) * SECONDS_PER_HOUR

    return (
        current_hour
        + SECONDS_PER_HOUR
        + HOURLY_LABEL_OFFSET_SECONDS
    )


def determine_common_time_range(
    raw_data: dict[str, pd.DataFrame],
) -> tuple[float, float]:
    """
    Find the time interval covered by all four measurement streams.

    The start is moved to the next historical hourly label (+1 second);
    the end is moved back to the preceding exact clock hour.
    """
    first_times = [
        float(df.iloc[0, 0])
        for df in raw_data.values()
    ]
    last_times = [
        float(df.iloc[-1, 0])
        for df in raw_data.values()
    ]

    latest_start = max(first_times)
    earliest_end = min(last_times)

    common_start = next_hour_plus_offset(latest_start)

    common_end = (
        np.floor(earliest_end / SECONDS_PER_HOUR)
        * SECONDS_PER_HOUR
    )

    if common_end <= common_start:
        raise ValueError(
            "The four raw datasets do not share a usable common time range."
        )

    return float(common_start), float(common_end)


# ---------------------------------------------------------------------------
# Hourly averaging
# ---------------------------------------------------------------------------

def hourly_average(
    raw_df: pd.DataFrame,
    common_start: float,
    common_end: float,
) -> pd.DataFrame:
    """
    Average one raw measurement stream into historical one-hour windows.

    This reproduces the MSc windowing convention:
        [hour_start, hour_start + 3600 seconds]

    If an hour contains no valid observations, its feature values are NaN so
    that all four measurement streams retain the same hourly time grid.
    """
    measurement_times = raw_df.iloc[:, 0].to_numpy(dtype=float)

    total_hours = int(
        (common_end - common_start) / SECONDS_PER_HOUR
    )

    if total_hours <= 0:
        raise ValueError(
            "Common time range is too short to form hourly averages."
        )

    hourly_rows: list[list[float]] = []

    for hour_index in range(total_hours):
        start_time = (
            common_start
            + hour_index * SECONDS_PER_HOUR
        )
        end_time = start_time + SECONDS_PER_HOUR

        start_index = np.searchsorted(
            measurement_times,
            start_time,
            side="left",
        )
        end_index = (
            np.searchsorted(
                measurement_times,
                end_time,
                side="right",
            )
            - 1
        )

        valid_window = (
            start_index < len(raw_df)
            and end_index >= 0
            and start_index <= end_index
            and measurement_times[start_index] > start_time
            and measurement_times[start_index] < end_time
            and measurement_times[end_index] > start_time
            and measurement_times[end_index] < end_time
        )

        if not valid_window:
            hourly_rows.append(
                [start_time]
                + [np.nan] * (len(raw_df.columns) - 1)
            )
            continue

        hourly_slice = raw_df.iloc[
            start_index:end_index + 1,
            1:,
        ]

        averages = (
            hourly_slice
            .mean(axis=0, skipna=True)
            .to_numpy(dtype=float)
        )

        hourly_rows.append(
            [start_time] + averages.tolist()
        )

    columns = [
        "Start Unix Time",
        *raw_df.columns[1:],
    ]

    return pd.DataFrame(
        hourly_rows,
        columns=columns,
    )


def save_hourly_data(
    key: str,
    hourly_df: pd.DataFrame,
) -> None:
    """Save one intermediate hourly dataset."""
    output_path = HOURLY_PATHS[key]
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    hourly_df.to_csv(
        output_path,
        index=False,
    )

    missing_rows = (
        hourly_df.iloc[:, 1:]
        .isna()
        .all(axis=1)
        .sum()
    )

    print(
        f"{key}: {len(hourly_df):,} hourly rows "
        f"({missing_rows:,} completely missing hours)"
    )


# ---------------------------------------------------------------------------
# Merge hourly streams
# ---------------------------------------------------------------------------

def merge_hourly_data(
    hourly_data: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Merge all streams explicitly on the shared hourly timestamp.

    Using a key-based merge is safer than concatenating rows by position and
    makes any alignment failure visible immediately.
    """
    ordered_keys = [
        "thb_fgm",
        "thc_fgm",
        "thb_mom",
        "thc_mom",
    ]

    dataframes = [
        hourly_data[key]
        for key in ordered_keys
    ]

    merged = reduce(
        lambda left, right: left.merge(
            right,
            on="Start Unix Time",
            how="inner",
            validate="one_to_one",
        ),
        dataframes,
    )

    expected_rows = len(dataframes[0])

    if len(merged) != expected_rows:
        raise ValueError(
            "Hourly streams did not merge one-to-one across the full "
            f"time grid: expected {expected_rows:,} rows, "
            f"got {len(merged):,}."
        )

    merged = merged.rename(
        columns={
            "Start Unix Time": "Mean Start Unix Time",
        }
    )

    # Remove the historical word "raw" from feature names while retaining
    # the spacecraft/instrument-independent physical quantities.
    merged.columns = merged.columns.str.replace(
        " raw ",
        " ",
        regex=False,
    )

    return merged


def validate_merged_data(merged: pd.DataFrame) -> None:
    """Run final structural checks before writing Part 1 output."""
    time_column = "Mean Start Unix Time"

    if merged[time_column].duplicated().any():
        raise ValueError(
            "Merged hourly dataset contains duplicate timestamps."
        )

    if not merged[time_column].is_monotonic_increasing:
        raise ValueError(
            "Merged hourly timestamps are not monotonically increasing."
        )

    expected_feature_columns = {
        "thb clock angles",
        "thb cone angles",
        "thb x velocity",
        "thb magnitude velocity",
        "thb n density",
        "thc clock angles",
        "thc cone angles",
        "thc x velocity",
        "thc magnitude velocity",
        "thc n density",
    }

    missing = expected_feature_columns - set(
        merged.columns
    )

    if missing:
        raise ValueError(
            f"Merged dataset is missing expected features: {sorted(missing)}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the complete Part 1 preprocessing workflow."""
    raw_data = load_all_raw_data()

    common_start, common_end = (
        determine_common_time_range(raw_data)
    )

    print(
        "Common hourly range:\n"
        f"  start: "
        f"{pd.to_datetime(common_start, unit='s')} "
        f"(Unix {common_start:.0f})\n"
        f"  end:   "
        f"{pd.to_datetime(common_end, unit='s')} "
        f"(Unix {common_end:.0f})\n"
    )

    hourly_data: dict[str, pd.DataFrame] = {}

    for key, raw_df in raw_data.items():
        hourly_df = hourly_average(
            raw_df,
            common_start,
            common_end,
        )
        hourly_data[key] = hourly_df
        save_hourly_data(
            key,
            hourly_df,
        )

    merged = merge_hourly_data(hourly_data)
    validate_merged_data(merged)

    PART_1_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    merged.to_csv(
        MERGED_OUTPUT_PATH,
        index=False,
    )

    first_time = pd.to_datetime(
        merged["Mean Start Unix Time"].iloc[0],
        unit="s",
    )
    last_time = pd.to_datetime(
        merged["Mean Start Unix Time"].iloc[-1],
        unit="s",
    )

    print(
        "\nPart 1 complete.\n"
        f"  merged rows: {len(merged):,}\n"
        f"  time range:  {first_time} -> {last_time}\n"
        f"  saved to:    {MERGED_OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()
