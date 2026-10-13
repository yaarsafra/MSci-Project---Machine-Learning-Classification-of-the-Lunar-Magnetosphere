#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Part 0: Download and prepare raw THEMIS spacecraft data.

This script downloads Level-2 THEMIS-B (THB) and THEMIS-C (THC) magnetic-field
and plasma measurements over the MSc project time range.

For each spacecraft it:
1. downloads GSM magnetic-field vectors from the FGM instrument;
2. derives magnetic-field clock and cone angles;
3. downloads ion velocity and number density from the MOM instrument;
4. derives plasma velocity magnitude; and
5. saves four raw CSV files for Part 1.

Outputs
-------
data/part1/raw_data/
    FULL_thb_fgm_raw_data_df.csv
    FULL_thb_mom_raw_data_df.csv
    FULL_thc_fgm_raw_data_df.csv
    FULL_thc_mom_raw_data_df.csv

Notes
-----
- Internet access is required because PySPEDAS downloads THEMIS mission data.
- The requested date range is intentionally kept identical to the MSc analysis.
- Column names retain the historical "raw" convention because Part 1 accepts
  these files directly and standardises the names after hourly averaging.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyspedas
import pytplot


# ---------------------------------------------------------------------------
# Project configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "part1" / "raw_data"

START_DATE = "2011-03-01 00:00:00"
END_DATE = "2024-09-29 00:00:00"

PROBES = ("b", "c")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_tplot_arrays(variable_name: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Return timestamps and values for a downloaded pytplot variable.

    Raises a clear error if the expected variable was not downloaded.
    """
    data = pytplot.get_data(variable_name)

    if data is None:
        raise RuntimeError(
            f"PySPEDAS did not return expected variable '{variable_name}'."
        )

    try:
        times = np.asarray(data[0])
        values = np.asarray(data[1])
    except (TypeError, IndexError) as exc:
        raise RuntimeError(
            f"Unexpected pytplot data structure for '{variable_name}'."
        ) from exc

    if len(times) != len(values):
        raise ValueError(
            f"Timestamp/value length mismatch for '{variable_name}': "
            f"{len(times)} timestamps vs {len(values)} values."
        )

    return times, values


def magnetic_field_angles(
    magnetic_field_gsm: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate magnetic-field clock and cone angles in degrees.

    Parameters
    ----------
    magnetic_field_gsm
        Array of shape (n_samples, 3) containing Bx, By and Bz in GSM
        coordinates.

    Returns
    -------
    clock_angle
        arctan2(By, Bz), in degrees.
    cone_angle
        arccos(Bx / |B|), in degrees.

    Invalid or zero-magnitude vectors are returned as NaN.
    """
    if magnetic_field_gsm.ndim != 2 or magnetic_field_gsm.shape[1] < 3:
        raise ValueError(
            "Magnetic-field data must have shape (n_samples, >=3)."
        )

    bx = magnetic_field_gsm[:, 0].astype(float)
    by = magnetic_field_gsm[:, 1].astype(float)
    bz = magnetic_field_gsm[:, 2].astype(float)

    magnitude = np.sqrt(bx**2 + by**2 + bz**2)

    valid = (
        np.isfinite(bx)
        & np.isfinite(by)
        & np.isfinite(bz)
        & (magnitude > 0)
    )

    clock_angle = np.full(len(bx), np.nan, dtype=float)
    cone_angle = np.full(len(bx), np.nan, dtype=float)

    clock_angle[valid] = np.degrees(
        np.arctan2(by[valid], bz[valid])
    )

    cosine = bx[valid] / magnitude[valid]
    cosine = np.clip(cosine, -1.0, 1.0)

    cone_angle[valid] = np.degrees(
        np.arccos(cosine)
    )

    return clock_angle, cone_angle


def download_fgm(probe: str) -> pd.DataFrame:
    """Download FGM data and calculate clock/cone angles for one probe."""
    spacecraft = f"th{probe}"
    variable_name = f"{spacecraft}_fgs_gsm"

    print(
        f"{spacecraft.upper()}: downloading FGM magnetic-field data..."
    )

    pyspedas.themis.fgm(
        probe=probe,
        trange=[START_DATE, END_DATE],
        level="l2",
        varnames=variable_name,
    )

    times, magnetic_field = get_tplot_arrays(variable_name)
    clock_angle, cone_angle = magnetic_field_angles(magnetic_field)

    return pd.DataFrame(
        {
            f"{spacecraft} fgm Unix Time": times,
            f"{spacecraft} raw clock angles": clock_angle,
            f"{spacecraft} raw cone angles": cone_angle,
        }
    )


def download_mom(probe: str) -> pd.DataFrame:
    """Download MOM plasma velocity/density data for one probe."""
    spacecraft = f"th{probe}"

    velocity_name = f"{spacecraft}_peim_velocity_gsm"
    density_name = f"{spacecraft}_peim_density"

    print(
        f"{spacecraft.upper()}: downloading MOM plasma data..."
    )

    pyspedas.themis.mom(
        probe=probe,
        trange=[START_DATE, END_DATE],
        level="l2",
        varnames=[velocity_name, density_name],
    )

    velocity_times, velocity = get_tplot_arrays(velocity_name)
    density_times, density = get_tplot_arrays(density_name)

    if velocity.ndim != 2 or velocity.shape[1] < 3:
        raise ValueError(
            f"{velocity_name} must contain x/y/z velocity components."
        )

    # The historical pipeline uses the velocity timestamps as the MOM time
    # coordinate. Verify that density timestamps are compatible before saving.
    if (
        len(velocity_times) != len(density_times)
        or not np.allclose(
            velocity_times,
            density_times,
            rtol=0.0,
            atol=1.0,
            equal_nan=False,
        )
    ):
        raise ValueError(
            f"{spacecraft.upper()} velocity and density timestamps are not "
            "aligned closely enough to combine safely."
        )

    x_velocity = velocity[:, 0].astype(float)

    velocity_magnitude = np.linalg.norm(
        velocity[:, :3].astype(float),
        axis=1,
    )

    density = np.asarray(density, dtype=float).squeeze()

    if density.ndim != 1:
        raise ValueError(
            f"{density_name} was expected to be one-dimensional."
        )

    return pd.DataFrame(
        {
            f"{spacecraft} mom Unix Time": velocity_times,
            f"{spacecraft} raw x velocity": x_velocity,
            f"{spacecraft} raw magnitude velocity": velocity_magnitude,
            f"{spacecraft} raw n density": density,
        }
    )


def save_raw_dataframe(
    dataframe: pd.DataFrame,
    output_path: Path,
) -> None:
    """Save one raw dataframe and print a concise summary."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output_path, index=False)

    first_time = pd.to_datetime(
        dataframe.iloc[0, 0],
        unit="s",
    )
    last_time = pd.to_datetime(
        dataframe.iloc[-1, 0],
        unit="s",
    )

    print(
        f"  saved {output_path.name}: "
        f"{len(dataframe):,} rows "
        f"({first_time} -> {last_time})"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Download all THB/THC raw datasets required by Part 1."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(
        "Downloading THEMIS data for "
        f"{START_DATE} -> {END_DATE}\n"
    )

    for probe in PROBES:
        spacecraft = f"th{probe}"

        fgm = download_fgm(probe)
        mom = download_mom(probe)

        save_raw_dataframe(
            fgm,
            RAW_DATA_DIR
            / f"FULL_{spacecraft}_fgm_raw_data_df.csv",
        )
        save_raw_dataframe(
            mom,
            RAW_DATA_DIR
            / f"FULL_{spacecraft}_mom_raw_data_df.csv",
        )

    print(
        "\nPart 0 complete. Raw files are ready in:\n"
        f"  {RAW_DATA_DIR}"
    )


if __name__ == "__main__":
    main()
