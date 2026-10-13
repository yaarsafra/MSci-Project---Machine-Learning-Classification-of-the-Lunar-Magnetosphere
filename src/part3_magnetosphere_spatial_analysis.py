#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Part 3: Attach spacecraft coordinates and map magnetosphere boundaries.

This script combines the current Part 2 region labels with pre-computed THEMIS
spacecraft coordinates, overlays independently identified bow-shock crossings,
and visualises the magnetopause (MP) and bow-shock (BS) boundaries.

For each spacecraft:
1. load the current Part 2 labels;
2. load hourly GSM coordinates;
3. merge labels and coordinates by hourly timestamp;
4. map bow-shock crossing timestamps onto the nearest coordinate sample.

The THB and THC samples are then combined to produce:
- a Y-Z cross-section with fitted MP and BS circles;
- an X-Y view with mean X boundary locations and +/-1 sigma bands;
- an X-Z view with mean X boundary locations and +/-1 sigma bands;
- a CSV summary of fitted boundary parameters.

Inputs
------
data/part2/
    labelled_thb_all_features.csv
    labelled_thc_all_features.csv

data/part3/
    thb_coords_labelled.csv
    thc_coords_labelled.csv
    valid_peaks_thb_14yr_6hr.csv
    valid_peaks_thc_14yr_6hr.csv

Outputs
-------
figures/
    part3_magnetosphere_yz.png
    part3_magnetosphere_xy.png
    part3_magnetosphere_xz.png

results/part3/
    boundary_fit_summary.csv
    thb_coordinates_with_regions.csv
    thc_coordinates_with_regions.csv

Notes
-----
- Coordinate values are treated as kilometres, matching the original MSc
  plotting code, and are converted to Earth radii for visualisation/fitting.
- Only the nightside / magnetotail region (X < 0) is used for the boundary
  plots, matching the original analysis.
- Bow-shock crossings overwrite the region label at the nearest hourly
  coordinate sample, reproducing the original plotting intent.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Project configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PART_2_DATA_DIR = PROJECT_ROOT / "data" / "part2"
PART_3_DATA_DIR = PROJECT_ROOT / "data" / "part3"

FIGURES_DIR = PROJECT_ROOT / "figures"
RESULTS_DIR = PROJECT_ROOT / "results" / "part3"

THB_LABELS_PATH = PART_2_DATA_DIR / "labelled_thb_all_features.csv"
THC_LABELS_PATH = PART_2_DATA_DIR / "labelled_thc_all_features.csv"

THB_COORDS_PATH = PART_3_DATA_DIR / "thb_coords_labelled.csv"
THC_COORDS_PATH = PART_3_DATA_DIR / "thc_coords_labelled.csv"

THB_BOW_SHOCK_PATH = (
    PART_3_DATA_DIR / "valid_peaks_thb_14yr_6hr.csv"
)
THC_BOW_SHOCK_PATH = (
    PART_3_DATA_DIR / "valid_peaks_thc_14yr_6hr.csv"
)

EARTH_RADIUS_KM = 6378.0

# Coordinate samples are hourly. A bow-shock timestamp should therefore have
# a nearby coordinate sample. This guard prevents a crossing from being
# silently attached many hours/days away if an input file is misaligned.
MAX_BOW_SHOCK_OFFSET = pd.Timedelta(hours=3)

BOUNDARY_LABELS = ("MP boundary", "Bow Shock")

BOUNDARY_COLORS = {
    "MP boundary": "blue",
    "Bow Shock": "orange",
}

BOUNDARY_DISPLAY_NAMES = {
    "MP boundary": "Magnetopause",
    "Bow Shock": "Bow Shock",
}


# ---------------------------------------------------------------------------
# Input validation and timestamp handling
# ---------------------------------------------------------------------------

def require_columns(
    dataframe: pd.DataFrame,
    required: list[str],
    source: Path,
) -> None:
    """Raise a clear error if an input file is missing required columns."""
    missing = [
        column
        for column in required
        if column not in dataframe.columns
    ]

    if missing:
        raise ValueError(
            f"{source.name} is missing required column(s): {missing}. "
            f"Available columns: {list(dataframe.columns)}"
        )


def unix_to_hour_key(
    unix_time: pd.Series,
) -> pd.Series:
    """
    Convert Unix seconds to integer Unix seconds at the nearest clock hour.

    Part 1 historically labels windows one second after the hour. Rounding to
    an hourly key therefore allows those labels to merge safely with coordinate
    files timestamped exactly on the hour.
    """
    numeric = pd.to_numeric(
        unix_time,
        errors="raise",
    )

    rounded = pd.to_datetime(
        numeric,
        unit="s",
    ).dt.round("h")

    return rounded.astype("int64") // 10**9


def load_part2_labels(path: Path) -> pd.DataFrame:
    """Load only the current Part 2 timestamp and human-readable label."""
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find Part 2 label file:\n{path}"
        )

    dataframe = pd.read_csv(path)
    require_columns(
        dataframe,
        ["Unix Time", "Label"],
        path,
    )

    labels = dataframe[
        ["Unix Time", "Label"]
    ].copy()

    labels["Hour Key"] = unix_to_hour_key(
        labels["Unix Time"]
    )

    if labels["Hour Key"].duplicated().any():
        raise ValueError(
            f"{path.name} contains duplicate hourly timestamps "
            "after rounding."
        )

    return labels


def load_coordinates(path: Path) -> pd.DataFrame:
    """
    Load GSM coordinates.

    Any historical Label column in this file is intentionally ignored. The
    current Part 2 labels are merged in explicitly so Part 3 cannot silently
    use stale labels from an older clustering run.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find coordinate file:\n{path}"
        )

    dataframe = pd.read_csv(path)
    require_columns(
        dataframe,
        ["Unix Time", "X", "Y", "Z"],
        path,
    )

    coordinates = dataframe[
        ["Unix Time", "X", "Y", "Z"]
    ].copy()

    for column in ["Unix Time", "X", "Y", "Z"]:
        coordinates[column] = pd.to_numeric(
            coordinates[column],
            errors="coerce",
        )

    if coordinates[
        ["Unix Time", "X", "Y", "Z"]
    ].isna().any().any():
        bad_rows = int(
            coordinates[
                ["Unix Time", "X", "Y", "Z"]
            ].isna().any(axis=1).sum()
        )
        raise ValueError(
            f"{path.name} contains {bad_rows:,} row(s) with missing or "
            "non-numeric time/coordinate values."
        )

    coordinates["Hour Key"] = unix_to_hour_key(
        coordinates["Unix Time"]
    )

    if coordinates["Hour Key"].duplicated().any():
        raise ValueError(
            f"{path.name} contains duplicate hourly timestamps "
            "after rounding."
        )

    return coordinates


def combine_labels_and_coordinates(
    labels_path: Path,
    coordinates_path: Path,
    spacecraft: str,
) -> pd.DataFrame:
    """Attach the current Part 2 region labels to one coordinate dataset."""
    labels = load_part2_labels(labels_path)
    coordinates = load_coordinates(coordinates_path)

    # The Part 2 timestamp is retained as the canonical Unix Time.
    labels = labels.rename(
        columns={
            "Unix Time": "Part 2 Unix Time",
        }
    )

    coordinates = coordinates.rename(
        columns={
            "Unix Time": "Coordinate Unix Time",
        }
    )

    combined = labels.merge(
        coordinates,
        on="Hour Key",
        how="inner",
        validate="one_to_one",
    )

    if combined.empty:
        raise ValueError(
            f"{spacecraft.upper()}: no timestamps matched between "
            "Part 2 labels and coordinate data."
        )

    matched_fraction = len(combined) / len(labels)

    print(
        f"{spacecraft.upper()}: matched {len(combined):,} of "
        f"{len(labels):,} Part 2 labels to coordinates "
        f"({matched_fraction:.1%})."
    )

    # Quantify the original timestamp difference before dropping the auxiliary
    # coordinate timestamp.
    time_offset_seconds = (
        combined["Coordinate Unix Time"]
        - combined["Part 2 Unix Time"]
    ).abs()

    print(
        f"  coordinate/label timestamp offset: "
        f"median={time_offset_seconds.median():.1f}s, "
        f"max={time_offset_seconds.max():.1f}s"
    )

    combined = combined.rename(
        columns={
            "Part 2 Unix Time": "Unix Time",
        }
    )

    combined["Spacecraft"] = spacecraft.upper()

    combined = combined[
        [
            "Unix Time",
            "Spacecraft",
            "X",
            "Y",
            "Z",
            "Label",
        ]
    ].sort_values("Unix Time").reset_index(drop=True)

    return combined


# ---------------------------------------------------------------------------
# Bow-shock crossings
# ---------------------------------------------------------------------------

def load_bow_shock_times(path: Path) -> np.ndarray:
    """Load bow-shock crossing dates and return Unix seconds."""
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find bow-shock crossing file:\n{path}"
        )

    dataframe = pd.read_csv(path)
    require_columns(
        dataframe,
        ["date"],
        path,
    )

    dates = pd.to_datetime(
        dataframe["date"],
        errors="coerce",
        utc=True,
    )

    if dates.isna().any():
        raise ValueError(
            f"{path.name} contains invalid values in its 'date' column."
        )

    return (
        dates.astype("int64")
        // 10**9
    ).to_numpy(dtype=np.int64)


def nearest_indices(
    sorted_unix_times: np.ndarray,
    target_unix_times: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Find the nearest sorted timestamp for every target timestamp.

    Returns
    -------
    indices
        Row index of the nearest coordinate timestamp.
    offsets
        Absolute time separation in seconds.
    """
    if len(sorted_unix_times) == 0:
        raise ValueError(
            "Cannot match bow-shock crossings to an empty coordinate array."
        )

    insertion_points = np.searchsorted(
        sorted_unix_times,
        target_unix_times,
    )

    right = np.clip(
        insertion_points,
        0,
        len(sorted_unix_times) - 1,
    )
    left = np.clip(
        insertion_points - 1,
        0,
        len(sorted_unix_times) - 1,
    )

    left_distance = np.abs(
        target_unix_times
        - sorted_unix_times[left]
    )
    right_distance = np.abs(
        target_unix_times
        - sorted_unix_times[right]
    )

    choose_right = right_distance < left_distance

    indices = np.where(
        choose_right,
        right,
        left,
    )
    offsets = np.minimum(
        left_distance,
        right_distance,
    )

    return indices.astype(int), offsets.astype(float)


def attach_bow_shock_labels(
    coordinates_with_regions: pd.DataFrame,
    bow_shock_path: Path,
    spacecraft: str,
) -> pd.DataFrame:
    """
    Relabel the nearest hourly coordinate sample for each bow-shock crossing.
    """
    output = coordinates_with_regions.copy()

    bow_shock_times = load_bow_shock_times(
        bow_shock_path
    )

    coordinate_times = (
        output["Unix Time"]
        .to_numpy(dtype=float)
    )

    match_indices, offsets = nearest_indices(
        coordinate_times,
        bow_shock_times,
    )

    max_offset_seconds = (
        MAX_BOW_SHOCK_OFFSET.total_seconds()
    )

    acceptable = offsets <= max_offset_seconds

    if not np.all(acceptable):
        rejected = int((~acceptable).sum())
        worst_hours = (
            float(offsets.max()) / 3600
        )

        raise ValueError(
            f"{spacecraft.upper()}: {rejected} bow-shock crossing(s) "
            f"have no coordinate sample within "
            f"{MAX_BOW_SHOCK_OFFSET}. "
            f"Largest nearest offset is {worst_hours:.2f} hours."
        )

    output.loc[
        np.unique(match_indices),
        "Label",
    ] = "Bow Shock"

    print(
        f"{spacecraft.upper()}: attached "
        f"{len(np.unique(match_indices)):,} unique bow-shock samples "
        f"from {len(bow_shock_times):,} crossing timestamps "
        f"(max offset {offsets.max() / 60:.1f} min)."
    )

    return output


# ---------------------------------------------------------------------------
# Boundary fitting
# ---------------------------------------------------------------------------

def fit_circle(
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[float, float, float, float]:
    """
    Fit a circle using the moment-based algebraic method used in the MSc work.

    Returns
    -------
    x_center, y_center
        Fitted circle centre.
    radius
        Fitted radius.
    residual_std
        Standard deviation of radial residuals around the fitted circle.

    Notes
    -----
    The original Part 3 implementation contained a radius typo:
        sqrt(uc*2 + vc*2 + Suu + Svv)
    This implementation uses the correct squared terms:
        sqrt(uc**2 + vc**2 + Suu + Svv)
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]

    if len(x) < 3:
        raise ValueError(
            "At least three finite points are required to fit a circle."
        )

    x_mean = x.mean()
    y_mean = y.mean()

    u = x - x_mean
    v = y - y_mean

    suu = np.mean(u**2)
    suv = np.mean(u * v)
    svv = np.mean(v**2)

    suuu = np.mean(u**3)
    suvv = np.mean(u * v**2)
    svvv = np.mean(v**3)
    svuu = np.mean(v * u**2)

    matrix = np.array(
        [
            [suu, suv],
            [suv, svv],
        ],
        dtype=float,
    )

    vector = np.array(
        [
            0.5 * (suuu + suvv),
            0.5 * (svvv + svuu),
        ],
        dtype=float,
    )

    try:
        u_center, v_center = np.linalg.solve(
            matrix,
            vector,
        )
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "Circle fit failed because the boundary points are "
            "geometrically degenerate."
        ) from exc

    x_center = u_center + x_mean
    y_center = v_center + y_mean

    radius_squared = (
        u_center**2
        + v_center**2
        + suu
        + svv
    )

    if radius_squared <= 0:
        raise ValueError(
            "Circle fit produced a non-positive squared radius."
        )

    radius = np.sqrt(radius_squared)

    radial_distance = np.sqrt(
        (x - x_center) ** 2
        + (y - y_center) ** 2
    )

    residual_std = np.std(
        radial_distance - radius
    )

    return (
        float(x_center),
        float(y_center),
        float(radius),
        float(residual_std),
    )


def prepare_plot_data(
    combined: pd.DataFrame,
) -> pd.DataFrame:
    """Restrict to nightside data and convert coordinates to Earth radii."""
    nightside = combined.loc[
        combined["X"] < 0
    ].copy()

    for coordinate in ["X", "Y", "Z"]:
        nightside[f"{coordinate}_RE"] = (
            nightside[coordinate]
            / EARTH_RADIUS_KM
        )

    return nightside


def calculate_boundary_summary(
    plot_data: pd.DataFrame,
) -> pd.DataFrame:
    """Fit Y-Z circles and calculate X-location statistics for both boundaries."""
    rows: list[dict[str, float | str | int]] = []

    for label in BOUNDARY_LABELS:
        points = plot_data.loc[
            plot_data["Label"] == label
        ].copy()

        if len(points) < 3:
            raise ValueError(
                f"Not enough {label!r} points for boundary fitting: "
                f"{len(points)}."
            )

        y_center, z_center, radius, radial_std = fit_circle(
            points["Y_RE"].to_numpy(),
            points["Z_RE"].to_numpy(),
        )

        rows.append(
            {
                "Boundary": BOUNDARY_DISPLAY_NAMES[label],
                "Points": len(points),
                "Y Centre (R_E)": y_center,
                "Z Centre (R_E)": z_center,
                "Radius (R_E)": radius,
                "Radial Residual Std (R_E)": radial_std,
                "Mean X (R_E)": points["X_RE"].mean(),
                "X Std (R_E)": points["X_RE"].std(),
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def scatter_boundary_points(
    ax: plt.Axes,
    data: pd.DataFrame,
    horizontal: str,
    vertical: str,
) -> None:
    """
    Plot magnetopause and bow-shock samples using the original MSc styling.
    """
    for label in BOUNDARY_LABELS:
        subset = data.loc[
            data["Label"] == label
        ]

        point_label = (
            "MP boundary"
            if label == "MP boundary"
            else "BS boundary"
        )

        ax.scatter(
            subset[horizontal],
            subset[vertical],
            s=200,
            alpha=0.3,
            color=BOUNDARY_COLORS[label],
            marker=".",
            label=point_label,
            zorder=1,
        )


def plot_yz_cross_section(
    plot_data: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    """
    Plot the Y-Z magnetosphere cross-section using the original MSc styling.

    The boundary points retain the original blue/orange colours, while the
    fitted circles and centre markers use darker shades so they stand out.
    """
    fig, ax = plt.subplots(figsize=(8.5, 8))
    ax.set_facecolor("whitesmoke")
    ax.grid(zorder=0)

    scatter_boundary_points(
        ax,
        plot_data,
        "Y_RE",
        "Z_RE",
    )

    theta = np.linspace(
        0,
        2 * np.pi,
        300,
    )

    fit_colors = {
        "MP boundary": "darkblue",
        "Bow Shock": "darkorange",
    }

    fill_colors = {
        "MP boundary": "blue",
        "Bow Shock": "orange",
    }

    fit_labels = {
        "MP boundary": "MP fit",
        "Bow Shock": "BS fit",
    }

    for label in BOUNDARY_LABELS:
        display_name = BOUNDARY_DISPLAY_NAMES[label]

        row = summary.loc[
            summary["Boundary"] == display_name
        ].iloc[0]

        y_center = row["Y Centre (R_E)"]
        z_center = row["Z Centre (R_E)"]
        radius = row["Radius (R_E)"]
        residual_std = row[
            "Radial Residual Std (R_E)"
        ]

        outer_radius = radius + residual_std
        inner_radius = max(
            radius - residual_std,
            0.0,
        )

        y_outer = (
            y_center
            + outer_radius * np.cos(theta)
        )
        z_outer = (
            z_center
            + outer_radius * np.sin(theta)
        )

        y_inner = (
            y_center
            + inner_radius * np.cos(theta)
        )
        z_inner = (
            z_center
            + inner_radius * np.sin(theta)
        )

        ring_y = np.concatenate(
            [y_outer, y_inner[::-1]]
        )
        ring_z = np.concatenate(
            [z_outer, z_inner[::-1]]
        )

        ax.fill(
            ring_y,
            ring_z,
            color=fill_colors[label],
            alpha=0.2,
            zorder=1,
        )

        ax.plot(
            y_center + radius * np.cos(theta),
            z_center + radius * np.sin(theta),
            color=fit_colors[label],
            linewidth=2,
            linestyle="--",
            label=fit_labels[label],
            zorder=3,
        )

        ax.scatter(
            y_center,
            z_center,
            color=fit_colors[label],
            marker="x",
            s=1000,
            linewidths=5,
            edgecolors="k",
            zorder=4,
        )

    ax.set_xlabel(
        r"Y Coordinate ($R_E$)",
        fontsize=30,
    )
    ax.set_ylabel(
        r"Z Coordinate ($R_E$)",
        fontsize=30,
    )

    ax.set_xticks(
        np.arange(-60, 61, 20)
    )
    ax.set_yticks(
        np.arange(-60, 61, 20)
    )
    ax.set_xticks(
        np.arange(-60, 61, 10),
        minor=True,
    )
    ax.set_yticks(
        np.arange(-60, 61, 10),
        minor=True,
    )

    ax.tick_params(
        axis="both",
        which="major",
        labelsize=20,
    )

    ax.set_xlim(-60, 60)
    ax.set_ylim(-60, 60)
    ax.set_aspect("equal", adjustable="box")

    ax.legend(
        loc="lower left",
        fontsize=20,
    )

    fig.tight_layout()

    fig.savefig(
        FIGURES_DIR / "part3_magnetosphere_yz.png",
        dpi=500,
        bbox_inches="tight",
    )

    plt.show()
    plt.close(fig)


def plot_x_projection(
    plot_data: pd.DataFrame,
    summary: pd.DataFrame,
    vertical_coordinate: str,
    output_name: str,
) -> None:
    """
    Plot the X-Y or X-Z projection using the original MSc formatting.
    """
    vertical_key = f"{vertical_coordinate}_RE"

    if vertical_coordinate == "Y":
        figsize = (8, 10)
        x_minor_step = 2.5
        y_minor_step = 5.0
    elif vertical_coordinate == "Z":
        figsize = (12, 8)
        x_minor_step = 2.5
        y_minor_step = 5.0
    else:
        raise ValueError(
            "vertical_coordinate must be 'Y' or 'Z'."
        )

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor("whitesmoke")
    ax.grid(zorder=0)

    scatter_boundary_points(
        ax,
        plot_data,
        "X_RE",
        vertical_key,
    )

    fit_colors = {
        "MP boundary": "darkblue",
        "Bow Shock": "darkorange",
    }

    fill_colors = {
        "MP boundary": "blue",
        "Bow Shock": "orange",
    }

    fit_labels = {
        "MP boundary": "MP fit",
        "Bow Shock": "BS fit",
    }

    for label in BOUNDARY_LABELS:
        display_name = BOUNDARY_DISPLAY_NAMES[label]

        row = summary.loc[
            summary["Boundary"] == display_name
        ].iloc[0]

        x_mean = row["Mean X (R_E)"]
        x_std = row["X Std (R_E)"]

        ax.axvline(
            x_mean,
            color=fit_colors[label],
            linestyle="--",
            linewidth=2,
            label=fit_labels[label],
            zorder=3,
        )

        ax.axvspan(
            x_mean - x_std,
            x_mean + x_std,
            color=fill_colors[label],
            alpha=0.2,
            zorder=1,
        )

    ax.set_xlabel(
        r"X Coordinate ($R_E$)",
        fontsize=30,
    )
    ax.set_ylabel(
        rf"{vertical_coordinate} Coordinate ($R_E$)",
        fontsize=30,
    )

    # Match the original Part 3 tick spacing.
    ax.xaxis.set_major_locator(
        MultipleLocator(10)
    )
    ax.yaxis.set_major_locator(
        MultipleLocator(10)
    )
    ax.xaxis.set_minor_locator(
        MultipleLocator(x_minor_step)
    )
    ax.yaxis.set_minor_locator(
        MultipleLocator(y_minor_step)
    )

    ax.tick_params(
        axis="both",
        which="major",
        labelsize=20,
    )

    ax.legend(
        loc="center right",
        fontsize=20,
    )

    fig.tight_layout()

    fig.savefig(
        FIGURES_DIR / output_name,
        dpi=500,
        bbox_inches="tight",
    )

    plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the complete Part 3 spatial-boundary analysis."""
    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    thb = combine_labels_and_coordinates(
        THB_LABELS_PATH,
        THB_COORDS_PATH,
        spacecraft="thb",
    )
    thc = combine_labels_and_coordinates(
        THC_LABELS_PATH,
        THC_COORDS_PATH,
        spacecraft="thc",
    )

    thb = attach_bow_shock_labels(
        thb,
        THB_BOW_SHOCK_PATH,
        spacecraft="thb",
    )
    thc = attach_bow_shock_labels(
        thc,
        THC_BOW_SHOCK_PATH,
        spacecraft="thc",
    )

    thb.to_csv(
        RESULTS_DIR / "thb_coordinates_with_regions.csv",
        index=False,
    )
    thc.to_csv(
        RESULTS_DIR / "thc_coordinates_with_regions.csv",
        index=False,
    )

    combined = (
        pd.concat(
            [thb, thc],
            ignore_index=True,
        )
        .sort_values(
            ["Unix Time", "Spacecraft"]
        )
        .reset_index(drop=True)
    )

    plot_data = prepare_plot_data(
        combined
    )

    print("\nBoundary counts on the nightside (X < 0):")
    for label in BOUNDARY_LABELS:
        count = int(
            (plot_data["Label"] == label).sum()
        )
        print(
            f"  {BOUNDARY_DISPLAY_NAMES[label]}: "
            f"{count:,}"
        )

    summary = calculate_boundary_summary(
        plot_data
    )

    summary.to_csv(
        RESULTS_DIR / "boundary_fit_summary.csv",
        index=False,
    )

    print("\nBoundary fit summary:")
    print(
        summary.to_string(
            index=False,
            float_format=lambda value: f"{value:.3f}",
        )
    )

    plot_yz_cross_section(
        plot_data,
        summary,
    )

    plot_x_projection(
        plot_data,
        summary,
        vertical_coordinate="Y",
        output_name="part3_magnetosphere_xy.png",
    )

    plot_x_projection(
        plot_data,
        summary,
        vertical_coordinate="Z",
        output_name="part3_magnetosphere_xz.png",
    )

    print(
        "\nPart 3 complete.\n"
        f"  Figures: {FIGURES_DIR}\n"
        f"  Results: {RESULTS_DIR}"
    )


if __name__ == "__main__":
    main()
