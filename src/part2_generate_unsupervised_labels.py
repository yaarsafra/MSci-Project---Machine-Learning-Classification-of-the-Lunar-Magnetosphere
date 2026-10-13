#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Part 2: Generate magnetotail region labels using hierarchical K-means.

Workflow
--------
1. Split the merged hourly THEMIS dataset into THB and THC measurements.
2. First-stage K-means:
      plasma velocity magnitude + number density
      -> outside magnetotail / inside magnetotail
3. Remove isolated one-hour classification flips.
4. Second-stage K-means on samples classified inside the magnetotail:
      magnetic-field clock angle + cone angle
      -> south lobe / north lobe
5. Identify magnetopause crossings from transitions in the first-stage label.
6. Save labelled THB and THC datasets for Parts 3 and 4.

Important reproducibility detail
--------------------------------
The original MSc analysis was run with a scikit-learn version whose KMeans
default used 10 initialisations. Newer scikit-learn versions use n_init="auto",
which can change the fitted clusters substantially for this dataset.

For reproducibility, this implementation therefore sets n_init=10 explicitly.

K-means cluster IDs (0/1) are arbitrary. Rather than assuming that a particular
raw K-means ID always has the same physical meaning, this script assigns the
final semantic classes using the fitted cluster centres.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Project configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = PROJECT_ROOT / "data" / "part1" / "FULL_merged_df.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "part2"

THB_OUTPUT_PATH = OUTPUT_DIR / "labelled_thb_all_features.csv"
THC_OUTPUT_PATH = OUTPUT_DIR / "labelled_thc_all_features.csv"

RANDOM_STATE = 42
N_INIT = 10
CLEANING_WINDOW_HOURS = 4

# Stable semantic labels used by downstream Parts 3 and 4.
OUTSIDE_MAGNETOTAIL = 0
INSIDE_MAGNETOTAIL = 1

NOT_MAGNETOTAIL = 0
SOUTH_LOBE = 2
NORTH_LOBE = 3


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def load_merged_hourly_data(path: Path) -> pd.DataFrame:
    """Load the merged hourly dataset produced by Part 1."""
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find Part 1 output:\n{path}\n\n"
            "Run Part 1 first, or update INPUT_PATH."
        )

    df = pd.read_csv(path)

    # Part 1 historically used both "Mean Start Unix Time" and "Unix Time".
    if "Mean Start Unix Time" in df.columns:
        df = df.rename(columns={"Mean Start Unix Time": "Unix Time"})

    if "Unix Time" not in df.columns:
        raise ValueError(
            "Input data must contain either 'Mean Start Unix Time' "
            "or 'Unix Time'."
        )

    # Support both the historical column names ("thb raw ...") and the
    # GitHub-friendly Part 1 names ("thb ...").
    df.columns = df.columns.str.replace(" raw ", " ", regex=False)

    return df


def split_spacecraft_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create separate THB and THC dataframes while retaining Unix time."""
    thb_columns = [
        column
        for column in df.columns
        if column == "Unix Time" or column.lower().startswith("thb ")
    ]
    thc_columns = [
        column
        for column in df.columns
        if column == "Unix Time" or column.lower().startswith("thc ")
    ]

    thb = df[thb_columns].copy()
    thc = df[thc_columns].copy()

    return thb, thc


def _validate_columns(
    df: pd.DataFrame,
    columns: list[str],
    spacecraft: str,
) -> None:
    """Raise a clear error if a required feature is missing."""
    missing = [column for column in columns if column not in df.columns]

    if missing:
        raise ValueError(
            f"{spacecraft.upper()} is missing required feature(s): {missing}\n"
            f"Available columns: {list(df.columns)}"
        )


def fit_kmeans(
    df: pd.DataFrame,
    features: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit reproducible two-cluster K-means to selected features.

    Missing values are replaced with the feature mean, matching the original
    MSc implementation. Features are standardised before clustering.

    Returns
    -------
    raw_labels
        Arbitrary K-means cluster IDs (0/1).
    physical_centres
        Cluster centres transformed back into the original feature units.
    """
    data = df[features].copy()
    data = data.fillna(data.mean())

    if data.isna().any().any():
        raise ValueError(
            "At least one clustering feature contains only missing values."
        )

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)

    model = KMeans(
        n_clusters=2,
        init="k-means++",
        n_init=N_INIT,
        max_iter=300,
        tol=1e-4,
        random_state=RANDOM_STATE,
        algorithm="lloyd",
    )

    raw_labels = model.fit_predict(scaled_data)
    physical_centres = scaler.inverse_transform(model.cluster_centers_)

    return raw_labels, physical_centres


def clean_isolated_labels(
    labels: np.ndarray,
    window_size: int = CLEANING_WINDOW_HOURS,
) -> np.ndarray:
    """
    Correct isolated one-hour classification flips.

    A point is flipped only when every label in the preceding `window_size`
    hours and every label in the following `window_size` hours agrees with
    the opposite class.

    Example for window_size=4:
        1 1 1 1 [0] 1 1 1 1  ->  1 1 1 1 [1] 1 1 1 1
    """
    cleaned = np.asarray(labels).copy()

    for index in range(window_size, len(cleaned) - window_size):
        before = cleaned[index - window_size:index]
        after = cleaned[index + 1:index + 1 + window_size]

        if cleaned[index] == 0:
            if np.all(before == 1) and np.all(after == 1):
                cleaned[index] = 1

        elif cleaned[index] == 1:
            if np.all(before == 0) and np.all(after == 0):
                cleaned[index] = 0

    return cleaned


# ---------------------------------------------------------------------------
# First-stage clustering: outside vs inside magnetotail
# ---------------------------------------------------------------------------

def classify_magnetotail_membership(
    df: pd.DataFrame,
    spacecraft: str,
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Classify each hourly sample as outside or inside the magnetotail.

    The clustering features are plasma velocity magnitude and number density.
    The magnetotail cluster is identified physically as the cluster with the
    lower plasma number density. For this dataset, that cluster also has the
    lower velocity magnitude.

    Final first-stage labels:
        0 = outside magnetotail
        1 = inside magnetotail
    """
    features = [
        f"{spacecraft} magnitude velocity",
        f"{spacecraft} n density",
    ]
    _validate_columns(df, features, spacecraft)

    raw_labels, centres = fit_kmeans(df, features)

    density_index = features.index(f"{spacecraft} n density")
    velocity_index = features.index(f"{spacecraft} magnitude velocity")

    inside_raw_label = int(np.argmin(centres[:, density_index]))
    outside_raw_label = 1 - inside_raw_label

    # The historical/physical interpretation should also have lower plasma
    # speed inside the magnetotail. Fail loudly if that ceases to be true.
    if not (
        centres[inside_raw_label, velocity_index]
        < centres[outside_raw_label, velocity_index]
    ):
        raise ValueError(
            f"{spacecraft.upper()} cluster centres do not match the expected "
            "physical inside/outside pattern. Inspect the input data before "
            "assigning semantic labels."
        )

    semantic_labels = np.where(
        raw_labels == inside_raw_label,
        INSIDE_MAGNETOTAIL,
        OUTSIDE_MAGNETOTAIL,
    )

    cleaned_labels = clean_isolated_labels(semantic_labels)

    centre_table = pd.DataFrame(
        {
            "Region": ["Outside Magnetotail", "Inside Magnetotail"],
            "Velocity Magnitude": [
                centres[outside_raw_label, velocity_index],
                centres[inside_raw_label, velocity_index],
            ],
            "Number Density": [
                centres[outside_raw_label, density_index],
                centres[inside_raw_label, density_index],
            ],
        }
    )

    return cleaned_labels, centre_table


# ---------------------------------------------------------------------------
# Second-stage clustering: south vs north lobe
# ---------------------------------------------------------------------------

def classify_magnetotail_lobes(
    df: pd.DataFrame,
    first_stage_labels: np.ndarray,
    spacecraft: str,
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Split magnetotail samples into south and north lobes.

    Only samples classified inside the magnetotail enter this second stage.
    K-means uses magnetic-field clock and cone angles.

    The physical lobe identity is assigned from the cluster-centre cone angle:
        lower cone angle  -> South Lobe -> class 2
        higher cone angle -> North Lobe -> class 3

    Samples outside the magnetotail remain class 0.
    """
    features = [
        f"{spacecraft} clock angles",
        f"{spacecraft} cone angles",
    ]
    _validate_columns(df, features, spacecraft)

    inside_mask = first_stage_labels == INSIDE_MAGNETOTAIL
    inside_data = df.loc[inside_mask].copy()

    if inside_data.empty:
        raise ValueError(
            f"No {spacecraft.upper()} samples were classified inside the "
            "magnetotail, so lobe clustering cannot be performed."
        )

    raw_labels, centres = fit_kmeans(inside_data, features)

    cone_index = features.index(f"{spacecraft} cone angles")
    clock_index = features.index(f"{spacecraft} clock angles")

    south_raw_label = int(np.argmin(centres[:, cone_index]))
    north_raw_label = 1 - south_raw_label

    second_stage_labels = np.full(
        len(df),
        NOT_MAGNETOTAIL,
        dtype=int,
    )

    mapped_inside_labels = np.where(
        raw_labels == south_raw_label,
        SOUTH_LOBE,
        NORTH_LOBE,
    )

    second_stage_labels[np.flatnonzero(inside_mask)] = mapped_inside_labels

    centre_table = pd.DataFrame(
        {
            "Region": ["South Lobe", "North Lobe"],
            "Clock Angle": [
                centres[south_raw_label, clock_index],
                centres[north_raw_label, clock_index],
            ],
            "Cone Angle": [
                centres[south_raw_label, cone_index],
                centres[north_raw_label, cone_index],
            ],
        }
    )

    return second_stage_labels, centre_table


# ---------------------------------------------------------------------------
# Human-readable labels and complete spacecraft pipeline
# ---------------------------------------------------------------------------

def create_region_labels(
    first_stage_labels: np.ndarray,
    second_stage_labels: np.ndarray,
) -> np.ndarray:
    """
    Convert cluster classes into human-readable region labels.

    A magnetopause boundary is assigned to the first hourly sample after a
    transition between outside and inside magnetotail classifications.
    """
    labels = np.full(len(first_stage_labels), "Not MT", dtype=object)

    labels[second_stage_labels == SOUTH_LOBE] = "South Lobe"
    labels[second_stage_labels == NORTH_LOBE] = "North Lobe"

    boundary_mask = np.zeros(len(first_stage_labels), dtype=bool)
    boundary_mask[1:] = (
        first_stage_labels[1:] != first_stage_labels[:-1]
    )

    labels[boundary_mask] = "MP boundary"

    return labels


def process_spacecraft(
    df: pd.DataFrame,
    spacecraft: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run both clustering stages and attach final labels."""
    first_stage, membership_centres = classify_magnetotail_membership(
        df,
        spacecraft,
    )

    second_stage, lobe_centres = classify_magnetotail_lobes(
        df,
        first_stage,
        spacecraft,
    )

    result = df.copy()
    result["1st Cluster"] = first_stage
    result["2nd Cluster"] = second_stage
    result["Label"] = create_region_labels(first_stage, second_stage)

    return result, membership_centres, lobe_centres


def print_summary(
    spacecraft: str,
    labelled_df: pd.DataFrame,
    membership_centres: pd.DataFrame,
    lobe_centres: pd.DataFrame,
) -> None:
    """Print a concise diagnostic summary of the fitted clustering."""
    print(f"\n{'=' * 72}")
    print(f"{spacecraft.upper()} CLUSTERING SUMMARY")
    print("=" * 72)

    print("\nFirst-stage physical cluster centres")
    print(membership_centres.to_string(index=False))

    print("\nSecond-stage physical cluster centres")
    print(lobe_centres.to_string(index=False))

    print("\nFinal region counts")
    print(labelled_df["Label"].value_counts().to_string())

    print("\nNumeric class counts")
    print(
        labelled_df[["1st Cluster", "2nd Cluster"]]
        .value_counts()
        .sort_index()
        .to_string()
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run Part 2 for THB and THC and save the labelled datasets."""
    merged = load_merged_hourly_data(INPUT_PATH)
    thb, thc = split_spacecraft_data(merged)

    thb_labelled, thb_membership_centres, thb_lobe_centres = (
        process_spacecraft(thb, "thb")
    )
    thc_labelled, thc_membership_centres, thc_lobe_centres = (
        process_spacecraft(thc, "thc")
    )

    print_summary(
        "thb",
        thb_labelled,
        thb_membership_centres,
        thb_lobe_centres,
    )
    print_summary(
        "thc",
        thc_labelled,
        thc_membership_centres,
        thc_lobe_centres,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    thb_labelled.to_csv(THB_OUTPUT_PATH, index=False)
    thc_labelled.to_csv(THC_OUTPUT_PATH, index=False)

    print("\nSaved Part 2 outputs:")
    print(f"  THB: {THB_OUTPUT_PATH}")
    print(f"  THC: {THC_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
