
import csv
from collections import Counter
from pathlib import Path

import requests


# ============================================================
# AIRRC -> LUNA16 SUBSET MAPPING
# Does not download CT images.
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

INPUT_CSV = ROOT / "outputs" / "metrics" / "airrc_uid_manifest.csv"

OUTPUT_CSV = (
    ROOT / "outputs" / "metrics" /
    "airrc_uid_manifest_with_luna_subset.csv"
)

REPO = "MedOtter/LUNA16"

API_BASE = (
    f"https://huggingface.co/api/datasets/{REPO}/tree/main"
)

TIMEOUT = 90


def fetch_subset_uids(subset_number):

    folder = f"subset{subset_number}"

    url = f"{API_BASE}/{folder}"

    params = {
        "recursive": "false",
        "expand": "false",
        "limit": 1000,
    }

    response = requests.get(
        url,
        params=params,
        timeout=TIMEOUT
    )

    response.raise_for_status()

    entries = response.json()

    if not isinstance(entries, list):
        raise RuntimeError(
            f"Unexpected response for {folder}"
        )

    uids = set()

    for item in entries:

        name = item.get("path", "").rsplit("/", 1)[-1]

        if name.endswith(".mhd"):
            uids.add(name[:-4])

    if not uids:
        raise RuntimeError(
            f"No MHD files found in {folder}"
        )

    return uids


def main():

    if not INPUT_CSV.exists():
        raise FileNotFoundError(INPUT_CSV)

    with INPUT_CSV.open(
        "r",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        rows = list(csv.DictReader(f))

    if len(rows) != 254:
        raise RuntimeError(
            f"Expected 254 AirRC rows, found {len(rows)}"
        )

    print("=" * 65)
    print("AIRRC TO LUNA16 SUBSET MAPPING")
    print("=" * 65)

    print("AirRC annotations:", len(rows))
    print("Querying public LUNA16 file index...")
    print("No CT images will be downloaded.")

    uid_to_subset = {}

    for number in range(10):

        subset = f"subset{number}"

        print(f"\nChecking {subset}...")

        uids = fetch_subset_uids(number)

        matches = 0

        for uid in uids:

            if uid in uid_to_subset:
                raise RuntimeError(
                    f"Duplicate UID in index: {uid}"
                )

            uid_to_subset[uid] = subset

        matches = sum(
            row["series_instance_uid"] in uids
            for row in rows
        )

        print("LUNA16 scans indexed:", len(uids))
        print("AirRC matches:", matches)

    # Attach subset information without changing original CSV.
    for row in rows:

        uid = row["series_instance_uid"]

        row["luna_subset"] = uid_to_subset.get(
            uid,
            "NOT_FOUND"
        )

    counts = Counter(
        row["luna_subset"]
        for row in rows
    )

    found = sum(
        value for key, value in counts.items()
        if key != "NOT_FOUND"
    )

    missing = [
        row["series_instance_uid"]
        for row in rows
        if row["luna_subset"] == "NOT_FOUND"
    ]

    OUTPUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with OUTPUT_CSV.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys())
        )

        writer.writeheader()
        writer.writerows(rows)

    # Identify the annotation orientation outlier.
    orientation_outliers = [
        (
            row["patient_id"],
            row["series_instance_uid"],
            row["mask_orientation"],
            row["luna_subset"]
        )
        for row in rows
        if row["mask_orientation"] != "('L', 'P', 'S')"
    ]

    print("\n" + "=" * 65)
    print("MAPPING COMPLETE")
    print("=" * 65)

    print("Total AirRC scans:", len(rows))
    print("Matched to LUNA16:", found)
    print("Unmatched:", len(missing))

    print("\nAIRRC CASES PER LUNA16 SUBSET:")

    for subset in sorted(counts):

        print(f"  {subset}: {counts[subset]}")

    print("\nORIENTATION OUTLIERS:")

    for item in orientation_outliers:
        print(item)

    print("\nUNMATCHED UIDS:")
    for uid in missing[:20]:
        print(uid)

    print("\nOUTPUT CSV:")
    print(OUTPUT_CSV)

    print("\nCT IMAGES DOWNLOADED: 0")


if __name__ == "__main__":
    main()
