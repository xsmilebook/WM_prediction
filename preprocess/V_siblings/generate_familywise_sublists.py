#!/usr/bin/env python3
"""Generate ABCD family-wise subject lists for cognition and pfactor tasks."""

from pathlib import Path
from typing import List

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
TABLE_DIR = REPO_ROOT / "data" / "ABCD" / "table"
FAMILY_TABLE = TABLE_DIR / "abcd_y_lt.csv"
BASELINE_EVENT = "baseline_year_1_arm_1"

INPUT_SUBLISTS = {
    "cognition": TABLE_DIR / "cognition_sublist.txt",
    "pfactor": TABLE_DIR / "pfactor_sublist.txt",
}

OUTPUT_SUBLISTS = {
    "cognition": TABLE_DIR / "cognition_sublist_unique_family.txt",
    "pfactor": TABLE_DIR / "pfactor_sublist_unique_family.txt",
}


def to_subid(src_subject_id: str) -> str:
    return f"sub-{str(src_subject_id).replace('_', '')}"


def read_sublist(path: Path) -> List[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_family_map() -> pd.Series:
    family_df = pd.read_csv(
        FAMILY_TABLE,
        usecols=["src_subject_id", "eventname", "rel_family_id"],
        low_memory=False,
    )
    family_df = family_df.loc[family_df["eventname"] == BASELINE_EVENT, ["src_subject_id", "rel_family_id"]].copy()
    family_df["subid"] = family_df["src_subject_id"].map(to_subid)

    if family_df["rel_family_id"].isna().any():
        missing = family_df.loc[family_df["rel_family_id"].isna(), "subid"].tolist()
        raise ValueError(f"Missing rel_family_id for {len(missing)} baseline subjects")

    if family_df["subid"].duplicated().any():
        duplicated = family_df.loc[family_df["subid"].duplicated(), "subid"].tolist()
        raise ValueError(f"Duplicated baseline subjects in family table: {duplicated[:5]}")

    family_df["rel_family_id"] = family_df["rel_family_id"].astype("Int64")
    return family_df.set_index("subid")["rel_family_id"]


def keep_one_subject_per_family(subids: List[str], family_map: pd.Series) -> List[str]:
    missing = [subid for subid in subids if subid not in family_map.index]
    if missing:
        raise KeyError(f"{len(missing)} subjects missing rel_family_id mapping, first few: {missing[:5]}")

    selected = []
    seen_families = set()
    for subid in subids:
        family_id = int(family_map.loc[subid])
        if family_id in seen_families:
            continue
        seen_families.add(family_id)
        selected.append(subid)
    return selected


def write_sublist(path: Path, subids: List[str]) -> None:
    path.write_text("\n".join(subids) + "\n", encoding="utf-8")


def summarize(task_name: str, original: List[str], deduplicated: List[str], family_map: pd.Series) -> None:
    family_ids = family_map.loc[original]
    repeated_family_count = int((family_ids.value_counts() > 1).sum())
    print(
        f"{task_name}: {len(original)} -> {len(deduplicated)} subjects; "
        f"removed {len(original) - len(deduplicated)}; repeated families {repeated_family_count}"
    )


def main() -> None:
    family_map = load_family_map()

    for task_name, input_path in INPUT_SUBLISTS.items():
        original_subids = read_sublist(input_path)
        deduplicated_subids = keep_one_subject_per_family(original_subids, family_map)
        output_path = OUTPUT_SUBLISTS[task_name]
        write_sublist(output_path, deduplicated_subids)
        summarize(task_name, original_subids, deduplicated_subids, family_map)
        print(f"  wrote {output_path}")


if __name__ == "__main__":
    main()
