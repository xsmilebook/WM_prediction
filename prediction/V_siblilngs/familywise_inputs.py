from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import pandas as pd


def to_subid(src_subject_id: str) -> str:
    return f"sub-{str(src_subject_id).replace('_', '')}"


def read_sublist(path: Union[str, Path]) -> List[str]:
    return [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def build_familywise_indices(
    original_sublist_path: Union[str, Path],
    familywise_sublist_path: Union[str, Path],
) -> Tuple[List[str], np.ndarray]:
    original_subids = read_sublist(original_sublist_path)
    familywise_subids = read_sublist(familywise_sublist_path)

    if len(original_subids) != len(set(original_subids)):
        raise ValueError(f"Duplicated subjects found in {original_sublist_path}")
    if len(familywise_subids) != len(set(familywise_subids)):
        raise ValueError(f"Duplicated subjects found in {familywise_sublist_path}")

    original_positions = {subid: index for index, subid in enumerate(original_subids)}
    missing = [subid for subid in familywise_subids if subid not in original_positions]
    if missing:
        raise KeyError(f"{len(missing)} family-wise subjects missing from original sublist, first few: {missing[:5]}")

    indices = np.array([original_positions[subid] for subid in familywise_subids], dtype=int)
    return familywise_subids, indices


def load_feature_subset(
    feature_paths: List[str],
    original_sublist_path: Union[str, Path],
    familywise_sublist_path: Union[str, Path],
) -> Tuple[List[np.ndarray], List[str]]:
    familywise_subids, indices = build_familywise_indices(original_sublist_path, familywise_sublist_path)
    original_size = len(read_sublist(original_sublist_path))

    feature_arrays = []
    for feature_path in feature_paths:
        feature_array = np.load(feature_path)
        if feature_array.shape[0] != original_size:
            raise ValueError(
                f"{feature_path} row count {feature_array.shape[0]} does not match original sublist size {original_size}"
            )
        feature_arrays.append(feature_array[indices, :])
    return feature_arrays, familywise_subids


def ensure_subid_column(dataframe: pd.DataFrame) -> pd.DataFrame:
    if "subid" in dataframe.columns:
        return dataframe.copy()
    if "src_subject_id" not in dataframe.columns:
        raise KeyError("Neither subid nor src_subject_id exists in dataframe")

    dataframe = dataframe.copy()
    dataframe["subid"] = dataframe["src_subject_id"].map(to_subid)
    return dataframe


def filter_and_sort_by_subids(dataframe: pd.DataFrame, target_subids: List[str]) -> pd.DataFrame:
    dataframe = ensure_subid_column(dataframe)
    filtered = dataframe[dataframe["subid"].isin(target_subids)].copy()
    filtered["sort_order"] = filtered["subid"].map({subid: index for index, subid in enumerate(target_subids)})
    filtered = filtered.sort_values("sort_order").drop(columns="sort_order")

    missing = [subid for subid in target_subids if subid not in set(filtered["subid"])]
    if missing:
        raise KeyError(f"{len(missing)} subjects missing after filtering, first few: {missing[:5]}")
    return filtered
