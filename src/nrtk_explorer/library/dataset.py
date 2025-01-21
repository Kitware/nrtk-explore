"""
Module to load a dataset.

Example:
    dataset = get_dataset("path/to/dataset.json")
"""

from typing import Sequence as SequenceType, Union
from abc import ABC, abstractmethod
import os
from functools import lru_cache
from pathlib import Path
from PIL import Image
import kwcoco
from datasets import (
    load_dataset,
    get_dataset_infos,
    Sequence as SequenceDataset,
    ClassLabel,
    Image as DatasetImage,
)

HF_ROWS_TO_TAKE_STREAMING = 300


class BaseDataset(ABC):
    @abstractmethod
    def get_image(self, id: int):
        """Get the image given an image id."""
        pass


class CategoryIndex:
    def build_cat_index(self):
        # follows kwcoco.name_to_cat
        self.name_to_cat = {cat["name"]: cat for cat in self.cats.values()}


class CocoDataset(kwcoco.CocoDataset, BaseDataset):
    def get_image(self, id: int):
        image_fpath = self.get_image_fpath(id)
        return Image.open(image_fpath)


def is_coco_dataset(path: str):
    if not os.path.exists(path) or os.path.isdir(path):
        return False
    required_keys = ['"images"', '"categories"', '"annotations"']
    with open(path) as f:
        content = f.read()
    return all(key in content for key in required_keys)

def discover_datasets(repository: Union[Path, None]) -> list[Path]:
    datasets: list[Path] = []

    if repository is None:
        return datasets

    for item in repository.iterdir():
        if item.is_dir():
            for file in item.iterdir():
                if file.is_file() and file.suffix == ".json" and is_coco_dataset(str(file)):
                    datasets.append(file)

    return datasets


def dataset_select_options(datasets: list[str]):
    return [{"label": Path(ds).name, "value": ds} for ds in datasets]


def expand_hugging_face_datasets(dataset_identifiers: SequenceType[str], streaming=True):
    expanded_identifiers = []
    for identifier in dataset_identifiers:
        if is_coco_dataset(identifier):
            expanded_identifiers.append(identifier)
        else:
            infos = get_dataset_infos(identifier)
            for config_name, info in infos.items():
                for split_name in info.splits:
                    streaming_str = "streaming" if streaming else "download"
                    expanded_identifiers.append(
                        f"{identifier}@{config_name}@{split_name}@{streaming_str}"
                    )
    return expanded_identifiers


def find_column_name(features, column_names):
    return next((key for key in column_names if key in features), None)


class HuggingFaceDataset(BaseDataset, CategoryIndex):
    """Interface for Hugging Face datasets with a similar API to CocoDataset."""

    def __init__(self, identifier: str):
        self.imgs: dict[str, dict] = {}
        self.anns: dict[str, dict] = {}
        self.cats: dict[str, dict] = {}
        self._id_to_row_idx: dict[str, int] = {}

        repo, config, split, streaming = identifier.split("@")
        self._streaming = streaming == "streaming"
        self._dataset = load_dataset(repo, config, split=split, streaming=self._streaming)
        if self._streaming:
            self._dataset = self._dataset.take(HF_ROWS_TO_TAKE_STREAMING)
        self._load_data()
        self.build_cat_index()

    def _load_data(self):
        image_key = find_column_name(self._dataset.features, ["image", "img"])
        self._image_key = image_key
        # transforms and base64 encoding require RGB mode
        self._dataset.cast_column(image_key, DatasetImage(mode="RGB"))

        counter = 0

        def make_id():
            nonlocal counter
            counter += 1
            return f"ann_{counter}"

        def extract_labels(feature):
            if isinstance(feature, ClassLabel):
                return feature.names
            if hasattr(feature, "names"):
                return feature.names
            if isinstance(feature, SequenceDataset):
                return extract_labels(feature.feature)
            if isinstance(feature, list) and len(feature) >= 1:
                return extract_labels(feature[0])
            if isinstance(feature, dict):
                for key in ["category", "category_id", "label", "labels", "objects"]:
                    if key in feature:
                        labels = extract_labels(feature[key])
                        if labels:
                            return labels
            return None

        labels = extract_labels(self._dataset.features)
        if labels:
            self.cats = {i: {"id": i, "name": str(name)} for i, name in enumerate(labels)}

        objects_key = find_column_name(self._dataset.features, ["objects"])

        classifications_key = find_column_name(
            self._dataset.features,
            [
                "labels",
                "label",
                "classifications",
            ],
        )

        new_cats = set()
        # speed initial metadata process by not loading images if we can random access rows (not streaming)
        maybe_no_image = (
            self._dataset if self._streaming else self._dataset.remove_columns([image_key])
        )
        for idx, example in enumerate(maybe_no_image):
            id = example.get("id", example.get("image_id", idx))
            if self._streaming:
                self.imgs[id] = {"id": id, "image": example[image_key]}
            else:
                self.imgs[id] = {"id": id}
                self._id_to_row_idx[id] = idx

            if objects_key:
                objects = example[objects_key]
                if isinstance(objects, list):
                    # Convert list of dicts to dict of lists. We want columns, not rows.
                    cat_keys = ["category", "category_id", "label"]
                    cat_key = next(
                        (key for key in cat_keys if objects and key in objects[0]), cat_keys[0]
                    )
                    keys = ["id", "bbox", cat_key]
                    objects = {key: [obj.get(key) for obj in objects] for key in keys}
                ids = objects.get("id") or [make_id() for _ in range(len(objects.get("bbox", [])))]
                categories = (
                    objects.get("category")
                    or objects.get("category_id")
                    or objects.get("label", [])
                )
                for ann_id, bbox, cat_id in zip(ids, objects.get("bbox", []), categories):
                    if cat_id not in self.cats:
                        new_cats.add(cat_id)
                    self.anns[ann_id] = {
                        "id": ann_id,
                        "image_id": id,
                        "category_id": cat_id,
                        "bbox": bbox,
                    }

            if classifications_key:
                classes = example[classifications_key]
                if not isinstance(classes, list):
                    classes = [classes]
                for cat_id in classes:
                    if cat_id not in self.cats:
                        new_cats.add(cat_id)
                    ann_id = make_id()
                    self.anns[ann_id] = {
                        "id": ann_id,
                        "image_id": id,
                        "category_id": cat_id,
                    }

        if new_cats:
            max_existing_id = max(self.cats.keys(), default=0)
            for new_cat in new_cats:
                max_existing_id += 1
                self.cats[max_existing_id] = {"id": max_existing_id, "name": new_cat}
            name_to_id = {cat["name"]: cat["id"] for cat in self.cats.values()}
            for annotation in self.anns.values():
                annotation["category_id"] = name_to_id[annotation["category_id"]]

    def get_image(self, id):
        if self._streaming:
            return self.imgs[id]["image"]
        else:
            row_idx = self._id_to_row_idx[id]
            return self._dataset[row_idx][self._image_key]


@lru_cache
def get_dataset(identifier: str):
    """Get the dataset object.
    Args:
        identifier (str): Path to the dataset file or HuggingFace hub dataset identifier.
    Return:
        dataset: Dataset object.
    """
    absolute_path = str(Path(identifier).resolve())

    if is_coco_dataset(absolute_path):
        return CocoDataset(absolute_path)

    # Assume identifier is a Hugging Face Dataset
    return HuggingFaceDataset(identifier)
