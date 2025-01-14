import logging
from typing import Iterable

from trame.widgets import html
from trame_server.utils.namespace import Translator
from nrtk_explorer.library.filtering import FilterProtocol
from nrtk_explorer.library.dataset import get_dataset, expand_hugging_face_datasets
from nrtk_explorer.library.debounce import debounce
from nrtk_explorer.library.app_config import process_config


from nrtk_explorer.app.images.images import Images
from nrtk_explorer.app.embeddings import EmbeddingsApp
from nrtk_explorer.app.transforms import TransformsApp
from nrtk_explorer.app.filtering import FilteringApp
from nrtk_explorer.app.applet import Applet
from nrtk_explorer.app import ui
import nrtk_explorer.test_data

import os

import random

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

html.Template.slot_names.add("before")
html.Template.slot_names.add("after")


DIR_NAME = os.path.dirname(nrtk_explorer.test_data.__file__)
DEFAULT_DATASETS = [
    f"{DIR_NAME}/coco-od-2017/test_val2017.json",
]
NUM_IMAGES_DEFAULT = 500
NUM_IMAGES_DEBOUNCE_TIME = 0.3  # seconds


# ---------------------------------------------------------
# Engine class
# ---------------------------------------------------------

config_options = {
    "dataset": {
        "flags": ["--dataset"],
        "params": {
            "nargs": "+",
            "default": DEFAULT_DATASETS,
            "help": "Path to the JSON file describing the image dataset",
        },
    },
    "download": {
        "flags": ["--download"],
        "params": {
            "action": "store_true",
            "default": False,
            "help": "Download Hugging Face Hub datasets instead of streaming them",
        },
    },
}


class Engine(Applet):
    def __init__(self, server=None, **kwargs):
        super().__init__(server)

        config = process_config(self.server.cli, config_options, **kwargs)

        dataset_identifiers = expand_hugging_face_datasets(
            config["dataset"], not config["download"]
        )
        self.input_paths = dataset_identifiers
        self.state.current_dataset = self.input_paths[0]

        images = Images(server=self.server)

        self._transforms_app = TransformsApp(
            server=self.server.create_child_server(), images=images, **kwargs
        )

        self._embeddings_app = EmbeddingsApp(
            server=self.server.create_child_server(),
            images=images,
        )

        filtering_translator = Translator()
        filtering_translator.add_translation("categories", "annotation_categories")
        self._filtering_app = FilteringApp(
            server=self.server.create_child_server(translator=filtering_translator),
        )

        self._transforms_app.set_on_transform(self._embeddings_app.on_run_transformations)
        self._embeddings_app.set_on_hover(self._transforms_app.on_image_hovered)
        self._transforms_app.set_on_hover(self._embeddings_app.on_image_hovered)
        self._filtering_app.set_on_apply_filter(self.on_filter_apply)

        # Bind instance methods to controller
        self.ctrl.on_server_reload = self._build_ui
        self.ctrl.add("on_server_ready")(self.on_server_ready)

        self.state.num_images_max = 0
        self.state.num_images_disabled = True
        self.state.random_sampling = False
        self.state.random_sampling_disabled = True
        self.state.dataset_ids = []
        self.state.hovered_id = None

        def clear_hovered(**kwargs):
            self.state.hovered_id = None

        self.state.change("dataset_ids")(clear_hovered)

        self._build_ui()

    def on_server_ready(self, *args, **kwargs):
        # Bind instance methods to state change
        self.state.change("current_dataset")(self.on_dataset_change)
        self.state.change("num_images")(
            debounce(NUM_IMAGES_DEBOUNCE_TIME, self.state)(self.resample_images)
        )
        self.state.change("random_sampling")(self.resample_images)

        self.on_dataset_change()

    def on_dataset_change(self, **kwargs):
        self.state.dataset_ids = []  # sampled images
        self.context.dataset = get_dataset(self.state.current_dataset)
        self.state.num_images_max = len(self.context.dataset.imgs)
        self.state.num_images = min(self.state.num_images_max, NUM_IMAGES_DEFAULT)
        self.state.dirty("num_images")  # Trigger resample_images()
        self.state.random_sampling_disabled = False
        self.state.num_images_disabled = False

        self.state.annotation_categories = {
            category["id"]: category for category in self.context.dataset.cats.values()
        }

    def on_filter_apply(self, filter: FilterProtocol[Iterable[int]], **kwargs):
        selected_ids = []
        for dataset_id in self.state.dataset_ids:
            image_annotations_categories = [
                annotation["category_id"]
                for annotation in self.context.dataset.anns.values()
                if annotation["image_id"] == int(dataset_id)
            ]
            include = filter.evaluate(image_annotations_categories)
            if include:
                selected_ids.append(dataset_id)

        self._embeddings_app.on_select(selected_ids)

    def resample_images(self, **kwargs):
        images = list(self.context.dataset.imgs.values())

        selected_images = []
        if self.state.num_images:
            if self.state.random_sampling:
                selected_images = random.sample(images, min(len(images), self.state.num_images))
            else:
                selected_images = images[: self.state.num_images]
        else:
            selected_images = images

        self.state.dataset_ids = [str(img["id"]) for img in selected_images]
        self.state.user_selected_ids = self.state.dataset_ids

    def _build_ui(self):
        extra_args = {}
        if self.server.hot_reload:
            ui.reload(ui)
            extra_args["reload"] = self._build_ui

        self.ui = ui.NrtkExplorerLayout(
            server=self.server,
            dataset_paths=self.input_paths,
            embeddings_app=self._embeddings_app,
            filtering_app=self._filtering_app,
            transforms_app=self._transforms_app,
            **extra_args,
        )
