import logging
from typing import Dict, Callable
from collections.abc import Mapping

from trame.ui.quasar import QLayout
from trame.widgets import quasar
from trame.widgets import html
from trame.app import get_server, asynchronous
from trame_server import Server

import nrtk_explorer.library.transforms as trans
import nrtk_explorer.library.yaml_transforms as nrtk_yaml
from nrtk_explorer.library.multiprocess_predictor import MultiprocessPredictor
from nrtk_explorer.library.app_config import process_config
from nrtk_explorer.library.scoring import (
    compute_score,
)

from nrtk_explorer.app.applet import Applet
from nrtk_explorer.app.parameters import ParametersApp
from nrtk_explorer.app.images.image_meta import update_image_meta, dataset_id_to_meta
from nrtk_explorer.app.trame_utils import change_checker, delete_state
from nrtk_explorer.app.images.image_ids import (
    dataset_id_to_image_id,
    dataset_id_to_transformed_image_id,
)
from nrtk_explorer.app.images.images import Images
from nrtk_explorer.app.images.stateful_annotations import (
    make_stateful_annotations,
    make_stateful_predictor,
)
from nrtk_explorer.app.ui import ImageList
from nrtk_explorer.app.ui.image_list import (
    TRANSFORM_COLUMNS,
    ORIGINAL_COLUMNS,
    init_visible_columns,
)


INFERENCE_MODELS_DEFAULT = [
    "facebook/detr-resnet-50",
    "hustvl/yolos-tiny",
    "valentinafeve/yolos-fashionpedia",
]


UPDATE_IMAGES_CHUNK_SIZE = 32

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class LazyDict(Mapping):
    """If function provided for value, run function when value is accessed"""

    def __init__(self, *args, **kw):
        self._raw_dict = dict(*args, **kw)

    def __getitem__(self, key):
        val = self._raw_dict[key]
        return val() if callable(val) else val

    def __setitem__(self, key, value):
        self._raw_dict[key] = value

    def __iter__(self):
        return iter(self._raw_dict)

    def __len__(self):
        return len(self._raw_dict)

    def values(self):
        return (self[k] for k in self._raw_dict)

    def items(self):
        return ((k, self[k]) for k in self._raw_dict)


class ProcessingStep:
    def __init__(
        self,
        server: Server,
        feature_enabled_state_key: str,
        gui_switch_key: str,
        column_name: str,
        enabled_callback: Callable,
    ):
        self.state = server.state
        self.feature_enabled_state_key = feature_enabled_state_key
        self.gui_switch_key = gui_switch_key
        self.enabled_callback = enabled_callback
        self.column_name = column_name
        self.state.change(self.gui_switch_key)(self.on_gui_switch)
        self.update_feature_enabled_state()
        self.state.change("visible_columns", self.gui_switch_key)(
            self.update_feature_enabled_state
        )
        self.state.change(self.feature_enabled_state_key)(self.on_change_feature_enabled)

    def on_gui_switch(self, **kwargs):
        if self.state[self.gui_switch_key]:
            self.state.visible_columns = list(set([*self.state.visible_columns, self.column_name]))
        else:
            self.state.visible_columns = [
                col for col in self.state.visible_columns if col != self.column_name
            ]

    def update_feature_enabled_state(self, **kwargs):
        self.state[self.feature_enabled_state_key] = (
            self.column_name in self.state.visible_columns and self.state[self.gui_switch_key]
        )

    def on_change_feature_enabled(self, **kwargs):
        if self.state[self.feature_enabled_state_key]:
            self.enabled_callback()


config_options = {
    "models": {
        "flags": ["--models"],
        "params": {
            "nargs": "+",
            "default": INFERENCE_MODELS_DEFAULT,
            "help": "Space separated list of inference models",
        },
    },
}


class TransformsApp(Applet):
    def __init__(
        self,
        server,
        images=None,
        ground_truth_annotations=None,
        original_detection_annotations=None,
        transformed_detection_annotations=None,
        **kwargs,
    ):
        super().__init__(server)

        config = process_config(self.server.cli, config_options, **kwargs)
        self.state.inference_models = config["models"]
        self.state.inference_model = self.state.inference_models[0]
        self.state.setdefault("image_list_ids", [])
        self.state.setdefault("dataset_ids", [])
        self.state.setdefault("user_selected_ids", [])

        self.images = images or Images(server)

        ground_truth_annotations = ground_truth_annotations or make_stateful_annotations(server)
        self.ground_truth_annotations = ground_truth_annotations.annotations_factory

        original_detection_annotations = original_detection_annotations or make_stateful_predictor(
            server
        )
        self.original_detection_annotations = original_detection_annotations.annotations_factory

        transformed_detection_annotations = (
            transformed_detection_annotations or make_stateful_predictor(server)
        )
        self.transformed_detection_annotations = (
            transformed_detection_annotations.annotations_factory
        )

        def clear_transformed(**kwargs):
            self.transformed_detection_annotations.cache_clear()
            for id in self.state.dataset_ids:
                update_image_meta(
                    self.state,
                    id,
                    {
                        "original_detection_to_transformed_detection_score": 0,
                        "ground_truth_to_transformed_detection_score": 0,
                    },
                )

        server.controller.apply_transform.add(clear_transformed)

        # delete score from state of old ids that are not in new
        def delete_meta_state(old_ids, new_ids):
            if old_ids is not None:
                to_clean = set(old_ids) - set(new_ids)
                for id in to_clean:
                    delete_state(self.state, dataset_id_to_meta(id))

        change_checker(self.state, "dataset_ids")(delete_meta_state)
        # clear score when changing model
        # clear score when changing transform

        self._parameters_app = ParametersApp(
            server=server,
        )

        self._ui = None

        self._on_transform_fn = None

        self._transform_classes: Dict[str, type[trans.ImageTransform]] = {
            "blur": trans.GaussianBlurTransform,
            "invert": trans.InvertTransform,
            "downsample": trans.DownSampleTransform,
            "identity": trans.IdentityTransform,
        }

        # Add transform from YAML definition
        self._transform_classes.update(nrtk_yaml.generate_transforms())

        self._parameters_app._transform_classes = self._transform_classes

        # Initialize the transforms pipeline to the identity
        self._parameters_app._default_transform = "blur"
        self._parameters_app.on_add_transform()

        init_visible_columns(self.state)

        # On annotations enabled, run whole pipeline to possibly compute transforms. Why? Transforms compute scores are based on original images
        self.annotations_enable_control = ProcessingStep(
            server,
            feature_enabled_state_key="predictions_original_images_enabled",
            gui_switch_key="annotations_enabled_switch",
            column_name=ORIGINAL_COLUMNS[0],
            enabled_callback=self._start_update_images,
        )

        self.transform_enable_control = ProcessingStep(
            server,
            feature_enabled_state_key="transform_enabled",
            gui_switch_key="transform_enabled_switch",
            column_name=TRANSFORM_COLUMNS[0],
            enabled_callback=self._start_update_images,
        )

        self.server.controller.on_server_ready.add(self.on_server_ready)
        self.server.controller.apply_transform.add(self.on_apply_transform)
        self._on_hover_fn = None

        self.visible_dataset_ids = []  # set by ImageList via self.on_scroll callback

        self.predictor = MultiprocessPredictor(model_name=self.state.inference_model)

    def on_server_ready(self, *args, **kwargs):
        self.state.change("inference_model")(self.on_inference_model_change)
        self.state.change("current_dataset")(self._cancel_update_images)
        self.state.change("current_dataset")(self.reset_predictor)
        self.state.change("confidence_score_threshold")(self._start_update_images)

    def on_inference_model_change(self, **kwargs):
        self.original_detection_annotations.cache_clear()
        self.transformed_detection_annotations.cache_clear()
        self.predictor.set_model(self.state.inference_model)
        self._start_update_images()

    def reset_predictor(self, **kwargs):
        self.predictor.reset()

    def set_on_transform(self, fn):
        self._on_transform_fn = fn

    def on_transform(self, *args, **kwargs):
        if self._on_transform_fn:
            self._on_transform_fn(*args, **kwargs)

    def on_apply_transform(self, **kwargs):
        # Turn on switch if user clicked lower apply button
        self.state.transform_enabled_switch = True
        self._start_update_images()

    async def update_transformed_images(
        self, dataset_ids, predictions_original_images, visible=False
    ):
        if not self.state.transform_enabled:
            return

        transforms = list(map(lambda t: t["instance"], self.context.transforms))
        transform = trans.ChainedImageTransform(transforms)

        id_to_image = LazyDict()
        for id in dataset_ids:
            if visible:
                with self.state:
                    transformed = self.images.get_stateful_transformed_image(transform, id)
                    id_to_image[dataset_id_to_transformed_image_id(id)] = transformed
                await self.server.network_completion
            else:
                id_to_image[dataset_id_to_transformed_image_id(id)] = (
                    lambda id=id: self.images.get_transformed_image_without_cache_eviction(
                        transform, id
                    )
                )

        with self.state:
            annotations = await self.transformed_detection_annotations.get_annotations(
                self.predictor, id_to_image
            )
        await self.server.network_completion

        ground_truth_annotations = self.ground_truth_annotations.get_annotations(dataset_ids)
        scores = compute_score(
            self.context.dataset,
            ground_truth_annotations,
            annotations,
            self.state.confidence_score_threshold,
        )
        for id, score in scores:
            update_image_meta(
                self.state, id, {"ground_truth_to_transformed_detection_score": score}
            )

        # depends on original images predictions
        if predictions_original_images:
            scores = compute_score(
                self.context.dataset,
                predictions_original_images,
                annotations,
                self.state.confidence_score_threshold,
            )
            for id, score in scores:
                update_image_meta(
                    self.state,
                    id,
                    {"original_detection_to_transformed_detection_score": score},
                )

        self.state.flush()  # needed cuz in async func and modifying state or else UI does not update
        # sortable score value may have changed which may have changed images that are in view
        self.server.controller.check_images_in_view()

        self.on_transform(id_to_image)  # inform embeddings app
        self.state.flush()

    async def compute_predictions_original_images(self, dataset_ids):
        if not self.state.predictions_original_images_enabled:
            return

        image_id_to_image = LazyDict(
            {
                dataset_id_to_image_id(
                    id
                ): lambda id=id: self.images.get_image_without_cache_eviction(id)
                for id in dataset_ids
            }
        )

        predictions_original_images = await self.original_detection_annotations.get_annotations(
            self.predictor, image_id_to_image
        )

        ground_truth_annotations = self.ground_truth_annotations.get_annotations(dataset_ids)

        scores = compute_score(
            self.context.dataset,
            ground_truth_annotations,
            predictions_original_images,
            self.state.confidence_score_threshold,
        )
        for dataset_id, score in scores:
            update_image_meta(
                self.state, dataset_id, {"original_ground_to_original_detection_score": score}
            )

        return predictions_original_images

    async def _update_images(self, dataset_ids, visible=False):
        if visible:
            # load images on state for ImageList
            with self.state:
                for id in dataset_ids:
                    self.images.get_stateful_image(id)
                self.ground_truth_annotations.get_annotations(dataset_ids)
            await self.server.network_completion

        # always push to state because compute_predictions_original_images updates score metadata
        with self.state:
            predictions_original_images = await self.compute_predictions_original_images(
                dataset_ids
            )
        await self.server.network_completion
        # sortable score value may have changed which may have changed images that are in view
        self.server.controller.check_images_in_view()

        await self.update_transformed_images(
            dataset_ids, predictions_original_images, visible=visible
        )

    async def _chunk_update_images(self, dataset_ids, visible=False):
        ids = list(dataset_ids)
        for i in range(0, len(ids), UPDATE_IMAGES_CHUNK_SIZE):
            chunk = ids[i : i + UPDATE_IMAGES_CHUNK_SIZE]
            await self._update_images(chunk, visible=visible)

    async def _update_all_images(self, visible_images):
        with self.state:
            self.state.updating_images = True

        await self._chunk_update_images(visible_images, visible=True)

        other_images = set(self.state.user_selected_ids) - set(visible_images)
        await self._chunk_update_images(other_images, visible=False)

        with self.state:
            self.state.updating_images = False

    def _cancel_update_images(self, **kwargs):
        if hasattr(self, "_update_task"):
            self._update_task.cancel()

    def _start_update_images(self, **kwargs):
        """
        After updating the images visible in the image list, all other selected
        images are updated and their scores computed. After images are scored,
        the table sort may have changed the images that are visible, so
        ImageList is asked to send visible image IDs again, which may trigger
        a new _update_all_images if the set of images in view has changed.
        """
        self._cancel_update_images()
        self._update_task = asynchronous.create_task(
            self._update_all_images(self.visible_dataset_ids)
        )

    def on_scroll(self, visible_ids):
        self.visible_dataset_ids = visible_ids
        self._start_update_images()

    def on_image_hovered(self, id):
        self.state.hovered_id = id

    def set_on_hover(self, fn):
        self._on_hover_fn = fn

    def on_hover(self, hover_event):
        id_ = hover_event["id"]
        self.on_image_hovered(id_)
        if self._on_hover_fn:
            self._on_hover_fn(id_)

    def settings_widget(self):
        with html.Div(classes="col"):
            self._parameters_app.transforms_ui()

    def apply_ui(self):
        with html.Div():
            self._parameters_app.transform_apply_ui()

    def dataset_widget(self):
        ImageList(self.on_scroll, self.on_hover)

    # This is only used within when this module (file) is executed as an Standalone app.
    @property
    def ui(self):
        if self._ui is None:
            with QLayout(
                self.server, view="lhh LpR lff", classes="shadow-2 rounded-borders bg-grey-2"
            ) as layout:
                # # Toolbar
                with quasar.QHeader():
                    with quasar.QToolbar(classes="shadow-4"):
                        quasar.QBtn(
                            flat=True,
                            click="drawerLeft = !drawerLeft",
                            round=True,
                            dense=False,
                            icon="menu",
                        )
                        quasar.QToolbarTitle("Transforms")

                # # Main content
                with quasar.QPageContainer():
                    with quasar.QPage():
                        with html.Div(classes="row"):
                            with html.Div(classes="col-2 q-pa-md"):
                                with html.Div(
                                    classes="column justify-center", style="padding:1rem"
                                ):
                                    with html.Div(classes="col"):
                                        quasar.QSelect(
                                            label="Dataset",
                                            v_model=("current_dataset",),
                                            options=([],),
                                            filled=True,
                                            emit_value=True,
                                            map_options=True,
                                        )

                                        html.P("Number of elements:", classes="text-body2")
                                        quasar.QSlider(
                                            v_model=("current_num_elements",),
                                            min=(0,),
                                            max=(25,),
                                            step=(1,),
                                            label=True,
                                            label_always=True,
                                        )
                                self.settings_widget()
                                self.apply_ui()

                            self.dataset_widget()

                self._ui = layout
        return self._ui


def main(server=None, *args, **kwargs):
    server = get_server(client_type="vue3")

    transforms_app = TransformsApp(server)
    transforms_app.ui

    server.start(**kwargs)


if __name__ == "__main__":
    main()
