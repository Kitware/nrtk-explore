from nrtk_explorer.widgets.nrtk_explorer import ScatterPlot
from nrtk_explorer.library import embeddings_extractor
from nrtk_explorer.library import dimension_reducers
from nrtk_explorer.library import images_manager
from nrtk_explorer.app.applet import Applet
import nrtk_explorer.test_data

import json

from trame.widgets import quasar, html
from trame.ui.quasar import QLayout
from trame.app import get_server

import numpy as np

import os

os.environ["TRAME_DISABLE_V3_WARNING"] = "1"

DIR_NAME = os.path.dirname(nrtk_explorer.test_data.__file__)
DATASET_DIRS = [
    f"{DIR_NAME}/OIRDS_v1_0/oirds.json",
    f"{DIR_NAME}/OIRDS_v1_0/oirds_test.json",
    f"{DIR_NAME}/OIRDS_v1_0/oirds_train.json",
]


def on_click(*args, **kwargs):
    pass


class EmbeddingsApp(Applet):
    def __init__(self, server):
        super().__init__(server)

        self._ui = None
        self._on_select_fn = None
        self.reducer = dimension_reducers.DimReducerManager()
        self.is_standalone_app = self.server.state.parent is None
        if self.is_standalone_app:
            self.context["images_manager"] = images_manager.ImagesManager()

        if self.state.current_dataset is None:
            self.state.current_dataset = DATASET_DIRS[0]
        self.context["features"] = np.array([])

        self.state.tab = "PCA"
        self.state.camera_position = []
        self.state.points_sources = []
        self.state.points_transformations = []
        self.state.user_selected_points_indices = []
        self.state.current_model = "resnet50.a1_in1k"

        self.server.controller.add("on_server_ready")(self.on_server_ready)

    def on_server_ready(self, *args, **kwargs):
        # Bind instance methods to state change
        self.on_current_dataset_change()
        self.on_current_model_change()
        self.state.change("current_dataset")(self.on_current_dataset_change)
        self.state.change("current_model")(self.on_current_model_change)

    def on_current_model_change(self, **kwargs):
        current_model = self.state.current_model
        self.extractor = embeddings_extractor.EmbeddingsExtractor(
            current_model, self.context["images_manager"]
        )

    def on_current_dataset_change(self, **kwargs):
        self.state.num_elements_disabled = True
        with open(self.state.current_dataset) as f:
            dataset = json.load(f)
            self.images = dataset["images"]
            self.state.num_elements_max = len(self.images)
        self.state.num_elements_disabled = False

        if self.is_standalone_app:
            self.context["images_manager"] = images_manager.ImagesManager()

    def on_run_clicked(self):
        self.state.run_button_loading = True
        paths = list()
        for image_metadata in self.images:
            paths.append(
                os.path.join(
                    os.path.dirname(self.state.current_dataset),
                    image_metadata["file_name"],
                )
            )

        features, self.selected_paths = self.extractor.extract(
            paths=paths, n=self.state.num_elements, rand=self.state.random_sampling
        )
        if self.state.tab == "PCA":
            self.state.points_sources = self.reducer.reduce(
                name="PCA",
                fit_features=features,
                dims=self.state.dimensionality,
                whiten=self.state.pca_whiten,
                solver=self.state.pca_solver,
            )

        elif self.state.tab == "UMAP":
            self.state.points_sources = self.reducer.reduce(
                name="UMAP",
                fit_features=features,
                dims=self.state.dimensionality,
            )

        self.context["features"] = features
        self.state.run_button_loading = False

    def on_run_transformations(self, transformed_image_ids):
        transformation_features, _ = self.extractor.extract(
            paths=transformed_image_ids,
            cache=False,
            content=self.context["image_objects"],
        )

        if self.state.tab == "PCA":
            self.state.points_transformations = self.reducer.reduce(
                name="PCA",
                fit_features=self.context["features"],
                features=transformation_features,
                dims=self.state.dimensionality,
                whiten=self.state.pca_whiten,
                solver=self.state.pca_solver,
            )

        elif self.state.tab == "UMAP":
            self.state.points_transformations = self.reducer.reduce(
                name="UMAP",
                fit_features=self.context["features"],
                features=transformation_features,
                dims=self.state.dimensionality,
            )

    def set_on_select(self, fn):
        self._on_select_fn = fn

    def on_select(self, ids):
        self.state.user_selected_points_indices = ids
        if self._on_select_fn:
            self._on_select_fn(ids)

    def on_move(self, camera_position):
        self.state.camera_position = camera_position

    def visualization_widget(self):
        ScatterPlot(
            cameraMove=(self.on_move, "[$event]"),
            cameraPosition=("get('camera_position')",),
            click=(on_click, "[$event]"),
            points=("get('points_sources')",),
            select=(self.on_select, "[$event]"),
            userSelectedPoints=("get('user_selected_points_indices')",),
        )

    def visualization_widget_transformation(self):
        ScatterPlot(
            cameraMove=(self.on_move, "[$event]"),
            cameraPosition=("get('camera_position')",),
            click=(on_click, "[$event]"),
            plotTransformations=("true",),
            points=("get('points_transformations')",),
            select=(self.on_select, "[$event]"),
            userSelectedPoints=("get('user_selected_points_indices')",),
        )

    def settings_widget(self):
        with html.Div(
            trame_server=self.server, classes="column justify-center", style="padding:1rem"
        ):
            with html.Div(classes="col"):
                quasar.QSelect(
                    label="Embeddings Model",
                    v_model=("current_model",),
                    options=(
                        [
                            {"label": "ResNet50", "value": "resnet50.a1_in1k"},
                            {"label": "EfficientNet_b0", "value": "efficientnet_b0.ra_in1k"},
                            {
                                "label": "MobileNetV3Large",
                                "value": "mobilenetv3_large_100.ra_in1k",
                            },
                        ],
                    ),
                    filled=True,
                    emit_value=True,
                    map_options=True,
                )
                quasar.QSeparator(inset=True)
                html.P("Number of elements:", classes="text-body2")
                quasar.QSlider(
                    v_model=("num_elements", 15),
                    min=(0,),
                    max=("num_elements_max", 25),
                    disable=("num_elements_disabled", True),
                    step=(1,),
                    label=True,
                    label_always=True,
                )
                quasar.QToggle(
                    v_model=("random_sampling", False),
                    label="Random selection",
                    left_label=True,
                )

                html.P("Dimensionality:", classes="text-body2")
                with html.Div(classes="q-gutter-y-md"):
                    quasar.QBtnToggle(
                        v_model=("dimensionality", "3"),
                        toggler_color="primary",
                        flat=True,
                        spread=True,
                        options=(
                            [
                                {"label": "2D", "value": "2"},
                                {"label": "3D", "value": "3"},
                            ],
                        ),
                    )
            with html.Div(classes="col"):
                html.P("Method:", classes="text-body2")
                with quasar.QTabs(
                    v_model="tab",
                    dense=True,
                    narrow_indicator=True,
                    active_color="primary",
                    indicator_color="primary",
                    align="justify",
                ):
                    quasar.QTab(name="PCA", label="pca")
                    quasar.QTab(name="UMAP", label="umap")
                quasar.QSeparator()
                with quasar.QTabPanels(v_model="tab"):
                    with quasar.QTabPanel(name="PCA"):
                        quasar.QToggle(
                            v_model=("pca_whiten", False),
                            label="Whiten",
                            left_label=True,
                        )
                        quasar.QSelect(
                            v_model=("pca_solver", "auto"),
                            label="SVD Solver",
                            toggler_color="primary",
                            options=(
                                [
                                    "auto",
                                    "full",
                                    "arpack",
                                    "randomized",
                                ],
                            ),
                        )

                    with quasar.QTabPanel(name="UMAP"):
                        with html.Div(classes="row"):
                            html.Div(classes="col-md-auto")

                quasar.QBtn(
                    label="Run",
                    color="red",
                    loading=("run_button_loading", False),
                    click=self.on_run_clicked,
                )

    # This is only used within when this module (file) is executed as an Standalone app.
    @property
    def ui(self):
        if self._ui is None:
            with QLayout(self.server) as layout:
                with quasar.QHeader():
                    with quasar.QToolbar(classes="shadow-4"):
                        quasar.QToolbarTitle("Embeddings")
                        quasar.QBtn("Reset")

                with quasar.QDrawer(
                    v_model=("leftDrawerOpen", True),
                    side="left",
                    elevated=True,
                ):
                    with html.Div(classes="column justify-center", style="padding:1rem"):
                        with html.Div(classes="col"):
                            quasar.QSeparator()
                            quasar.QSelect(
                                label="Dataset",
                                v_model=("current_dataset",),
                                options=(
                                    [
                                        {"label": "oirds_test", "value": DATASET_DIRS[0]},
                                        {"label": "oirds_train", "value": DATASET_DIRS[1]},
                                    ],
                                ),
                                filled=True,
                                emit_value=True,
                                map_options=True,
                            )
                    self.settings_widget()

                # Main content
                with quasar.QPageContainer():
                    with quasar.QPage():
                        with html.Div(classes="row", style="min-height: inherit;"):
                            with html.Div(classes="col q-pa-md"):
                                self.visualization_widget()

                self._ui = layout
        return self._ui


def embeddings(server=None, *args, **kwargs):
    server = get_server()
    server.client_type = "vue3"

    embeddings_app = EmbeddingsApp(server)
    embeddings_app.ui

    server.start(**kwargs)


if __name__ == "__main__":
    embeddings()