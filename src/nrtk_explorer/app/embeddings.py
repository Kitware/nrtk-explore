from nrtk_explorer.widgets.nrtk_explorer import ScatterPlot
from nrtk_explorer.library import embeddings_extractor
from nrtk_explorer.library import dimension_reducers
from nrtk_explorer.library.dataset import get_dataset
from nrtk_explorer.app.applet import Applet

from nrtk_explorer.app.images.image_ids import (
    image_id_to_dataset_id,
    dataset_id_to_transformed_image_id,
    dataset_id_to_image_id,
    is_transformed,
)
from nrtk_explorer.app.images.images import Images

from pathlib import Path

from trame.widgets import quasar, html
from trame.ui.quasar import QLayout
from trame.app import get_server, asynchronous


class EmbeddingsApp(Applet):
    def __init__(
        self,
        server,
        datasets=None,
        images=None,
    ):
        super().__init__(server)

        self._dataset_paths = datasets
        self.images = images or Images(server)
        self._on_hover_fn = None
        self._ui = None
        self.reducer = dimension_reducers.DimReducerManager()

        # Local initialization if standalone
        self.is_standalone_app = self.server.root_server == self.server
        if self.is_standalone_app and datasets:
            self.state.dataset_ids = []
            self.state.current_dataset = datasets[0]
            self.on_current_dataset_change()

        self.features = None

        self.state.client_only("camera_position")
        self.state.feature_extraction_model = "resnet50.a1_in1k"

        self.server.controller.add("on_server_ready")(self.on_server_ready)
        self.transformed_images_cache = {}
        self.state.highlighted_image = {
            "id": "",
            "is_transformed": True,
        }

    def on_server_ready(self, *args, **kwargs):
        # Bind instance methods to state change
        self.on_current_dataset_change()
        self.state.change("current_dataset")(self.on_current_dataset_change)

        self.on_feature_extraction_model_change()
        self.state.change("feature_extraction_model")(self.on_feature_extraction_model_change)

        self.update_points()
        self.state.change("dataset_ids")(self.update_points)

        self.server.controller.apply_transform.add(self.clear_points_transformations)
        self.state.change("transform_enabled_switch")(
            self.update_points_transformations_visibility
        )

    def on_feature_extraction_model_change(self, **kwargs):
        feature_extraction_model = self.state.feature_extraction_model
        self.extractor = embeddings_extractor.EmbeddingsExtractor(
            model_name=feature_extraction_model
        )

    def on_current_dataset_change(self, **kwargs):
        self.state.num_elements_disabled = True
        if self.context.dataset is None:
            self.context.dataset = get_dataset(self.state.current_dataset)

        self.state.num_elements_max = len(list(self.context.dataset.imgs))
        self.state.num_elements_disabled = False

    def compute_points(self, fit_features, features):
        if len(features) == 0:
            # reduce will fail if no features
            return []

        if self.state.tab == "PCA":
            return self.reducer.reduce(
                name="PCA",
                fit_features=fit_features,
                features=features,
                dims=self.state.dimensionality,
                whiten=self.state.pca_whiten,
                solver=self.state.pca_solver,
            )

        # must be UMAP
        args = {}
        if self.state.umap_random_seed:
            args["random_state"] = int(self.state.umap_random_seed_value)

        if self.state.umap_n_neighbors:
            args["n_neighbors"] = int(self.state.umap_n_neighbors_number)

        return self.reducer.reduce(
            name="UMAP",
            fit_features=fit_features,
            features=features,
            dims=self.state.dimensionality,
            **args,
        )

    def clear_points_transformations(self, **kwargs):
        self.state.points_transformations = {}  # ID to point
        self._stashed_points_transformations = {}

    def update_points_transformations_visibility(self, **kwargs):
        if self.state.transform_enabled_switch:
            self.state.points_transformations = self._stashed_points_transformations
        else:
            self._stashed_points_transformations = self.state.points_transformations
            self.state.points_transformations = {}

    async def compute_source_points(self):
        with self.state:
            self.state.is_loading = True

        # Don't lock server before enabling the spinner on client
        await self.server.network_completion

        images = [
            self.images.get_image_without_cache_eviction(id) for id in self.state.dataset_ids
        ]
        self.features = self.extractor.extract(
            images,
            batch_size=int(self.state.model_batch_size),
        )

        points = self.compute_points(self.features, self.features)

        self.state.points_sources = {
            id: point for id, point in zip(self.state.dataset_ids, points)
        }

        self.clear_points_transformations()

        self.state.camera_position = []

        with self.state:
            self.state.is_loading = False

    def update_points(self, **kwargs):
        if hasattr(self, "_update_task"):
            self._update_task.cancel()
        self._update_task = asynchronous.create_task(self.compute_source_points())

    def on_run_clicked(self):
        self.update_points()

    def on_run_transformations(self, id_to_image):
        transformation_features = self.extractor.extract(
            id_to_image.values(),
            batch_size=int(self.state.model_batch_size),
        )

        points = self.compute_points(self.features, transformation_features)

        ids = id_to_image.keys()
        updated_points = {image_id_to_dataset_id(id): point for id, point in zip(ids, points)}
        self.state.points_transformations = {**self.state.points_transformations, **updated_points}

    # called by category filter
    def on_select(self, image_ids):
        self.state.user_selected_ids = image_ids

    def on_scatter_select(self, image_ids):
        self.state.user_selected_ids = image_ids or self.state.dataset_ids

    def on_move(self, camera_position):
        self.state.camera_position = camera_position

    def set_on_hover(self, fn):
        self._on_hover_fn = fn

    def get_dataset_id_index(self, point_index):
        if point_index < len(self.state.dataset_ids):
            return point_index
        return point_index - len(self.state.dataset_ids)

    def on_point_hover(self, event):
        self.state.highlighted_image = event
        if not self._on_hover_fn:
            return
        if event["is_transformed"]:
            image_id = dataset_id_to_transformed_image_id(event["id"])
        else:
            image_id = dataset_id_to_image_id(event["id"])
        self._on_hover_fn(image_id)

    def on_image_hovered(self, image_id):
        self.state.highlighted_image = {
            "id": image_id_to_dataset_id(image_id),
            "is_transformed": is_transformed(image_id),
        }

    def visualization_widget(self):
        ScatterPlot(
            cameraMove="camera_position=$event",
            cameraPosition=("camera_position",),
            highlightedPoint=("highlighted_image",),
            hover=(self.on_point_hover, "[$event]"),
            points=("points_sources", {}),
            transformedPoints=("points_transformations", {}),
            select=(self.on_scatter_select, "[$event]"),
            selectedPoints=("user_selected_ids", []),
        )

    def settings_widget(self):
        with html.Div(trame_server=self.server, classes="col"):
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

            quasar.QSelect(
                label="Embeddings Model",
                v_model=("feature_extraction_model",),
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
            quasar.QInput(
                v_model=("model_batch_size", 32),
                filled=True,
                stack_label=True,
                label="Batch Size",
                type="number",
            )

        with html.Div(classes="col"):
            with quasar.QTabs(
                v_model=("tab", "PCA"),
                dense=True,
                narrow_indicator=True,
                active_color="primary",
                indicator_color="primary",
                align="justify",
            ):
                quasar.QTab(name="PCA", label="pca")
                quasar.QTab(name="UMAP", label="umap")
            quasar.QSeparator()
            with quasar.QTabPanels(v_model=("tab", "PCA")):
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
                    quasar.QToggle(
                        v_model=("umap_n_neighbors", False),
                        label="Number of neighbors",
                        left_label=True,
                    )
                    quasar.QInput(
                        v_model=("umap_n_neighbors_number", 15),
                        disable=("!umap_n_neighbors",),
                        filled=True,
                        stack_label=True,
                        label="Neighbors amount",
                        type="number",
                    )
                    quasar.QToggle(
                        v_model=("umap_random_seed", True),
                        label="Random seed",
                        left_label=True,
                    )
                    quasar.QInput(
                        v_model=("umap_random_seed_value", 1),
                        disable=("!umap_random_seed",),
                        filled=True,
                        stack_label=True,
                        label="Seed value",
                        type="number",
                    )

    def compute_ui(self):
        with html.Div(trame_server=self.server):
            quasar.QBtn(
                label="Compute",
                loading=("is_loading", False),
                click=self.on_run_clicked,
                flat=True,
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
                                    "dataset_options",
                                    [
                                        {"label": Path(p).name, "value": p}
                                        for p in self._dataset_paths
                                    ],
                                ),
                                filled=True,
                                emit_value=True,
                                map_options=True,
                            )
                    self.settings_widget()
                    self.compute_ui()

                # Main content
                with quasar.QPageContainer():
                    with quasar.QPage():
                        with html.Div(classes="row", style="min-height: inherit;"):
                            with html.Div(classes="col q-pa-md"):
                                self.visualization_widget()

                self._ui = layout
        return self._ui


def main(server=None, *args, **kwargs):
    server = get_server(client_type="vue3")
    server.cli.add_argument(
        "--dataset",
        nargs="+",
        required=True,
        help="Path of the json file describing the image dataset",
    )

    known_args, _ = server.cli.parse_known_args()

    embeddings_app = EmbeddingsApp(server, known_args.dataset)
    embeddings_app.ui

    server.start(**kwargs)


if __name__ == "__main__":
    main()
