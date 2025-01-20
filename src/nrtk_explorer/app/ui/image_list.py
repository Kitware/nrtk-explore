from pathlib import Path
from trame.widgets import html, quasar, client
from trame.decorators import TrameApp, change
from trame_annotations.widgets.annotations import ImageDetection
from nrtk_explorer.app.trame_utils import change_checker
from nrtk_explorer.app.images.image_ids import get_image_state_keys


CSS_FILE = Path(__file__).with_name("image_list.css")

COLUMNS = [
    {"name": "id", "label": "Dataset ID", "field": "id", "sortable": True},
    {"name": "truth", "label": "Original: Ground Truth Annotations", "field": "truth"},
    {"name": "original", "label": "Original: Detection Annotations", "field": "original"},
    {
        "name": "transformed",
        "label": "Transformed: Detection Annotations",
        "field": "transformed",
    },
    {
        "name": "original_ground_to_original_detection_score",
        "label": "Ground Truth - Original Detection | Annotations Similarity",
        "field": "original_ground_to_original_detection_score",
        "sortable": True,
    },
    {
        "name": "ground_truth_to_transformed_detection_score",
        "label": "Ground Truth - Transformed Detection | Annotations Similarity",
        "field": "ground_truth_to_transformed_detection_score",
        "sortable": True,
    },
    {
        "name": "original_detection_to_transformed_detection_score",
        "label": "Original Detection - Transformed Detection | Annotations Similarity",
        "field": "original_detection_to_transformed_detection_score",
        "sortable": True,
    },
]


def make_dependent_columns_handler(state, columns):
    toggle_column = columns[0]
    dependent_columns = columns[1:]

    def column_toggler(old_columns, new_columns):
        dependant_columns_visible = any(col in state.visible_columns for col in dependent_columns)
        if toggle_column not in state.visible_columns and dependant_columns_visible:
            state.visible_columns = [
                col for col in state.visible_columns if col not in dependent_columns
            ]
            return

        toggle_column_turned_on = toggle_column in new_columns and toggle_column not in old_columns
        if toggle_column_turned_on:
            state.visible_columns = list(set([*state.visible_columns, *dependent_columns]))

    change_checker(state, "visible_columns")(column_toggler)


ORIGINAL_COLUMNS = [
    "original",
    "original_ground_to_original_detection_score",
]


TRANSFORM_COLUMNS = [
    "transformed",
    "ground_truth_to_transformed_detection_score",
    "original_detection_to_transformed_detection_score",
]

visible_columns_initialized = False


def init_visible_columns(state):
    global visible_columns_initialized
    if visible_columns_initialized:
        return
    state.visible_columns = [col["name"] for col in COLUMNS]
    make_dependent_columns_handler(state, ORIGINAL_COLUMNS)
    make_dependent_columns_handler(state, TRANSFORM_COLUMNS)
    visible_columns_initialized = True


class ImageWithSpinner(html.Div):
    def __init__(
        self,
        identifier=None,
        src=None,
        annotations=None,
        categories=None,
        selected=None,
        hover=None,
        container_selector=None,
        **kwargs,
    ):
        super().__init__(
            classes="relative-position",
            **kwargs,
        )
        with self:
            ImageDetection(
                identifier=identifier,
                src=src,
                annotations=(f"show_annotations_on_images ? {annotations[0]} : []",),
                categories=categories,
                selected=selected,
                hover=hover,
                container_selector=container_selector,
                score_threshold=("confidence_score_threshold",),
            )
            quasar.QInnerLoading(
                showing=(f"!{src[0]} || (show_annotations_on_images && !{annotations[0]}.value)",)
            )


@TrameApp()
class ImageList(html.Div):

    # keep identical ID across datasets from stopping update
    @change("current_dataset")
    def clear_old_visible_ids(self, **kwargs):
        self.visible_ids = set()

    def set_in_view_ids(self, ids):
        visible = set(ids)
        if self.visible_ids != visible:
            self.visible_ids = visible
            self.scroll_callback(self.visible_ids)

    def _set_image_list_ids(self, dataset_ids):
        # create reactive variables so ImageDetection components have live Refs
        for id in dataset_ids:
            keys = get_image_state_keys(id)
            for key in keys.values():
                if not self.state.has(key):
                    self.state[key] = None
        self.state.image_list_ids = dataset_ids

    @change("user_selected_ids")
    def update_image_list_ids(self, **kwargs):
        self._set_image_list_ids(self.state.user_selected_ids)

    @change("image_list_ids")
    def check_images_in_view(self, **kwargs):
        if self.state.image_list_view_mode == "grid":
            self.server.controller.get_visible_ids_for_grid()
            return
        self.server.js_call(ref="image-list", method="resetVirtualScroll")

    @change("image_list_view_mode")
    def update_pagination(self, **kwargs):
        old_pagination = self.state.pagination or {}
        if self.state.image_list_view_mode == "grid":
            self.state.pagination = {**old_pagination, "rowsPerPage": 12}
            self.server.controller.get_visible_ids_for_grid()
        else:
            self.state.pagination = {**old_pagination, "rowsPerPage": 0}  # show all rows

    def __init__(self, on_scroll, on_hover, **kwargs):
        super().__init__(classes="full-height", **kwargs)
        self.visible_ids = set()
        self.scroll_callback = on_scroll
        self.update_pagination()
        self.state.client_only("image_size_image_list")

        self.state.columns = COLUMNS
        self.state.client_only("columns")

        with self:
            client.Style(CSS_FILE.read_text())
            get_visible_ids_for_grid = client.JSEval(
                exec=f'''
                            ;const list = trame.refs['image-list']
                            if (!list) return
                            // wait a tick so pagination prop is applied to computedRows
                            window.setTimeout(() => {{
                                const ids = list.computedRows.map(i => i.id)
                                trigger('{ self.ctrl.trigger_name(self.set_in_view_ids) }', [ids])
                            }}, 0)
                        "''',
            )
            self.ctrl.get_visible_ids_for_grid = get_visible_ids_for_grid.exec
            self.ctrl.check_images_in_view = self.check_images_in_view
            with quasar.QTable(
                ref=("image-list"),
                classes="full-height sticky-header",
                flat=True,
                hide_bottom=("image_list_view_mode !== 'grid'", True),
                title="Sampled Images",
                loading=("updating_images", False),
                grid=("image_list_view_mode === 'grid'", False),
                filter=("image_list_search", ""),
                id="image-list",  # set id so that the ImageDetection component can select the container for tooltip positioning
                visible_columns=("visible_columns",),
                columns=("columns",),
                rows=(
                    r"""image_list_ids.map((id) =>
                            {
                                const meta = get(`meta_${id}`)?.value ?? {original_ground_to_original_detection_score: 0, ground_truth_to_transformed_detection_score: 0, original_detection_to_transformed_detection_score: 0}
                                const original_id = `img_${id}`
                                const transformed_id = `transformed_img_${id}`
                                return {
                                    ...meta,
                                    original_ground_to_original_detection_score: meta.original_ground_to_original_detection_score.toFixed(2),
                                    ground_truth_to_transformed_detection_score: meta.ground_truth_to_transformed_detection_score.toFixed(2),
                                    original_detection_to_transformed_detection_score: meta.original_detection_to_transformed_detection_score.toFixed(2),
                                    id,
                                    original: original_id,
                                    original_src: get(original_id).value,
                                    transformed: transformed_id,
                                    transformed_src: get(transformed_id).value,
                                    groundTruthAnnotations: get(`result_${id}`),
                                    originalAnnotations: get(`result_img_${id}`),
                                    transformedAnnotations: get(`result_transformed_img_${id}`),
                                }
                            })
                        """,
                ),
                row_key="id",
                rows_per_page_options=(
                    "image_list_view_mode === 'table' ? [0] : [6, 12, 24]",
                    "[0]",
                ),
                raw_attrs=[
                    "virtual-scroll",
                    "virtual-scroll-slice-size='2'",
                    "virtual-scroll-item-size='200'",
                    # e.ref._.props.items is sorted+filtered rows like the QTable.computedRows computed prop
                    f'''@virtual-scroll="(e) => {{
                        const ids = e.ref._.props.items.map(i => i.id).slice(e.from, e.to + 1)
                        trigger('{ self.server.controller.trigger_name(self.set_in_view_ids) }', [ids])
                        }}"''',
                    "virtual-scroll-sticky-size-start='48'",
                    r"v-model:pagination='pagination'",
                    f'''@update:pagination="(e) => {{
                            trigger('{ self.server.controller.trigger_name(self.ctrl.check_images_in_view) }')
                        }}"''',
                ],
            ):
                # ImageDetection component for image columns
                with html.Template(
                    v_slot_body_cell_truth=True,
                    __properties=[("v_slot_body_cell_truth", "v-slot:body-cell-truth='props'")],
                ):
                    with quasar.QTd():
                        ImageWithSpinner(
                            style=("`width: ${image_size_image_list}rem; float: inline-end;`",),
                            identifier=("props.row.original",),
                            src=("props.row.original_src",),
                            annotations=("props.row.groundTruthAnnotations",),
                            categories=("annotation_categories",),
                            selected=("(props.row.original == hovered_id)",),
                            hover=(on_hover, "[$event]"),
                            container_selector="#image-list .q-table__middle",
                        )
                with html.Template(
                    v_slot_body_cell_original=True,
                    __properties=[
                        ("v_slot_body_cell_original", "v-slot:body-cell-original='props'")
                    ],
                ):
                    with quasar.QTd():
                        ImageWithSpinner(
                            style=("`width: ${image_size_image_list}rem; float: inline-end;`",),
                            identifier=("props.row.original",),
                            src=("props.row.original_src",),
                            annotations=("props.row.originalAnnotations",),
                            categories=("annotation_categories",),
                            selected=("(props.row.original == hovered_id)",),
                            hover=(on_hover, "[$event]"),
                            container_selector="#image-list .q-table__middle",
                        )
                with html.Template(
                    v_slot_body_cell_transformed=True,
                    __properties=[
                        (
                            "v_slot_body_cell_transformed",
                            "v-slot:body-cell-transformed='props'",
                        )
                    ],
                ):
                    with quasar.QTd():
                        ImageWithSpinner(
                            style=("`width: ${image_size_image_list}rem; float: inline-end;`",),
                            identifier=("props.row.transformed",),
                            src=("props.row.transformed_src",),
                            annotations=("props.row.transformedAnnotations",),
                            categories=("annotation_categories",),
                            selected=("(props.row.transformed == hovered_id)",),
                            hover=(on_hover, "[$event]"),
                            container_selector="#image-list .q-table__middle",
                        )
                # Grid Mode template for each row/grid-item
                with html.Template(
                    v_slot_item=True,
                    __properties=[("v_slot_item", "v-slot:item='props'")],
                ):
                    with html.Div(classes="q-pa-xs col-xs-12 col-sm-6 col-md-4"):
                        with quasar.QCard(flat=True, bordered=True):
                            with html.Div(classes="row"):
                                with html.Div(
                                    classes="col-4 q-pa-sm",
                                    v_if=("props.cols.map(c => c.name).includes('truth')", True),
                                ):
                                    html.Div(
                                        "Original: Ground Truth Annotations",
                                        classes="text-center",
                                    )
                                    ImageWithSpinner(
                                        identifier=("props.row.original",),
                                        src=("props.row.original_src",),
                                        annotations=("props.row.groundTruthAnnotations",),
                                        categories=("annotation_categories",),
                                        selected=("(props.row.original == hovered_id)",),
                                        hover=(on_hover, "[$event]"),
                                    )
                                with html.Div(
                                    classes="col-4 q-pa-sm",
                                    v_if=(
                                        "props.cols.map(c => c.name).includes('original')",
                                        True,
                                    ),
                                ):
                                    html.Div(
                                        "Original: Detection Annotations",
                                        classes="text-center",
                                    )
                                    ImageWithSpinner(
                                        identifier=("props.row.original",),
                                        src=("props.row.original_src",),
                                        annotations=("props.row.originalAnnotations",),
                                        categories=("annotation_categories",),
                                        selected=("(props.row.original == hovered_id)",),
                                        hover=(on_hover, "[$event]"),
                                    )
                                with html.Div(
                                    classes="col-4 q-pa-sm",
                                    v_if=(
                                        "props.cols.map(c => c.name).includes('transformed')",
                                        True,
                                    ),
                                ):
                                    html.Div(
                                        "Transformed: Detection Annotations",
                                        classes="text-center",
                                    )
                                    ImageWithSpinner(
                                        identifier=("props.row.transformed",),
                                        src=("props.row.transformed_src",),
                                        annotations=("props.row.transformedAnnotations",),
                                        categories=("annotation_categories",),
                                        selected=("(props.row.transformed == hovered_id)",),
                                        hover=(on_hover, "[$event]"),
                                    )
                            with quasar.QList(
                                dense=True,
                            ):
                                with quasar.QItem(
                                    v_for=(
                                        "col in props.cols.filter(col => !(['truth', 'original', 'transformed'].includes(col.name)))",
                                    ),
                                    key=("col.name",),
                                ):
                                    with quasar.QItemSection():
                                        with quasar.QItemLabel():
                                            html.Div("{{col.label}}")
                                    with quasar.QItemSection(side=True):
                                        with quasar.QItemLabel(
                                            caption=True,
                                        ):
                                            html.Div("{{col.value}}")
                # Top control bar for visible-columns, search, table-grid switch, full-screen
                with html.Template(
                    v_slot_top=True,
                    __properties=[("v_slot_top", "v-slot:top='props'")],
                ):
                    html.Span("Sampled Images", classes="col q-table__title")
                    # Image size
                    quasar.QIcon(name="zoom_in", size="1.2rem", classes="q-px-sm")
                    html.Span("Image Size")
                    quasar.QSlider(
                        classes="q-pl-sm q-pr-lg",
                        v_model=("image_size_image_list", 12),
                        raw_attrs=[
                            ":min='5'",
                            ":max='40'",
                        ],
                        style="width: 12rem;",
                    )
                    # Annotations visible switch
                    quasar.QIcon(
                        name="picture_in_picture", size="1.2rem", classes="q-pl-lg q-pr-sm"
                    )
                    html.Span("Show Annotations")
                    quasar.QToggle(
                        v_model=("show_annotations_on_images", True),
                    )
                    quasar.QSelect(
                        classes="q-pl-xl q-pr-lg",
                        v_model=("visible_columns"),
                        multiple=True,
                        dense=True,
                        options_dense=True,
                        emit_value=True,
                        map_options=True,
                        options=("columns",),
                        option_value="name",
                        options_cover=True,
                        raw_attrs=[
                            ":display-value='$q.lang.table.columns'",
                        ],
                    )
                    quasar.QBtn(
                        classes="q-pl-lg q-pr-xl",
                        icon="fullscreen",
                        dense=True,
                        flat=True,
                        click="props.toggleFullscreen",
                    )
                    quasar.QBtnToggle(
                        v_model=("image_list_view_mode", "table"),
                        raw_attrs=[
                            ":options=\"[{label: 'Table', value: 'table'}, {label: 'Grid', value: 'grid'}]\"",
                        ],
                    )
                    quasar.QInput(
                        classes="q-pl-xl",
                        v_model=("image_list_search", ""),
                        debounce="300",
                        label="Search",
                        dense=True,
                    )
