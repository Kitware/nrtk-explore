from pathlib import Path

from trame.app import get_server, asynchronous
from trame.widgets import quasar, html
from trame.ui.quasar import QLayout

from nrtk_explorer.app.applet import Applet
from nrtk_explorer.library.dataset import (
    discover_datasets,
    dataset_select_options,
)
import nrtk_explorer.library.transforms as trans
from nrtk_explorer.widgets.nrtk_explorer import ExportWidget


def recursive_rmdir(path: Path):
    if not path.is_dir():
        return

    for item in path.iterdir():
        if item.is_file():
            item.unlink()
        else:
            recursive_rmdir(item)

    path.rmdir()


class ExportApp(Applet):
    def __init__(self, server):
        super().__init__(server)

        self.context.setdefault("repository", None)
        self.state.setdefault("current_dataset", "")
        self.state.setdefault("repository_datasets", [])
        self.state.setdefault("export_status", "idle")
        self.state.setdefault("export_progress", 0)

        self._ui = None

    def on_export_clicked(self, event):
        self.start_export(event["name"], event["full"])

    def start_export(self, name, full):
        if self._exporting_dataset():
            return

        self._export_task = asynchronous.create_task(self.export_dataset(name, full))

    async def export_dataset(self, name, full):
        if self.context.repository is None:
            return

        with self.state:
            self.state.export_status = "pending"
            self.state.export_progress = 0
        await self.server.network_completion

        try:
            await self._export_dataset(name, full)
            with self.state:
                self.state.export_status = "success"
        except Exception:
            with self.state:
                self.state.export_status = "fail"
        finally:
            with self.state:
                self.state.export_progress = 1
                # Update list of available datasets
                self.state.repository_datasets = [
                    str(path) for path in discover_datasets(self.context.repository)
                ]
                self.state.all_datasets = (
                    self.state.input_datasets + self.state.repository_datasets
                )
                self.state.all_datasets_options = dataset_select_options(self.state.all_datasets)
            await self.server.network_completion

    async def _export_dataset(self, name, full):
        tmp_dataset_dir = self.context.repository / "tmp" / name
        dataset_dir = self.context.repository / name
        recursive_rmdir(tmp_dataset_dir)
        Path.mkdir(tmp_dataset_dir, parents=True)

        # Ensure the transform parameters are frozen for the duration of the task
        transforms = []
        for t in self.context.transforms:
            instance = t["instance"]
            new_instance = instance.__class__()
            new_instance.set_parameters(instance.get_parameters())
            transforms.append(new_instance)

        transform = trans.ChainedImageTransform(transforms)

        dataset = self.context.dataset

        if full:
            image_ids = set(dataset.imgs.keys())
        else:
            image_ids = set(self.context.dataset_ids)

        import kwcoco

        new_dataset = kwcoco.CocoDataset()

        # How often to update the progress
        PROGRESS_UPDATE_STEP = 30
        # Ensure a directory doesn't have too many files
        MAX_FILES_PER_DIRECTORY = 100

        def subdir_generator(max_files):
            i = 0

            while True:
                yield str(i // max_files)
                i += 1

        subdir = subdir_generator(MAX_FILES_PER_DIRECTORY)

        for i, image_id in enumerate(image_ids):
            subdir_name = next(subdir)
            destination_dir = tmp_dataset_dir / subdir_name

            if not Path.exists(destination_dir):
                Path.mkdir(destination_dir, parents=True)

            img = dataset.get_image(image_id)
            # transforms require RGB mode
            img = img.convert("RGB") if img.mode != "RGB" else img

            if img.format is not None:
                img_format = img.format
            else:
                img_format = "PNG"

            img_destination = destination_dir / f"{image_id}.{img_format.lower()}"
            transformed_img = transform.execute(img)
            transformed_img.save(img_destination, img_format)

            new_dataset.add_image(img_destination, id=image_id)

            if i % PROGRESS_UPDATE_STEP == 0:
                with self.state:
                    self.state.export_progress = i / len(image_ids)
                await self.server.network_completion

        for cat in dataset.cats.values():
            new_dataset.add_category(**cat)

        for ann in dataset.anns.values():
            if ann["image_id"] in image_ids:
                new_dataset.add_annotation(**ann)

        new_dataset.fpath = tmp_dataset_dir / f"{name}.json"
        new_dataset.reroot()
        new_dataset.dump()

        recursive_rmdir(dataset_dir)
        Path.rename(tmp_dataset_dir, dataset_dir)

    def _exporting_dataset(self):
        return hasattr(self, "_export_task") and not self._export_task.done()

    def export_ui(self):
        with html.Div(trame_server=self.server):
            ExportWidget(
                current_dataset=("current_dataset",),
                repository_datasets=("repository_datasets",),
                export_dataset=(self.on_export_clicked, "[$event]"),
                status=("export_status",),
                progress=("export_progress",),
            )

    @property
    def ui(self):
        if self._ui is None:
            with QLayout(self.server) as layout:
                self._ui = layout

                with quasar.QDrawer(
                    v_model=("leftDrawerOpen", True),
                    side="left",
                    elevated=True,
                    width="500",
                ):
                    self.export_ui()

                with quasar.QPageContainer():
                    with quasar.QPage():
                        with html.Div(classes="row", style="min-height: inherit;"):
                            with html.Div(classes="col q-pa-md"):
                                pass

        return self._ui


def main(server=None, *args, **kwargs):
    server = get_server()
    server.client_type = "vue3"

    app = ExportApp(server)
    app.ui

    server.start(**kwargs)


if __name__ == "__main__":
    main()
