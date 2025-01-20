import base64
from io import BytesIO
from PIL import Image
from trame.decorators import TrameApp, change, controller
from nrtk_explorer.app.images.image_ids import (
    dataset_id_to_image_id,
    dataset_id_to_transformed_image_id,
)
from nrtk_explorer.app.trame_utils import delete_state
from nrtk_explorer.app.images.cache import LruCache
from nrtk_explorer.library.transforms import ImageTransform


def convert_to_base64(img: Image.Image) -> str:
    """Convert image to base64 string"""
    buf = BytesIO()
    img.save(buf, format="png")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


IMAGE_CACHE_SIZE = 500


@TrameApp()
class Images:
    def __init__(self, server):
        self.server = server
        self.original_images = LruCache(
            IMAGE_CACHE_SIZE,
        )
        self.transformed_images = LruCache(
            IMAGE_CACHE_SIZE,
        )

    def _load_image(self, dataset_id: str):
        img = self.server.context.dataset.get_image(int(dataset_id))
        img.load()  # Avoid OSError(24, 'Too many open files')
        # transforms and base64 encoding require RGB mode
        return img.convert("RGB") if img.mode != "RGB" else img

    def get_image(self, dataset_id: str, **kwargs):
        """For cache side effects pass on_add_item and on_clear_item callbacks as kwargs"""
        image_id = dataset_id_to_image_id(dataset_id)
        image = self.original_images.get_item(image_id) or self._load_image(dataset_id)
        self.original_images.add_item(image_id, image, **kwargs)
        return image

    def get_stateful_image(self, dataset_id: str):
        return self.get_image(
            dataset_id, on_add_item=self._add_image_to_state, on_clear_item=self._delete_from_state
        )

    def _add_image_to_state(self, image_id: str, image: Image.Image):
        self.server.state[image_id] = convert_to_base64(image)

    def _delete_from_state(self, state_key: str):
        delete_state(self.server.state, state_key)

    def get_image_without_cache_eviction(self, dataset_id: str):
        """
        Does not remove items from cache, only adds.
        For computing metrics on all images.
        """
        image_id = dataset_id_to_image_id(dataset_id)
        image = self.original_images.get_item(image_id) or self._load_image(dataset_id)
        self.original_images.add_if_room(image_id, image)
        return image

    def _load_transformed_image(self, transform: ImageTransform, dataset_id: str):
        original = self.get_image_without_cache_eviction(dataset_id)
        transformed = transform.execute(original)
        # So pixel-wise annotation similarity score works
        if original.size != transformed.size:
            return transformed.resize(original.size)
        return transformed

    def _get_transformed_image(self, transform: ImageTransform, dataset_id: str, **kwargs):
        image_id = dataset_id_to_transformed_image_id(dataset_id)
        image = self.transformed_images.get_item(image_id) or self._load_transformed_image(
            transform, dataset_id
        )
        return image_id, image

    def get_transformed_image(self, transform: ImageTransform, dataset_id: str, **kwargs):
        image_id, image = self._get_transformed_image(transform, dataset_id, **kwargs)
        self.transformed_images.add_item(image_id, image, **kwargs)
        return image

    def get_stateful_transformed_image(self, transform: ImageTransform, dataset_id: str):
        return self.get_transformed_image(
            transform,
            dataset_id,
            on_add_item=self._add_image_to_state,
            on_clear_item=self._delete_from_state,
        )

    def get_transformed_image_without_cache_eviction(
        self, transform: ImageTransform, dataset_id: str
    ):
        image_id, image = self._get_transformed_image(transform, dataset_id)
        self.transformed_images.add_if_room(image_id, image)
        return image

    @change("current_dataset")
    def clear_all(self, **kwargs):
        self.original_images.clear()
        self.clear_transformed()

    @controller.add("apply_transform")
    def clear_transformed(self, **kwargs):
        self.transformed_images.clear()
