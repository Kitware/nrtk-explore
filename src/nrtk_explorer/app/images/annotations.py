from typing import Dict, Sequence
from functools import lru_cache, partial
from PIL import Image
from nrtk_explorer.app.images.cache import LruCache
from nrtk_explorer.library.object_detector import ObjectDetector
from nrtk_explorer.library.scoring import partition


ANNOTATION_CACHE_SIZE = 1000


class DeleteCallbackRef:
    def __init__(self, del_callback, value):
        self.del_callback = del_callback
        self.value = value

    def __del__(self):
        self.del_callback()


def get_annotations_from_dataset(
    context, add_to_cache_callback, delete_from_cache_callback, dataset_id: str
):
    dataset = context.dataset
    annotations = [
        annotation
        for annotation in dataset.anns.values()
        if str(annotation["image_id"]) == dataset_id
    ]
    add_to_cache_callback(dataset_id, annotations)
    with_id = partial(delete_from_cache_callback, dataset_id)
    return DeleteCallbackRef(with_id, annotations)


class GroundTruthAnnotations:
    def __init__(
        self,
        context,  # for dataset
        add_to_cache_callback,
        delete_from_cache_callback,
    ):
        with_callbacks = partial(
            get_annotations_from_dataset,
            context,
            add_to_cache_callback,
            delete_from_cache_callback,
        )
        self.get_annotations_for_image = lru_cache(maxsize=ANNOTATION_CACHE_SIZE)(with_callbacks)

    def get_annotations(self, dataset_ids: Sequence[str]):
        return {
            dataset_id: self.get_annotations_for_image(dataset_id).value
            for dataset_id in dataset_ids
        }

    def cache_clear(self):
        self.get_annotations_for_image.cache_clear()


class DetectionAnnotations:
    def __init__(
        self,
        add_to_cache_callback,
        delete_from_cache_callback,
    ):
        self.cache = LruCache(ANNOTATION_CACHE_SIZE)
        self.add_to_cache_callback = add_to_cache_callback
        self.delete_from_cache_callback = delete_from_cache_callback

    def get_annotations(self, detector: ObjectDetector, id_to_image: Dict[str, Image.Image]):
        hits, misses = partition(
            lambda id: self.cache.get_item(id) is not None, id_to_image.keys()
        )

        to_detect = {id: id_to_image[id] for id in misses}
        predictions = detector.eval(
            to_detect,
        )
        for id, annotations in predictions.items():
            self.cache.add_item(
                id, annotations, self.add_to_cache_callback, self.delete_from_cache_callback
            )

        return {id: self.cache.get_item(id) for id in id_to_image.keys()}

    def cache_clear(self):
        self.cache.clear()
