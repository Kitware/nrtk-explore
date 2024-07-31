import gc
import logging
import torch
import transformers
from .images_manager import ImageWithId

from typing import Optional, Sequence

ImageIdToAnnotations = Optional[dict[str, Sequence[dict]]]


class ObjectDetector:
    """Object detection using Hugging Face's transformers library"""

    def __init__(
        self,
        model_name: str = "facebook/detr-resnet-50",
        task: Optional[str] = None,
        force_cpu: bool = False,
    ):
        self.task = task
        self.device = "cuda" if torch.cuda.is_available() and not force_cpu else "cpu"
        self.pipeline = model_name

    @property
    def device(self) -> str:
        """Get the device to use for feature extraction"""
        return str(self._device)

    @device.setter
    def device(self, dev: str):
        """Set the device to use for feature extraction"""
        logging.info(f"Using {dev} devices for feature extraction")
        self._device = torch.device(dev)

    @property
    def pipeline(self) -> transformers.pipeline:
        """Get the pipeline for object detection using Hugging Face's transformers library"""
        return self._pipeline

    @pipeline.setter
    def pipeline(self, model_name: str):
        """Set the pipeline for object detection using Hugging Face's transformers library"""
        if self.task is None:
            self._pipeline = transformers.pipeline(model=model_name, device=self.device)
        else:
            self._pipeline = transformers.pipeline(
                model=model_name, device=self.device, task=self.task
            )
        # Do not display warnings
        transformers.utils.logging.set_verbosity_error()

    def eval(
        self,
        images: Sequence[ImageWithId],
        batch_size: int = 32,
    ) -> ImageIdToAnnotations:
        """Compute object recognition. Returns Annotations grouped by input image paths."""

        # Some models require all the images in a batch to be the same size,
        # otherwise crash or UB.
        batches: dict = {}
        for image in images:
            size = image.image.size
            batches.setdefault(size, [])
            batches[size].append(image)

        adjusted_batch_size = batch_size
        while adjusted_batch_size > 0:
            try:
                predictions_in_baches = [
                    zip(
                        [image.id for image in imagesInBatch],
                        self.pipeline(
                            [image.image for image in imagesInBatch],
                            batch_size=adjusted_batch_size,
                        ),
                    )
                    for imagesInBatch in batches.values()
                ]

                predictions_by_image_id = {
                    image_id: predictions
                    for batch in predictions_in_baches
                    for image_id, predictions in batch
                }
                return predictions_by_image_id

            except RuntimeError as e:
                if "out of memory" in str(e) and adjusted_batch_size > 1:
                    previous_batch_size = adjusted_batch_size
                    adjusted_batch_size = adjusted_batch_size // 2
                    print(
                        f"OOM (Pytorch exception {e}) due to batch_size={previous_batch_size}, setting batch_size={adjusted_batch_size}"
                    )
                else:
                    raise

            finally:
                # Pytorch needs to freed its allocations outside of the exception context
                gc.collect()
                torch.cuda.empty_cache()

        # We should never reach here
        return None
