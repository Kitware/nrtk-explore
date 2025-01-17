import gc
import logging
import numpy as np
import timm
import torch
from PIL.Image import Image

from torch.utils.data import DataLoader, Dataset

IMAGE_MODEL_RESOLUTION = (224, 224)
STARTING_BATCH_SIZE = 32


# Create a dataset for images
class ImagesDataset(Dataset):
    def __init__(self, images):
        self.images = images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        return self.images[i][0]


class EmbeddingsExtractor:
    def __init__(self, model_name="resnet50d", force_cpu=False):
        self.device = "cuda" if torch.cuda.is_available() and not force_cpu else "cpu"
        self.model = model_name
        self.reset()

    def reset(self):
        self.batch_size = STARTING_BATCH_SIZE

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, dev):
        logging.info(f"Using {dev} devices for feature extraction")
        self._device = torch.device(dev)

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model_name):
        # Create model in inference mode
        self._model = (
            timm.create_model(model_name, pretrained=True, num_classes=0).to(self._device).eval()
        )

        # Create transformer for image
        self._model_transformer = timm.data.create_transform(
            **timm.data.resolve_model_data_config(self._model.pretrained_cfg)
        )

    def transform_image(self, image: Image):
        """Transform image to fit model input size and format"""
        img = image.resize(IMAGE_MODEL_RESOLUTION).convert("RGB")
        return self._model_transformer(img).unsqueeze(0)

    def extract(self, images, batch_size=0):
        """Extract features from images"""
        if len(images) == 0:
            return []

        if batch_size != 0:
            self.batch_size = batch_size

        features = list()
        transformed_images = [self.transform_image(img) for img in images]

        while self.batch_size > 0:
            try:
                for batch in DataLoader(
                    ImagesDataset(transformed_images), batch_size=self.batch_size
                ):
                    # Copy image to device if using device
                    if self.device.type == "cuda":
                        batch = batch.cuda()

                    features.append(self.model(batch).numpy(force=True))
                return np.vstack(features)

            except RuntimeError as e:
                if "out of memory" in str(e) and self.batch_size > 1:
                    previous_batch_size = self.batch_size
                    self.batch_size //= 2
                    print(
                        f"Changing extract batch_size from {previous_batch_size} to {self.batch_size} because caught out of memory exception:\n{e}"
                    )
                else:
                    raise

            finally:
                # Pytorch needs to freed its allocations outside of the exception context
                gc.collect()
                torch.cuda.empty_cache()

        # We should never reach here
        return []
