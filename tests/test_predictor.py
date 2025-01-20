import pytest
from nrtk_explorer.library.predictor import Predictor
from nrtk_explorer.library.multiprocess_predictor import MultiprocessPredictor
from nrtk_explorer.library.scoring import compute_score
from nrtk_explorer.library.dataset import get_dataset
from utils import get_images, DATASET
import asyncio


def test_predictor_small():
    sample = get_images()
    predictor = Predictor(model_name="hustvl/yolos-tiny")
    img = predictor.eval(sample)
    assert len(img) == len(sample.keys())


@pytest.fixture
def predictor():
    predictor = MultiprocessPredictor(model_name="facebook/detr-resnet-50")
    yield predictor
    predictor.shutdown()


def test_detect(predictor):
    """Test the detect method with sample images."""
    images = get_images()
    results = asyncio.run(predictor.infer(images))
    assert len(results) == len(images), "Number of results should match number of images"
    for img_id, preds in results.items():
        assert isinstance(preds, list), f"Predictions for {img_id} should be a list"


def test_set_model(predictor):
    """Test setting a new model and performing detection."""
    predictor.set_model(model_name="hustvl/yolos-tiny")
    images = get_images()
    results = asyncio.run(predictor.infer(images))
    assert len(results) == len(
        images
    ), "Number of results should match number of images after setting new model"
    for img_id, preds in results.items():
        assert isinstance(
            preds, list
        ), f"Predictions for {img_id} should be a list after setting new model"


def test_scorer():
    ds = get_dataset(DATASET)
    sample = get_images()
    predictor = Predictor(model_name="facebook/detr-resnet-50")
    predictions = predictor.eval(sample)

    dataset_annotations = dict()
    for annotation in ds.anns.values():
        image_annotations = dataset_annotations.setdefault(annotation["image_id"], [])
        image_annotations.append(annotation)

    ground_truth_annotations = dict()
    for img in ds.imgs.values():
        ground_truth_annotations[img["id"]] = dataset_annotations[img["id"]]

    score_output = compute_score(ds, ground_truth_annotations, predictions, 0)

    image_count = len(sample.keys())
    assert len(predictions) == image_count
    assert len(score_output) == image_count
