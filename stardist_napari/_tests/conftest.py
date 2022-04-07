import napari
import pytest
from stardist import data

from .. import make_dock_widget


@pytest.fixture(scope="function")
def plugin():
    return make_dock_widget()


@pytest.fixture(scope="session")
def nuclei_2d():
    img = data.test_image_nuclei_2d()
    img = img[128:-128, 128:-128]  # make smaller to speed up tests
    return napari.layers.Image(img, name="nuclei_2d")


@pytest.fixture(scope="session")
def nuclei_3d():
    return napari.layers.Image(data.test_image_nuclei_3d(), name="nuclei_3d")


@pytest.fixture(scope="session")
def he_2d():
    return napari.layers.Image(data.test_image_he_2d(), rgb=True, name="he_2d")
