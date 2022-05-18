import os
import sys

os.environ["STARDIST_NAPARI_DEBUG"] = "1"

import napari
import numpy as np
from csbdeep.utils import normalize
from stardist.data import test_image_nuclei_2d, test_image_nuclei_3d
from stardist.models import StarDist2D, StarDist3D

from stardist_napari._dock_widget import surface_from_polys


def show_surface():
    model = _model3d()
    img, mask = test_image_nuclei_3d(return_mask=True)
    x = normalize(img, 1, 99.8)
    labels, polys = model.predict_instances(x)
    surface = surface_from_polys(polys)
    # add the surface
    viewer = napari.view_image(img)
    viewer.add_surface(surface)

    return viewer


def show_napari_2d():
    x = test_image_nuclei_2d()
    viewer = napari.Viewer()
    viewer.add_image(x)
    viewer.window.add_plugin_dock_widget("stardist-napari", "StarDist")
    return viewer


def show_napari_2d_time():
    import napari
    from scipy.ndimage import rotate

    x = np.stack(
        [
            rotate(test_image_nuclei_2d(), deg, reshape=False, mode="reflect")
            for deg in np.linspace(0, 50, 5)
        ],
        axis=0,
    )

    viewer = napari.Viewer()
    viewer.add_image(x, scale=(1, 1, 1))
    # viewer.add_image(x[...,:1], scale=(1,1,1))
    # viewer.add_image(x[:1], scale=(1,1,1))
    # viewer.add_image(np.random.uniform(size=(128,128,3)), rgb=True)
    viewer.window.add_plugin_dock_widget("stardist-napari", "StarDist")
    return viewer


def show_napari_3d_time():
    import napari

    x = test_image_nuclei_3d()
    x = np.stack([np.roll(x, n) for n in np.arange(0, 30, 10)], axis=0)

    viewer = napari.Viewer()
    viewer.add_image(x, scale=(1, 1, 1, 1))
    viewer.window.add_plugin_dock_widget("stardist-napari", "StarDist")
    return viewer


def show_napari_3d():
    x = test_image_nuclei_3d()
    viewer = napari.Viewer()
    viewer.add_image(x)
    viewer.window.add_plugin_dock_widget("stardist-napari", "StarDist")

    viewer = napari.Viewer()
    viewer.add_image(x, scale=(2, 1, 1))
    viewer.window.add_plugin_dock_widget("stardist-napari", "StarDist")
    return viewer


if __name__ == "__main__":
    viewer = show_napari_2d()
    # viewer = show_napari_2d_time()
    # viewer = show_napari_3d_time()
    # if 'run' in sys.argv:
    napari.run()
