import sys
import numpy as np
import napari

from stardist.models import Config3D, StarDist3D
from stardist.data import  test_image_nuclei_2d, test_image_nuclei_3d
from csbdeep.utils import normalize
import napari
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
    viewer =  napari.Viewer()
    viewer.add_image(x)
    viewer.window.add_plugin_dock_widget('StarDist')

    return viewer
        

def show_napari_3d():
    x = test_image_nuclei_3d()
    viewer =  napari.Viewer()
    viewer.add_image(x)
    viewer.window.add_plugin_dock_widget('StarDist')

    with napari.gui_qt():
        viewer =  napari.Viewer()

        viewer.add_image(x, scale=(2,1,1))

        viewer.window.add_plugin_dock_widget('StarDist')
        
    return viewer

        
if __name__ == '__main__':

    viewer = show_napari_2d()
    if 'run' in sys.argv:
        napari.run()
