import sys
import numpy as np
from stardist.models import Config3D, StarDist3D
from stardist.data import  test_image_nuclei_2d, test_image_nuclei_3d
from csbdeep.utils import normalize


def show_surface():

    import napari
    
    model = _model3d()
    img, mask = test_image_nuclei_3d(return_mask=True)
    x = normalize(img, 1, 99.8)
    labels, polys = model.predict_instances(x)

    def surface_from_polys(polys): 
        from stardist.geometry import dist_to_coord3D 
        faces = polys["rays_faces"] 
        coord = dist_to_coord3D(polys["dist"], polys["points"], polys["rays_vertices"]) 
        faces = np.concatenate([faces+coord.shape[1]*i for i in np.arange(len(coord))]) 
        vertices = np.concatenate(coord, axis = 0) 
        values = np.concatenate([np.random.rand()*np.ones(len(c)) for c in coord]) 
        return (vertices,faces,values) 

    surface = surface_from_polys(polys)

    with napari.gui_qt(): 
        # add the surface
        viewer = napari.view_image(img) 
        viewer.add_surface(surface) 


def show_napari_2d():
    import napari
    x = test_image_nuclei_2d()

    with napari.gui_qt():
        viewer =  napari.Viewer()

        viewer.add_image(x)

        viewer.window.add_plugin_dock_widget('StarDist')

def show_napari_3d():
    import napari
    x = test_image_nuclei_3d()

    with napari.gui_qt():
        viewer =  napari.Viewer()

        viewer.add_image(x)

        viewer.window.add_plugin_dock_widget('StarDist')
        
if __name__ == '__main__':

    show_napari_2d()

