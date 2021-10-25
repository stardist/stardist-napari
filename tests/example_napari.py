import sys
import numpy as np
from stardist.models import Config3D, StarDist3D
from stardist.data import  test_image_nuclei_2d, test_image_nuclei_3d
from csbdeep.utils import normalize
import napari


# def show_surface():
#     model = StarDist3D.from_pretrained('3D_demo')
#     img, mask = test_image_nuclei_3d(return_mask=True)
#     x = normalize(img, 1, 99.8)
#     labels, polys = model.predict_instances(x)

#     def surface_from_polys(polys):
#         from stardist.geometry import dist_to_coord3D
#         faces = polys["rays_faces"]
#         coord = dist_to_coord3D(polys["dist"], polys["points"], polys["rays_vertices"])
#         faces = np.concatenate([faces+coord.shape[1]*i for i in np.arange(len(coord))])
#         vertices = np.concatenate(coord, axis = 0)
#         values = np.concatenate([np.random.rand()*np.ones(len(c)) for c in coord])
#         return (vertices,faces,values)

#     surface = surface_from_polys(polys)

#     # add the surface
#     viewer = napari.view_image(img)
#     viewer.add_surface(surface)


def show_napari_2d():
    x = test_image_nuclei_2d()
    viewer =  napari.Viewer()
    viewer.add_image(x)
    viewer.window.add_plugin_dock_widget('StarDist')

def show_napari_3d():
    x = test_image_nuclei_3d()
    viewer =  napari.Viewer()
    viewer.add_image(x)
    viewer.window.add_plugin_dock_widget('StarDist')

if __name__ == '__main__':

    show_napari_2d()
    if 'run' in sys.argv:
        napari.run()
