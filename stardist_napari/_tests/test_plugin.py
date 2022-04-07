import napari
from stardist.data import test_image_nuclei_2d as image_nuclei_2d

from stardist_napari._dock_widget import plugin_wrapper


def test_call():
    x = napari.layers.Image(image_nuclei_2d())
    plugin = plugin_wrapper()
    out = plugin(None, image=x, axes="YX")
    assert len(out) == 2


# def test_run(make_napari_viewer):
#     viewer = make_napari_viewer()
#     viewer.add_image(image_nuclei_2d())
#     plugin = plugin_wrapper()
#     viewer.window.add_dock_widget(plugin)
