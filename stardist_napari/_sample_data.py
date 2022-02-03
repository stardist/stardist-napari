# from napari_plugin_engine import napari_hook_implementation

def _test_image_nuclei_2d():
    from stardist import data
    return [(data.test_image_nuclei_2d(), {'name': 'nuclei_2d'})]

def _test_image_nuclei_3d():
    from stardist import data
    return [(data.test_image_nuclei_3d(), {'name': 'nuclei_3d'})]


# @napari_hook_implementation
# def napari_provide_sample_data():
#     from stardist import data
#     return {
#         'Nuclei (2D)': _test_image_nuclei_2d,
#         'Nuclei (3D)': _test_image_nuclei_3d,
#     }
