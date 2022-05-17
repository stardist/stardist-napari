def test_open(make_napari_viewer):
    viewer = make_napari_viewer()
    viewer.open_sample(plugin="stardist-napari", sample="nuclei_2d")
    viewer.open_sample(plugin="stardist-napari", sample="nuclei_3d")
    viewer.open_sample(plugin="stardist-napari", sample="he_2d")
