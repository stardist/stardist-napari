from concurrent.futures import Future


def test_default(make_napari_viewer, plugin, nuclei_2d):
    viewer = make_napari_viewer()
    viewer.add_layer(nuclei_2d)
    viewer.window.add_dock_widget(plugin)
    future: Future = plugin()
    assert isinstance(future, Future)
    assert future.done()
    result = future.result()
    assert len(result) == 2
