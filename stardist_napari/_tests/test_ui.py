from concurrent.futures import Future

import pytest

from stardist_napari import NOPERSIST, NOTHREADS

skip_if_threads = pytest.mark.skipif(
    not NOTHREADS, reason="STARDIST_NAPARI_NOTHREADS not set"
)
skip_if_persist = pytest.mark.skipif(
    not NOPERSIST, reason="STARDIST_NAPARI_NOPERSIST not set"
)


@skip_if_persist
@skip_if_threads
def test_basics(make_napari_viewer, plugin, nuclei_2d):
    viewer = make_napari_viewer()
    viewer.add_layer(nuclei_2d)
    viewer.window.add_dock_widget(plugin)
    future: Future = plugin()
    assert isinstance(future, Future)
    assert future.done()
    result = future.result()
    assert len(result) == 2
