import os

DEBUG = os.environ.get("STARDIST_NAPARI_DEBUG", "").lower() in (
    "y",
    "yes",
    "t",
    "true",
    "on",
    "1",
)
del os

from ._dock_widget import plugin_wrapper as make_dock_widget
from ._version import __version__
