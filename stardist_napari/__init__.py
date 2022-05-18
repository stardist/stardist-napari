import os

_true = set(("y", "yes", "t", "true", "on", "1"))
DEBUG = os.environ.get("STARDIST_NAPARI_DEBUG", "").lower() in _true
NOPERSIST = os.environ.get("STARDIST_NAPARI_NOPERSIST", "").lower() in _true

from ._dock_widget import plugin_wrapper as make_dock_widget
from ._version import __version__

del os, _true
