import os

_true = set(("y", "yes", "t", "true", "on", "1"))
DEBUG = os.environ.get("STARDIST_NAPARI_DEBUG", "").lower() in _true
NOPERSIST = os.environ.get("STARDIST_NAPARI_NOPERSIST", "").lower() in _true
NOTHREADS = os.environ.get("STARDIST_NAPARI_NOTHREADS", "").lower() in _true

from ._dock_widget import plugin_dock_widget as make_dock_widget
from ._dock_widget import plugin_function as make_dock_widget_function
from ._version import __version__

del os, _true
