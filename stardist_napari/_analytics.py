import platform

import csbdeep
import magicgui
import napari
import numpy as np
import stardist
import tensorflow
from csbdeep.utils.tf import keras_import
from stardist.models import StarDist2D, StarDist3D

from . import __version__

keras = keras_import()


try:
    from plausible_events import PlausibleEvents

    PE = PlausibleEvents(domain="stardist-napari")
except:
    PE = None


def consent_event(consent):
    if PE is None:
        return
    PE.event("Consent", dict(value=consent))


def launch_event():
    if PE is None:
        return
    # TODO: check for gpus?
    # gpus = len(tensorflow.config.list_physical_devices('GPU'))
    props = {
        "platform": platform.platform().strip(),
        "python": platform.python_version(),
        "stardist-napari": __version__,
    }
    props.update(
        {
            p.__name__: p.__version__
            for p in (
                napari,
                magicgui,
                tensorflow,
                keras,
                csbdeep,
                stardist,
            )
        }
    )
    PE.event("Launch", props)


def run_event(
    model,
    model_selected,
    models_reg,
    models_reg_public,
    x_shape,
    axes,
    perc_low,
    perc_high,
    norm_image,
    input_scale,
    n_tiles,
    prob_thresh,
    nms_thresh,
    cnn_output,
    norm_axes,
    output_type,
    timelapse,
):
    if PE is None:
        return

    def _model_name():
        # only disclose model names of "public" registered/pre-trained models
        model_type, model_name = model_selected
        if model_type in models_reg:
            return (
                model_name
                if (model_name in models_reg_public.get(model_type, {}))
                else "Custom (registered)"
            )
        else:
            return "Custom (folder)"

    def _shape_pow2(shape, axes):
        return tuple(
            s if a == "C" else int(2 ** np.ceil(np.log2(s))) for s, a in zip(shape, axes)
        )

    props = {
        "model": _model_name(),
        "image_shape": _shape_pow2(x_shape, axes),
        "image_axes": axes,
        "image_norm": (perc_low, perc_high) if norm_image else False,
        "image_scale": input_scale,
        "image_tiles": n_tiles,
        "thresh_prob": prob_thresh,
        "thresh_nms": nms_thresh,
        "output_type": output_type,
        "output_cnn": cnn_output,
        "norm_axes": norm_axes,
    }
    if "T" in axes:
        props["timelapse"] = timelapse
    name = {StarDist2D: "Run 2D", StarDist3D: "Run 3D"}[type(model)]
    PE.event(name, props)
