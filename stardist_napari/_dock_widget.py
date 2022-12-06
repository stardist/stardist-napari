"""
TODO:
- ability to cancel running stardist computation
- run only on field of view (needs testing)
  - https://forum.image.sc/t/how-could-i-get-the-viewed-coordinates/49709
  - https://napari.zulipchat.com/#narrow/stream/212875-general/topic/corner.20pixels.20and.20dims.20displayed
  - https://github.com/napari/napari/issues/2487
- add general tooltip help/info messages
- option to use CPU or GPU, limit tensorflow GPU memory ('allow_growth'?)
- alternative normalization options besides percentile (e.g. absolute min/max values for patho images)
"""

import functools
import numbers
import sys
import time
import warnings
from concurrent.futures import Future
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, List, Sequence, Type, Union

import napari
import numpy as np
from csbdeep.utils import (
    _raise,
    axes_check_and_normalize,
    axes_dict,
    load_json,
    move_image_axes,
    normalize,
)
from magicgui import magicgui, register_type
from napari.qt.threading import thread_worker
from napari.types import LayerDataTuple
from napari.utils import _magicgui, progress
from napari.utils.colormaps import label_colormap
from packaging import version
from psygnal import Signal
from qtpy.QtWidgets import QSizePolicy

from . import DEBUG, NOPERSIST, NOTHREADS

_Future = Future
if sys.version_info < (3, 9):
    # proxy type because Future is not subscriptable in Python 3.8 or lower
    _Future = List
    # register proxy type with magicgui
    register_type(
        _Future[List[LayerDataTuple]], return_callback=_magicgui.add_future_data
    )

SLOW_SHAPE_LAYER = True
if version.parse(napari.__version__) >= version.parse("0.4.16"):
    try:
        import triangle

        SLOW_SHAPE_LAYER = False
    except:
        pass

# region utils
# -------------------------------------------------------------------------

CUSTOM_MODEL = "CUSTOM_MODEL"


class Output(Enum):
    Labels = "Label Image"
    Polys = "Polygons / Polyhedra"
    Both = "Both"


class TimelapseLabels(Enum):
    Match = "Match to previous frame (via overlap)"
    Unique = "Unique through time"
    Separate = "Separate per frame (no processing)"


def get_enum_member(enum: Union[Type[Output], Type[TimelapseLabels]], value: str):
    for e in enum:
        if e.value == value:
            return e
    raise ValueError(f"no member of {enum} found with value '{value}'")


def get_model_config_and_thresholds(path):
    config = load_json(str(path / "config.json"))
    thresholds = None
    try:
        # not all models have associated thresholds
        thresholds = load_json(str(path / "thresholds.json"))
    except FileNotFoundError:
        pass
    return config, thresholds


def get_data(image):
    image = image.data[0] if image.multiscale else image.data
    if not all(hasattr(image, attr) for attr in ("shape", "ndim", "__getitem__")):
        image = np.asanyarray(image)
    return image


def change_handler(*widgets, init=True, debug=DEBUG):
    def decorator_change_handler(handler):
        @functools.wraps(handler)
        def wrapper(*args):
            source = Signal.sender()
            emitter = Signal.current_emitter()
            if debug:
                # print(f"{emitter}: {source} = {args!r}")
                print(f"EVENT '{str(emitter.name)}': {source.name:>20} = {args!r}")
                # print(f"                 {source.name:>14}.value = {source.value}")
            return handler(*args)

        for widget in widgets:
            widget.changed.connect(wrapper)
            if init:
                widget.changed(widget.value)
        return wrapper

    return decorator_change_handler


def surface_from_polys(polys):
    from stardist.geometry import dist_to_coord3D

    dist = polys["dist"]
    points = polys["points"]
    rays_vertices = polys["rays_vertices"]
    rays_faces = polys["rays_faces"].copy()
    coord = dist_to_coord3D(dist, points, rays_vertices)

    if not all((coord.ndim == 3, coord.shape[-1] == 3, rays_faces.shape[-1] == 3)):
        raise ValueError(f"Wrong shapes! coord -> (m,n,3) rays_faces -> (k,3)")

    vertices, faces, values = [], [], []
    for i, xs in enumerate(coord, start=1):
        values.extend(i + np.zeros(len(xs)))
        vertices.extend(xs)
        faces.extend(rays_faces.copy())
        rays_faces += len(xs)

    return [np.array(vertices), np.array(faces), np.array(values)]


def corner_pixels_multiscale(layer):
    # layer.corner_pixels are with respect to the currently used resolution level (layer.data_level)
    # -> convert to reference highest resolution level (layer.data[0]), which is used by stardist
    factor = layer.downsample_factors[layer.data_level]
    scaled_corner = np.round(layer.corner_pixels * factor).astype(int)
    shape_max = layer.data[0].shape
    # if layer.rgb -> len(shape_max) == 1 + len(factor)
    for i in range(len(factor)):
        scaled_corner[:, i] = np.clip(scaled_corner[:, i], 0, shape_max[i])
    return scaled_corner


def create_class_labels(labels: np.ndarray, class_ids: Sequence[int], n_classes: int):
    labels_cls = np.zeros_like(labels)
    for c in range(n_classes + 1):
        idx = (1 + np.where(class_ids == c)[0]).tolist()
        labels_cls[np.isin(labels, idx)] = c
    return labels_cls


def axes_permutation(axes_from: str, axes_to: str, x: Union[list, None] = None):
    """
    assumption: strings axes_from and axes_to are permutations of each other
    returns the index permutation to convert axes_from to axes_to
    if list x is given, applies the index permutation to x instead of returning it
    """
    axes_from = axes_check_and_normalize(axes_from)
    axes_to = axes_check_and_normalize(axes_to)
    assert len(set(axes_from).symmetric_difference(set(axes_to))) == 0
    assert x is None or (isinstance(x, list) and len(x) == len(axes_from))
    ind = [axes_from.index(a) for a in axes_to]
    return ind if x is None else [x[i] for i in ind]


def move_layer_axes(layer: napari.types.FullLayerData, axes_from: str, axes_to: str):
    """
    layer:
        napari full layer data tuple (data, options, type)
    axes_from:
        axes of output, that corresponds to layer data
    axes_to:
        axes of input image, to which the axes of the layer data should be changed

    returns modified layer data tuple, with adapted data and options (scale, translate)
    """
    axes_from = axes_check_and_normalize(axes_from)
    axes_to = axes_check_and_normalize(axes_to)
    if axes_from == axes_to:
        return layer

    data, options, ltype = layer
    # print(ltype, options)
    expand_axes = lambda x, *args: x  # do nothing
    c_axes_from = axes_from  # channel-adjusted axes_from

    # input (axes_to) has channels, but output (axes_from) doesn't
    if "C" in axes_to and "C" not in axes_from:
        if axes_to[0] == "C":
            # input starts with channel dimension -> ignore for outputs
            axes_to = axes_to.replace("C", "")
        else:
            # input doesn't start with channels -> add dummy channel dimension to output
            ch = axes_to.index("C")
            expand_axes = lambda x, v=0: np.insert(x, ch, v, axis=-1)

            c_axes_from = list(axes_from)
            c_axes_from.insert(ch, "C")
            c_axes_from = "".join(c_axes_from)

    if ltype.lower() in ("image", "labels"):
        data = move_image_axes(data, axes_from, axes_to, adjust_singletons=True)
    elif ltype.lower() in ("shapes",):
        data = expand_axes(data)[..., axes_permutation(c_axes_from, axes_to)]
    elif ltype.lower() in ("surface",):
        vertices, faces, values = data
        vertices = expand_axes(vertices)[..., axes_permutation(c_axes_from, axes_to)]
        faces = expand_axes(faces)[..., axes_permutation(c_axes_from, axes_to)]
        data = [vertices, faces, values]
    else:
        raise NotImplementedError(f"layer type '{ltype}' not supported")

    # update scale and translate
    for k, e in (("scale", 1), ("translate", 0)):
        v = options.get(k)
        if v is not None:
            options[k] = axes_permutation(c_axes_from, axes_to, list(expand_axes(v, e)))

    # print(ltype, options)
    return data, options, ltype


# endregion
# region plugin
# -------------------------------------------------------------------------


def _plugin_wrapper():
    # delay imports until plugin is requested by user
    # -> especially those importing tensorflow (csbdeep.models.*, stardist.models.*)
    # TODO: rethink wrapper, since not really necessary anymore with npe2,
    #       but still want to avoid importing tensorflow if not needed
    #       (e.g. just to open sample data)
    from csbdeep.models.pretrained import get_model_folder, get_registered_models
    from stardist.matching import group_matching_labels
    from stardist.models import StarDist2D, StarDist3D
    from stardist.utils import abspath

    # -------------------------------------------------------------------------

    _models, _aliases = {}, {}
    models_reg = {}
    for cls in (StarDist2D, StarDist3D):
        # get available models for class
        _models[cls], _aliases[cls] = get_registered_models(cls)
        # use first alias for model selection (if alias exists)
        models_reg[cls] = [
            ((_aliases[cls][m][0] if len(_aliases[cls][m]) > 0 else m), m)
            for m in _models[cls]
        ]

    model_configs = dict()
    model_threshs = dict()
    model_selected = None

    model_type_choices = [
        ("2D", StarDist2D),
        ("3D", StarDist3D),
        ("Custom 2D/3D", CUSTOM_MODEL),
    ]

    @functools.lru_cache(maxsize=None)
    def get_model(model_type, model):
        if model_type == CUSTOM_MODEL:
            path = Path(model)
            path.is_dir() or _raise(FileNotFoundError(f"{path} is not a directory"))
            config = model_configs.get(
                (model_type, model), get_model_config_and_thresholds(path)[0]
            )
            model_class = StarDist2D if config["n_dim"] == 2 else StarDist3D
            return model_class(None, name=path.name, basedir=str(path.parent))
        else:
            return model_type.from_pretrained(model)

    # -------------------------------------------------------------------------

    DEFAULTS = dict(
        model_type=StarDist2D,
        model2d=models_reg[StarDist2D][0][1],
        model3d=models_reg[StarDist3D][0][1],
        norm_image=True,
        fov_image=False,
        input_scale=None,
        perc_low=1.0,
        perc_high=99.8,
        norm_axes="ZYX",
        prob_thresh=0.5,
        nms_thresh=0.4,
        output_type=Output.Both,
        n_tiles=None,
        timelapse_opts=TimelapseLabels.Unique,
        cnn_output=False,
    )

    DEFAULTS_WIDGET = DEFAULTS.copy()
    DEFAULTS_WIDGET["output_type"] = DEFAULTS_WIDGET["output_type"].value
    DEFAULTS_WIDGET["timelapse_opts"] = DEFAULTS_WIDGET["timelapse_opts"].value
    DEFAULTS_WIDGET["input_scale"] = str(DEFAULTS_WIDGET["input_scale"])
    DEFAULTS_WIDGET["n_tiles"] = str(DEFAULTS_WIDGET["n_tiles"])

    # -------------------------------------------------------------------------

    def plugin_function_generator(
        model: Union[StarDist2D, StarDist3D],
        image: napari.layers.Image,
        axes: str,
        slice_image: Union[None, Iterable[slice]] = None,
        norm_image: bool = DEFAULTS["norm_image"],
        perc_low: float = DEFAULTS["perc_low"],
        perc_high: float = DEFAULTS["perc_high"],
        input_scale: Union[None, float, Iterable[float]] = DEFAULTS["input_scale"],
        prob_thresh: float = DEFAULTS["prob_thresh"],
        nms_thresh: float = DEFAULTS["nms_thresh"],
        output_type: Output = DEFAULTS["output_type"],
        n_tiles: Union[None, int, Iterable[int]] = DEFAULTS["n_tiles"],
        norm_axes: str = DEFAULTS["norm_axes"],
        timelapse_opts: TimelapseLabels = DEFAULTS["timelapse_opts"],
        cnn_output: bool = DEFAULTS["cnn_output"],
        image_data: Any = None,
    ) -> List[LayerDataTuple]:

        if image_data is None:
            x = get_data(image)
        else:
            # TODO: check 'image_data'
            x = image_data
        axes = axes_check_and_normalize(axes, length=x.ndim)

        if slice_image is None:
            origin_in_dict = {}
        else:
            # TODO: check 'slice_image'
            x = x[slice_image]
            origin_in_dict = dict(zip(axes, tuple(s.start for s in slice_image)))

        # semantic output axes of predictions
        assert model._axes_out[-1] == "C"
        axes_out = list(model._axes_out[:-1])

        if input_scale is not None:
            if isinstance(input_scale, numbers.Number):
                # apply scaling to all spatial axes
                input_scale = tuple(input_scale if a in "XYZ" else 1 for a in axes)
            input_scale_dict = dict(zip(axes, input_scale))
            # remove potential entry for T axis (since frames processed in outer loop)
            input_scale = tuple(s for a, s in zip(axes, input_scale) if a != "T")
            # print(f"input scaling: {input_scale_dict}")
        else:
            input_scale_dict = {}

        # TODO: adjust image.scale according to shuffled axes

        # enforce dense numpy array in case we are given a dask array etc
        # -> only after (potential) field of view cropping
        x = np.asanyarray(x)

        if norm_image:
            axes_norm = axes_check_and_normalize(norm_axes)
            axes_norm = "".join(
                set(axes_norm).intersection(set(axes))
            )  # relevant axes present in input image
            assert len(axes_norm) > 0
            # always jointly normalize channels for RGB images
            if ("C" in axes and image.rgb) and ("C" not in axes_norm):
                axes_norm = axes_norm + "C"
                warnings.warn("jointly normalizing channels of RGB input image")
            ax = axes_dict(axes)
            _axis = tuple(sorted(ax[a] for a in axes_norm))
            x = normalize(x, perc_low, perc_high, axis=_axis)

        if "T" in axes:
            t = axes_dict(axes)["T"]
            if n_tiles is not None:
                # remove tiling value for time axis
                n_tiles = tuple(v for i, v in enumerate(n_tiles) if i != t)

        if n_tiles is not None and np.prod(n_tiles) > 1:
            n_tiles = tuple(n_tiles)
            num_tiles = np.prod(n_tiles)

            # this is used as tqdm replacement for predict_instances
            def progress(it, **kwargs):
                nonlocal num_tiles
                if "total" in kwargs:
                    # get number of actual tiles (which may differ from 'np.prod(n_tiles)')
                    num_tiles = kwargs["total"]
                for item in it:
                    yield item

        else:
            progress = False

        # region: prediction
        def progress_msg(val):
            # take yielded value from model._predict_instances_generator
            # and convert to progress "message" that is yielded to caller
            if isinstance(val, str):
                if val == "tile":
                    return val, (1, num_tiles)
                else:
                    return val, None
            else:
                return None

        if "T" in axes:
            x_reorder = np.moveaxis(x, t, 0)
            axes_reorder = axes.replace("T", "")
            axes_out.insert(t, "T")
            res = []
            n_frames = len(x_reorder)
            for _x in x_reorder:
                out = None
                for out in model._predict_instances_generator(
                    _x,
                    axes=axes_reorder,
                    prob_thresh=prob_thresh,
                    nms_thresh=nms_thresh,
                    n_tiles=n_tiles,
                    show_tile_progress=progress,
                    scale=input_scale,
                    sparse=(not cnn_output),
                    return_predict=cnn_output,
                ):
                    msg = progress_msg(out)
                    if msg is not None:
                        yield msg

                res.append(out)
                yield "time", (1, n_frames)
            res = tuple(zip(*res))

            if cnn_output:
                labels, t_polys = tuple(zip(*res[0]))
                cnn_out = tuple(np.stack(c, t) for c in tuple(zip(*res[1])))
            else:
                labels, t_polys = res

            labels = np.asarray(labels)

            if isinstance(model, StarDist3D):
                # TODO poly output support for 3D timelapse
                polys = None
            else:
                polys = dict(
                    coord=np.concatenate(
                        tuple(
                            np.insert(p["coord"], t, _t, axis=-2)
                            for _t, p in enumerate(t_polys)
                        ),
                        axis=0,
                    ),
                    points=np.concatenate(
                        tuple(
                            np.insert(p["points"], t, _t, axis=-1)
                            for _t, p in enumerate(t_polys)
                        ),
                        axis=0,
                    ),
                )

            if model._is_multiclass():
                labels_multiclass = np.stack(
                    [
                        create_class_labels(_y, _p["class_id"], model.config.n_classes)
                        for _y, _p in zip(labels, t_polys)
                    ],
                    axis=0,
                )
                labels_multiclass = np.moveaxis(labels_multiclass, 0, t)
            else:
                labels_multiclass = None

            # optionally match labels if we have more than one timepoint
            if len(labels) > 1:
                if timelapse_opts == TimelapseLabels.Match:
                    # match labels in consecutive frames (-> simple IoU tracking)
                    labels = group_matching_labels(labels)
                elif timelapse_opts == TimelapseLabels.Unique:
                    # make label ids unique (shift by offset)
                    offsets = np.cumsum([len(p["points"]) for p in t_polys])
                    for y, off in zip(labels[1:], offsets):
                        y[y > 0] += off
                elif timelapse_opts == TimelapseLabels.Separate:
                    # each frame processed separately (nothing to do)
                    pass
                else:
                    raise NotImplementedError(
                        f"unknown option '{timelapse_opts}' for time-lapse labels"
                    )

            labels = np.moveaxis(labels, 0, t)

            if cnn_output:
                pred = (labels, polys), cnn_out
            else:
                pred = labels, polys

        else:
            pred = None
            for pred in model._predict_instances_generator(
                x,
                axes=axes,
                prob_thresh=prob_thresh,
                nms_thresh=nms_thresh,
                n_tiles=n_tiles,
                show_tile_progress=progress,
                scale=input_scale,
                sparse=(not cnn_output),
                return_predict=cnn_output,
            ):
                msg = progress_msg(pred)
                if msg is not None:
                    yield msg

            if model._is_multiclass():
                _labels = pred[0][0] if cnn_output else pred[0]
                _polys = pred[0][1] if cnn_output else pred[1]
                labels_multiclass = create_class_labels(
                    _labels, _polys["class_id"], model.config.n_classes
                )
            else:
                labels_multiclass = None
        # endregion

        # region: create output layer
        layer_axes_from = "".join(axes_out)
        layer_axes_to = axes.replace("C", "") if image.rgb else axes

        # determine scale for output axes
        scale_in_dict = dict(zip(axes, image.scale))
        scale_out = [scale_in_dict.get(a, 1.0) for a in axes_out]
        origin_out = [origin_in_dict.get(a, 0) for a in axes_out]

        # constructing the actual napari layers
        layers = []
        if cnn_output:
            (labels, polys), cnn_out = pred
            prob, dist = cnn_out[:2]
            dist = np.moveaxis(dist, -1, 0)

            assert len(model.config.grid) == len(model.config.axes) - 1
            grid_dict = dict(zip(model.config.axes.replace("C", ""), model.config.grid))
            # scale output axes to match input axes
            _scale = [
                s * grid_dict.get(a, 1) / input_scale_dict.get(a, 1)
                for a, s in zip(axes_out, scale_out)
            ]
            # small translation correction if grid > 1 (since napari centers objects)
            # TODO: this doesn't look correct
            _translate = [
                o + 0.5 * (grid_dict.get(a, 1) / input_scale_dict.get(a, 1) - s)
                for a, s, o in zip(axes_out, scale_out, origin_out)
            ]

            layers.append(
                move_layer_axes(
                    (
                        dist,
                        dict(
                            name="StarDist distances",
                            scale=[1] + _scale,
                            translate=[0] + _translate,
                        ),
                        "image",
                    ),
                    "C" + layer_axes_from,
                    layer_axes_to if "C" in layer_axes_to else "C" + layer_axes_to,
                )
            )
            layers.append(
                move_layer_axes(
                    (
                        prob,
                        dict(
                            name="StarDist probability",
                            scale=_scale,
                            translate=_translate,
                        ),
                        "image",
                    ),
                    layer_axes_from,
                    layer_axes_to,
                )
            )

            if model._is_multiclass():
                prob_class = np.moveaxis(cnn_out[2], -1, 0)
                layers.append(
                    move_layer_axes(
                        (
                            prob_class,
                            dict(
                                name="StarDist class probabilities",
                                scale=[1] + _scale,
                                translate=[0] + _translate,
                            ),
                            "image",
                        ),
                        "C" + layer_axes_from,
                        layer_axes_to if "C" in layer_axes_to else "C" + layer_axes_to,
                    )
                )
        else:
            labels, polys = pred

        if output_type in (Output.Labels, Output.Both):

            if model._is_multiclass():
                # TODO: class labels could be treated like instance labels, i.e. can be shown as label images or polygons / polyhedra,
                #       or the class labels could be merged with the instance labels, e.g. using a different class-associated color per polygon

                layers.append(
                    move_layer_axes(
                        (
                            labels_multiclass,
                            dict(
                                name="StarDist class labels",
                                visible=False,
                                scale=scale_out,
                                opacity=0.5,
                                translate=origin_out,
                            ),
                            "labels",
                        ),
                        layer_axes_from,
                        layer_axes_to,
                    )
                )

            layers.append(
                move_layer_axes(
                    (
                        labels,
                        dict(
                            name="StarDist labels",
                            scale=scale_out,
                            opacity=0.5,
                            translate=origin_out,
                        ),
                        "labels",
                    ),
                    layer_axes_from,
                    layer_axes_to,
                )
            )

        if output_type in (Output.Polys, Output.Both):

            if isinstance(model, StarDist3D):
                if "T" in axes:
                    raise NotImplementedError("Polyhedra output for 3D timelapse")

                n_objects = len(polys["points"])
                surface = surface_from_polys(polys)
                layers.append(
                    move_layer_axes(
                        (
                            surface,
                            dict(
                                name="StarDist polyhedra",
                                contrast_limits=(0, surface[-1].max()),
                                scale=scale_out,
                                colormap=label_colormap(n_objects),
                                translate=origin_out,
                            ),
                            "surface",
                        ),
                        layer_axes_from,
                        layer_axes_to,
                    )
                )
            else:
                # TODO: coordinates correct or need offset (0.5 or so)?
                shapes = np.moveaxis(polys["coord"], -1, -2)
                layers.append(
                    move_layer_axes(
                        (
                            shapes,
                            dict(
                                name="StarDist polygons",
                                shape_type="polygon",
                                scale=scale_out,
                                edge_width=0.75,
                                edge_color="yellow",
                                face_color=[0, 0, 0, 0],
                                translate=origin_out,
                            ),
                            "shapes",
                        ),
                        layer_axes_from,
                        layer_axes_to,
                    )
                )

        # endregion
        yield layers

    @functools.wraps(plugin_function_generator)
    def plugin_function(*args, **kwargs):
        # convention: last "yield" is the actual output that would have
        #             been "return"ed if this was a regular function
        r = None
        for r in plugin_function_generator(*args, **kwargs):
            pass
        return r

    # -------------------------------------------------------------------------

    logo = abspath(__file__, "resources/stardist_logo_napari.png")

    @magicgui(
        label_head=dict(
            widget_type="Label", label=f'<h1><img src="{logo}">StarDist</h1>'
        ),
        image=dict(label="Input Image"),
        axes=dict(widget_type="LineEdit", label="Image Axes"),
        fov_image=dict(
            widget_type="CheckBox",
            text="Predict on field of view (only for 2D models in 2D view)",
            value=DEFAULTS_WIDGET["fov_image"],
        ),
        label_nn=dict(widget_type="Label", label="<br><b>Neural Network Prediction:</b>"),
        model_type=dict(
            widget_type="RadioButtons",
            label="Model Type",
            orientation="horizontal",
            choices=model_type_choices,
            value=DEFAULTS_WIDGET["model_type"],
        ),
        model2d=dict(
            widget_type="ComboBox",
            visible=False,
            label="Pre-trained Model",
            choices=models_reg[StarDist2D],
            value=DEFAULTS_WIDGET["model2d"],
        ),
        model3d=dict(
            widget_type="ComboBox",
            visible=False,
            label="Pre-trained Model",
            choices=models_reg[StarDist3D],
            value=DEFAULTS_WIDGET["model3d"],
        ),
        model_folder=dict(
            widget_type="FileEdit",
            visible=False,
            label="Custom Model",
            mode="d",
        ),
        model_axes=dict(widget_type="LineEdit", label="Model Axes", value=""),
        norm_image=dict(
            widget_type="CheckBox",
            text="Normalize Image",
            value=DEFAULTS_WIDGET["norm_image"],
        ),
        label_nms=dict(widget_type="Label", label="<br><b>NMS Postprocessing:</b>"),
        perc_low=dict(
            widget_type="FloatSpinBox",
            label="Percentile low",
            min=0.0,
            max=100.0,
            step=0.1,
            value=DEFAULTS_WIDGET["perc_low"],
        ),
        perc_high=dict(
            widget_type="FloatSpinBox",
            label="Percentile high",
            min=0.0,
            max=100.0,
            step=0.1,
            value=DEFAULTS_WIDGET["perc_high"],
        ),
        input_scale=dict(
            widget_type="LiteralEvalLineEdit",
            label="Input image scaling",
            value=DEFAULTS_WIDGET["input_scale"],
        ),
        norm_axes=dict(
            widget_type="LineEdit",
            label="Normalization Axes",
            value=DEFAULTS_WIDGET["norm_axes"],
        ),
        prob_thresh=dict(
            widget_type="FloatSpinBox",
            label="Probability/Score Threshold",
            min=0.0,
            max=1.0,
            step=0.05,
            value=DEFAULTS_WIDGET["prob_thresh"],
        ),
        nms_thresh=dict(
            widget_type="FloatSpinBox",
            label="Overlap Threshold",
            min=0.0,
            max=1.0,
            step=0.05,
            value=DEFAULTS_WIDGET["nms_thresh"],
        ),
        output_type=dict(
            widget_type="ComboBox",
            label="Output Type",
            choices=[e.value for e in Output],
            value=DEFAULTS_WIDGET["output_type"],
        ),
        label_adv=dict(widget_type="Label", label="<br><b>Advanced Options:</b>"),
        n_tiles=dict(
            widget_type="LiteralEvalLineEdit",
            label="Number of Tiles",
            value=DEFAULTS_WIDGET["n_tiles"],
        ),
        timelapse_opts=dict(
            widget_type="ComboBox",
            label="Time-lapse Labels ",
            choices=[e.value for e in TimelapseLabels],
            value=DEFAULTS_WIDGET["timelapse_opts"],
        ),
        cnn_output=dict(
            widget_type="CheckBox",
            text="Show CNN Output",
            value=DEFAULTS_WIDGET["cnn_output"],
        ),
        set_thresholds=dict(
            widget_type="PushButton",
            text="Set optimized postprocessing thresholds (for selected model)",
        ),
        defaults_button=dict(widget_type="PushButton", text="Restore Defaults"),
        layout="vertical",
        persist=not NOPERSIST,
        call_button=True,
    )
    def plugin(
        viewer: Union[napari.Viewer, None],
        label_head,
        image: napari.layers.Image,
        axes,
        fov_image,
        label_nn,
        model_type,
        model2d,
        model3d,
        model_folder,
        model_axes,
        norm_image,
        perc_low,
        perc_high,
        input_scale,
        label_nms,
        prob_thresh,
        nms_thresh,
        output_type,
        label_adv,
        n_tiles,
        norm_axes,
        timelapse_opts,
        cnn_output,
        set_thresholds,
        defaults_button,
    ) -> _Future[List[LayerDataTuple]]:

        model = get_model(
            model_type,
            {
                StarDist2D: model2d,
                StarDist3D: model3d,
                CUSTOM_MODEL: model_folder,
            }[model_type],
        )

        x = get_data(image)
        axes = axes_check_and_normalize(axes, length=x.ndim)

        # axes and x correspond to the original (and immutable) order of image dimensions
        # -> i.e. not affected by changing the viewer dimensions ordering, etc.

        # region: field of view
        if fov_image and model.config.n_dim == 2 and viewer.dims.ndisplay == 2:
            # it's all a big mess based on shaky assumptions...
            if viewer is None:
                raise RuntimeError("viewer is None")
            if image.rgb and axes[-1] != "C":
                raise RuntimeError("rgb image must have channels as last axis/dimension")

            def get_slice_not_displayed(vdim, idim):
                # vdim: dimension index wrt. viewer
                # idim: dimension index wrt. image
                if axes[idim] == "T":
                    # if timelapse, return visible/selected frame
                    return slice(
                        viewer.dims.current_step[vdim],
                        1 + viewer.dims.current_step[vdim],
                    )
                elif axes[idim] == "C":
                    # if channel, return entire dimension
                    return slice(0, x.shape[idim])
                else:
                    return None

            corner_pixels = (
                corner_pixels_multiscale(image)
                if image.multiscale
                else image.corner_pixels
            )
            n_corners = corner_pixels.shape[1]
            n_corners <= x.ndim or _raise(RuntimeError("assumption violated"))

            # map viewer dimension index to image dimension index
            n_dims = x.ndim - (1 if image.rgb else 0)
            viewer_dim_to_image_dim = dict(
                zip(np.arange(viewer.dims.ndim)[-n_dims:], range(n_dims))
            )
            # map viewer dimension index to corner pixel
            viewer_dim_to_corner = dict(
                zip(
                    np.arange(viewer.dims.ndim)[-n_corners:],
                    zip(corner_pixels[0], corner_pixels[1]),
                )
            )
            # if DEBUG:
            #     print(f"{corner_pixels = }")
            #     print(f"{viewer_dim_to_image_dim = }")
            #     print(f"{viewer_dim_to_corner = }")
            #     print(f"{viewer.dims.displayed = }")

            sl = [None] * x.ndim
            for vdim in range(viewer.dims.ndim):
                idim = viewer_dim_to_image_dim.get(vdim)
                c = viewer_dim_to_corner.get(vdim)
                # DEBUG and print(
                #     f"{vdim=}, {idim=}{f'/{axes[idim]}' if idim is not None else ''}, {c=}"
                # )
                if c is not None:
                    if vdim in viewer.dims.displayed:
                        fr, to = c
                        sl[idim] = None if fr == to else slice(fr, to)
                    else:
                        sl[idim] = get_slice_not_displayed(vdim, idim)
                else:
                    # assert vdim in viewer.dims.not_displayed
                    if idim is not None:
                        sl[idim] = get_slice_not_displayed(vdim, idim)

            if image.rgb:
                idim = x.ndim - 1
                # set channel slice here, since channel of rgb image not part of viewer dimensions
                assert sl[idim] is None and axes[idim] == "C"
                sl[idim] = get_slice_not_displayed(None, idim)

            sl = tuple(sl)

            invalid_axes = "".join(a for s, a in zip(sl, axes) if s is None)
            if len(invalid_axes) > 0:
                raise ValueError(f"Invalid field of view range for axes {invalid_axes}")

            if DEBUG:
                for sh, s, a in zip(x.shape, sl, axes):
                    print(f"{a}({sh}): {s}")
        else:
            sl = None
        # endregion

        # region: progress bars
        uses_tiling = n_tiles is not None and np.prod(n_tiles) > 1
        pbar_time = progress(total=0, desc="StarDist Time-lapse") if "T" in axes else None

        def make_pbar(desc="StarDist Prediction"):
            return progress(
                total=(np.prod(n_tiles) if uses_tiling else 0),
                desc=desc,
                nest_under=pbar_time,
            )

        def set_pbar(pbar, total=None, desc=None):
            if isinstance(desc, str) and pbar.desc != f"{desc}: ":
                pbar.set_description(desc)
            if isinstance(total, int) and pbar.total != total:
                pbar.total = total

        pbar = make_pbar()

        def show_activity_dock(state=True):
            # show/hide activity dock if there is actual progress to see
            if uses_tiling or "T" in axes:
                try:
                    with warnings.catch_warnings():
                        # suppress FutureWarning for now: https://github.com/napari/napari/issues/4598
                        warnings.simplefilter(action="ignore", category=FutureWarning)
                        viewer.window._status_bar._toggle_activity_dock(state)
                except AttributeError:
                    print(f"show_activity_dock failed")

        def progress_update(value):
            nonlocal pbar
            m, args = value
            if m == "time":
                i, n = args
                set_pbar(pbar_time, total=n)
                pbar_time.update(i)
            elif m == "predict":
                if uses_tiling:
                    pbar.close()
                    pbar = make_pbar()
                else:
                    set_pbar(pbar, desc="StarDist Prediction", total=0)
            elif m == "tile":
                i, n = args
                set_pbar(pbar, total=n)
                pbar.update(i)
            elif m == "nms":
                set_pbar(pbar, desc="StarDist Postprocessing", total=0)

        def progress_close():
            pbar.close()
            if pbar_time is not None:
                pbar_time.close()
            show_activity_dock(False)

        # endregion

        # TODO: cancel button (cf. https://napari.zulipchat.com/#narrow/stream/309872-plugins/topic/Testing.20plugins/near/284578152)

        future = Future()

        computation_generator = plugin_function_generator(
            model=model,
            image=image,
            axes=axes,
            slice_image=sl,
            norm_image=norm_image,
            perc_low=perc_low,
            perc_high=perc_high,
            input_scale=input_scale,
            prob_thresh=prob_thresh,
            nms_thresh=nms_thresh,
            output_type=get_enum_member(Output, output_type),
            n_tiles=n_tiles,
            norm_axes=norm_axes,
            timelapse_opts=get_enum_member(TimelapseLabels, timelapse_opts),
            cnn_output=cnn_output,
            image_data=x,
        )

        def is_progress_msg(r):
            return isinstance(r, tuple) and len(r) > 1 and isinstance(r[0], str)

        if NOTHREADS:

            show_activity_dock()
            r = None
            for r in computation_generator:
                if is_progress_msg(r):
                    progress_update(r)
            future.set_result(r)
            progress_close()

        else:

            @thread_worker(
                connect={
                    "yielded": progress_update,
                    "returned": progress_close,
                    "started": show_activity_dock,
                },
                start_thread=False,
            )
            def run():
                r = None
                for r in computation_generator:
                    if is_progress_msg(r):
                        yield r
                return r

            worker = run()
            worker.returned.connect(future.set_result)
            worker.start()

        return future

    # -------------------------------------------------------------------------

    # region UI interaction

    # don't want to load persisted values for these widgets
    plugin.axes.value = ""
    plugin.n_tiles.value = DEFAULTS["n_tiles"]
    plugin.input_scale.value = DEFAULTS["input_scale"]
    plugin.label_head.value = '<small>Star-convex object detection for 2D and 3D images.<br>If you are using this in your research please <a href="https://github.com/stardist/stardist#how-to-cite" style="color:gray;">cite us</a>.</small><br><br><tt><a href="https://stardist.net" style="color:gray;">https://stardist.net</a></tt>'

    # make labels prettier (https://doc.qt.io/qt-5/qsizepolicy.html#Policy-enum)
    for w in (plugin.label_head, plugin.label_nn, plugin.label_nms, plugin.label_adv):
        w.native.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Fixed)

    # -------------------------------------------------------------------------

    widget_for_modeltype = {
        StarDist2D: plugin.model2d,
        StarDist3D: plugin.model3d,
        CUSTOM_MODEL: plugin.model_folder,
    }

    def widgets_inactive(*widgets, active):
        for widget in widgets:
            widget.visible = active
            # widget.native.setStyleSheet("" if active else "text-decoration: line-through")

    def widgets_valid(*widgets, valid):
        for widget in widgets:
            widget.native.setStyleSheet("" if valid else "background-color: lightcoral")

    class Updater:
        """Class that allows for joint validation of different parameters.

        update = Updater()
        update('param', valid=True, args=some_value)

        To add a new plugin field:
            * put single field validation logic inside the plugin fields change handler
            * add update() call inside change handler
            * add _param() inner function inside Updater._update() and call it therein

        """

        def __init__(self, debug=DEBUG):
            from types import SimpleNamespace

            self.debug = debug
            self.valid = SimpleNamespace(
                **{
                    k: False
                    for k in (
                        "image_axes",
                        "model",
                        "n_tiles",
                        "norm_axes",
                        "input_scale",
                    )
                }
            )
            self.args = SimpleNamespace()
            self.viewer = None

        def __call__(self, k, valid, args=None):
            assert k in vars(self.valid)
            setattr(self.valid, k, bool(valid))
            setattr(self.args, k, args)
            self._update()

        def help(self, msg):
            if self.viewer is not None:
                self.viewer.help = msg
            elif len(str(msg)) > 0:
                print(f"HELP: {msg}")

        def _update(self):
            def _fov():
                nonlocal model_selected
                config = model_configs.get(model_selected, {})
                model_dim = config.get("n_dim")
                active = (
                    self.viewer is not None
                    and self.viewer.dims.ndisplay == 2
                    and model_dim == 2
                )
                widgets_inactive(plugin.fov_image, active=active)

            # try to get a hold of the viewer (can be None when plugin starts)
            if self.viewer is None:
                # TODO: when is this not safe to do and will hang forever?
                # while plugin.viewer.value is None:
                #     time.sleep(0.01)
                if plugin.viewer.value is not None:
                    self.viewer = plugin.viewer.value
                    if DEBUG:
                        print("GOT viewer\n")

                    self.viewer.dims.events.ndisplay.connect(_fov)

                    @self.viewer.layers.events.removed.connect
                    def _layer_removed(event):
                        layers_remaining = event.source
                        if len(layers_remaining) == 0:
                            plugin.image.tooltip = ""
                            plugin.axes.value = ""
                            plugin.n_tiles.value = "None"
                            plugin.input_scale.value = "None"

            def _model(valid):
                widgets_valid(
                    plugin.model2d,
                    plugin.model3d,
                    plugin.model_folder.line_edit,
                    valid=valid,
                )
                _fov()
                if valid:
                    config = self.args.model
                    axes = config.get("axes", "ZYXC"[-len(config["net_input_shape"]) :])
                    if "T" in axes:
                        raise RuntimeError("model with axis 'T' not supported")
                    plugin.model_axes.value = axes.replace(
                        "C", f"C[{config['n_channel_in']}]"
                    )
                    plugin.model_folder.line_edit.tooltip = ""
                    return axes, config
                else:
                    plugin.model_axes.value = ""
                    plugin.model_folder.line_edit.tooltip = "Invalid model directory"

            def _image_axes(valid):
                axes, image, err = getattr(self.args, "image_axes", (None, None, None))
                widgets_valid(
                    plugin.axes,
                    valid=(valid or (image is None and (axes is None or len(axes) == 0))),
                )
                if (
                    SLOW_SHAPE_LAYER
                    and valid
                    and "T" in axes
                    and plugin.output_type.value
                    in (Output.Polys.value, Output.Both.value)
                ):
                    plugin.output_type.native.setStyleSheet("background-color: orange")
                    plugin.output_type.tooltip = (
                        "Displaying many polygons/polyhedra can be very slow."
                    )
                else:
                    plugin.output_type.native.setStyleSheet("")
                    plugin.output_type.tooltip = ""
                if valid:
                    shape = get_data(image).shape
                    plugin.axes.tooltip = "\n".join(
                        [f"{a} = {s}" for a, s in zip(axes, shape)]
                    )
                    return axes, image
                else:
                    if err is not None:
                        err = str(err)
                        err = err[:-1] if err.endswith(".") else err
                        plugin.axes.tooltip = err
                        # warnings.warn(err) # alternative to tooltip (gui doesn't show up in ipython)
                    else:
                        plugin.axes.tooltip = ""

            def _norm_axes(valid):
                norm_axes, err = getattr(self.args, "norm_axes", (None, None))
                widgets_valid(plugin.norm_axes, valid=valid)
                if valid:
                    plugin.norm_axes.tooltip = f"Axes to jointly normalize (if present in selected input image). Note: channels of RGB images are always normalized together."
                    return norm_axes
                else:
                    if err is not None:
                        err = str(err)
                        err = err[:-1] if err.endswith(".") else err
                        plugin.norm_axes.tooltip = err
                        # warnings.warn(err) # alternative to tooltip (gui doesn't show up in ipython)
                    else:
                        plugin.norm_axes.tooltip = ""

            def _n_tiles(valid):
                n_tiles, image, err = getattr(self.args, "n_tiles", (None, None, None))
                widgets_valid(plugin.n_tiles, valid=(valid or image is None))
                if valid:
                    plugin.n_tiles.tooltip = (
                        "no tiling"
                        if n_tiles is None
                        else "\n".join(
                            [f"{t}: {s}" for t, s in zip(n_tiles, get_data(image).shape)]
                        )
                    )
                    return n_tiles
                else:
                    msg = str(err) if err is not None else ""
                    plugin.n_tiles.tooltip = msg

            def _no_tiling_for_axis(axes_image, n_tiles, axis):
                if n_tiles is not None and axis in axes_image:
                    return n_tiles[axes_dict(axes_image)[axis]] == 1
                return True

            def _input_scale(valid):
                input_scale, image, err = getattr(
                    self.args, "input_scale", (None, None, None)
                )
                widgets_valid(plugin.input_scale, valid=(valid or image is None))
                if valid:
                    if input_scale is None:
                        plugin.input_scale.tooltip = "no scaling"
                    elif isinstance(input_scale, numbers.Number):
                        plugin.input_scale.tooltip = f"{input_scale} for all spatial axes"
                    else:
                        assert len(input_scale) == len(get_data(image).shape)
                        plugin.input_scale.tooltip = "\n".join(
                            [f"{s}" for s in input_scale]
                        )
                    return input_scale
                else:
                    msg = str(err) if err is not None else ""
                    plugin.input_scale.tooltip = msg

            def _input_scale_check(axes_image, input_scale):
                if input_scale is not None and not isinstance(
                    input_scale, numbers.Number
                ):
                    assert len(input_scale) == len(axes_image)
                    # s != 1 only allowed for spatial axes XYZ
                    return all(
                        s == 1 or a in "XYZ" for a, s in zip(axes_image, input_scale)
                    )
                return True

            def _restore():
                widgets_valid(plugin.image, valid=plugin.image.value is not None)

            all_valid = False
            help_msg = ""

            if (
                self.valid.image_axes
                and self.valid.n_tiles
                and self.valid.model
                and self.valid.norm_axes
                and self.valid.input_scale
            ):
                axes_image, image = _image_axes(True)
                axes_model, config = _model(True)
                axes_norm = _norm_axes(True)
                n_tiles = _n_tiles(True)
                input_scale = _input_scale(True)

                if not _no_tiling_for_axis(axes_image, n_tiles, "C"):
                    # check if image axes and n_tiles are compatible
                    widgets_valid(plugin.n_tiles, valid=False)
                    err = "number of tiles must be 1 for C axis"
                    plugin.n_tiles.tooltip = err
                    _restore()
                elif not _no_tiling_for_axis(axes_image, n_tiles, "T"):
                    # check if image axes and n_tiles are compatible
                    widgets_valid(plugin.n_tiles, valid=False)
                    err = "number of tiles must be 1 for T axis"
                    plugin.n_tiles.tooltip = err
                    _restore()
                elif not _input_scale_check(axes_image, input_scale):
                    # check if image axes and input_scale are compatible
                    widgets_valid(plugin.input_scale, valid=False)
                    _violations = ", ".join(
                        a
                        for a, s in zip(axes_image, input_scale)
                        if not (s == 1 or a in "XYZ")
                    )
                    err = f"values for non-spatial axes ({_violations}) must be 1"
                    plugin.input_scale.tooltip = err
                    _restore()
                elif set(axes_norm).isdisjoint(set(axes_image)):
                    # check if image axes and normalization axes are compatible
                    widgets_valid(plugin.norm_axes, valid=False)
                    err = f"Image axes ({axes_image}) must contain at least one of the normalization axes ({', '.join(axes_norm)})"
                    plugin.norm_axes.tooltip = err
                    _restore()
                elif (
                    "T" in axes_image
                    and config.get("n_dim") == 3
                    and plugin.output_type.value
                    in (Output.Polys.value, Output.Both.value)
                ):
                    # not supported
                    widgets_valid(plugin.output_type, valid=False)
                    plugin.output_type.tooltip = (
                        "Polyhedra output currently not supported for 3D timelapse data"
                    )
                    _restore()
                else:
                    # tooltip for input_scale
                    if isinstance(input_scale, numbers.Number):
                        plugin.input_scale.tooltip = "\n".join(
                            [
                                f'{a} = {input_scale if a in "XYZ" else 1}'
                                for a in axes_image
                            ]
                        )
                    elif input_scale is not None:
                        plugin.input_scale.tooltip = "\n".join(
                            [f"{a} = {s}" for a, s in zip(axes_image, input_scale)]
                        )

                    # check if image and model are compatible
                    ch_model = config["n_channel_in"]
                    ch_image = (
                        get_data(image).shape[axes_dict(axes_image)["C"]]
                        if "C" in axes_image
                        else 1
                    )
                    all_valid = (
                        set(axes_model.replace("C", ""))
                        == set(axes_image.replace("C", "").replace("T", ""))
                        and ch_model == ch_image
                    )

                    widgets_valid(
                        plugin.image,
                        plugin.model2d,
                        plugin.model3d,
                        plugin.model_folder.line_edit,
                        valid=all_valid,
                    )
                    if all_valid:
                        help_msg = ""
                    else:
                        help_msg = f'Model with axes {axes_model.replace("C", f"C[{ch_model}]")} and image with axes {axes_image.replace("C", f"C[{ch_image}]")} not compatible'
            else:
                _image_axes(self.valid.image_axes)
                _norm_axes(self.valid.norm_axes)
                _n_tiles(self.valid.n_tiles)
                _input_scale(self.valid.input_scale)
                _model(self.valid.model)
                _restore()

            self.help(help_msg)
            plugin.call_button.enabled = all_valid
            # widgets_valid(plugin.call_button, valid=all_valid)
            if self.debug:
                print(
                    f"UPDATER {all_valid}: {', '.join(f'{k}={v}' for k, v in vars(self.valid).items())}\n"
                )

    update = Updater()

    def select_model(key):
        nonlocal model_selected
        model_selected = key
        config = model_configs.get(key)
        update("model", config is not None, config)

    # -------------------------------------------------------------------------

    # hide percentile selection if normalization turned off
    @change_handler(plugin.norm_image)
    def _norm_image_change(active: bool):
        widgets_inactive(
            plugin.perc_low, plugin.perc_high, plugin.norm_axes, active=active
        )

    # ensure that percentile low < percentile high
    @change_handler(plugin.perc_low)
    def _perc_low_change(_value):
        plugin.perc_high.value = max(plugin.perc_low.value + 0.01, plugin.perc_high.value)

    @change_handler(plugin.perc_high)
    def _perc_high_change(_value):
        plugin.perc_low.value = min(plugin.perc_low.value, plugin.perc_high.value - 0.01)

    @change_handler(plugin.norm_axes)
    def _norm_axes_change(value: str):
        if value != value.upper():
            with plugin.axes.changed.blocked():
                plugin.norm_axes.value = value.upper()
        try:
            axes = axes_check_and_normalize(value, disallowed="S")
            if len(axes) >= 1:
                update("norm_axes", True, (axes, None))
            else:
                update("norm_axes", False, (axes, "Cannot be empty"))
        except ValueError as err:
            update("norm_axes", False, (value, err))

    # -------------------------------------------------------------------------

    # RadioButtons widget triggers a change event initially (either when 'value' is set in constructor, or via 'persist')
    # TODO: seems to be triggered too when a layer is added or removed (why?)
    @change_handler(plugin.model_type, init=False)
    def _model_type_change(model_type: Union[str, type]):
        selected = widget_for_modeltype[model_type]
        for w in set((plugin.model2d, plugin.model3d, plugin.model_folder)) - {selected}:
            w.hide()
        selected.show()
        # trigger _model_change
        selected.changed(selected.value)

    # show/hide model folder picker
    # load config/thresholds for selected pretrained model
    # -> triggered by _model_type_change
    @change_handler(plugin.model2d, plugin.model3d, init=False)
    def _model_change(model_name: str):
        model_class = StarDist2D if Signal.sender() is plugin.model2d else StarDist3D
        key = model_class, model_name

        if key not in model_configs:

            def process_model_folder(path):
                try:
                    _config, _thresholds = get_model_config_and_thresholds(path)
                    model_configs[key] = _config
                    if _thresholds is not None:
                        model_threshs[key] = _thresholds
                finally:
                    select_model(key)

            progress_kwargs = dict(total=0, desc="Obtaining model")

            if NOTHREADS:

                def obtain_model_folder():
                    with progress(**progress_kwargs):
                        process_model_folder(get_model_folder(*key))

            else:

                @thread_worker(
                    progress=progress_kwargs,
                    connect={"returned": process_model_folder},
                    start_thread=True,
                )
                def obtain_model_folder():
                    return get_model_folder(*key)

            plugin.call_button.enabled = False
            obtain_model_folder()
        else:
            select_model(key)

    # load config/thresholds from custom model path
    # -> triggered by _model_type_change
    # note: will be triggered at every keystroke (when typing the path)
    @change_handler(plugin.model_folder, init=False)
    def _model_folder_change(_path: str):
        path = Path(_path)
        key = CUSTOM_MODEL, path
        try:
            if not path.is_dir():
                return
            model_configs[key] = load_json(str(path / "config.json"))
            model_threshs[key] = load_json(str(path / "thresholds.json"))
        except FileNotFoundError:
            pass
        finally:
            select_model(key)

    # -------------------------------------------------------------------------

    # -> triggered by napari (if there are any open images on plugin launch)
    #    -> not true anymore since napari 0.4.17
    @change_handler(plugin.image, init=False)
    def _image_change(image: napari.layers.Image):
        shape = get_data(image).shape
        ndim = len(shape)
        plugin.image.tooltip = f"Shape: {shape}"

        # dimensionality of selected model: 2, 3, or None (unknown)
        ndim_model = None
        if plugin.model_type.value == StarDist2D:
            ndim_model = 2
        elif plugin.model_type.value == StarDist3D:
            ndim_model = 3
        else:
            if model_selected in model_configs:
                config = model_configs[model_selected]
                ndim_model = config.get("n_dim")

        # TODO: guess images axes better...
        axes = None
        if ndim == 2:
            axes = "YX"
        elif ndim == 3:
            axes = "YXC" if image.rgb else ("ZYX" if ndim_model == 3 else "TYX")
        elif ndim == 4:
            axes = ("ZYXC" if ndim_model == 3 else "TYXC") if image.rgb else "TZYX"
        else:
            raise NotImplementedError()

        if axes == plugin.axes.value:
            # make sure to trigger a changed event, even if value didn't actually change
            plugin.axes.changed(axes)
        else:
            plugin.axes.value = axes

        plugin.n_tiles.changed(plugin.n_tiles.value)
        plugin.input_scale.changed(plugin.input_scale.value)
        plugin.norm_axes.changed(plugin.norm_axes.value)

    # -> triggered by _image_change
    @change_handler(plugin.axes, init=False)
    def _axes_change(value: str):
        if value != value.upper():
            with plugin.axes.changed.blocked():
                plugin.axes.value = value.upper()
        image = plugin.image.value
        axes = ""
        try:
            image is not None or _raise(ValueError("no image selected"))
            axes = axes_check_and_normalize(
                value, length=get_data(image).ndim, disallowed="S"
            )
            update("image_axes", True, (axes, image, None))
        except ValueError as err:
            update("image_axes", False, (value, image, err))
        finally:
            widgets_inactive(plugin.timelapse_opts, active=("T" in axes))

    # -> triggered by _image_change
    @change_handler(plugin.n_tiles, init=False)
    def _n_tiles_change(_value):
        image = plugin.image.value
        try:
            image is not None or _raise(ValueError("no image selected"))
            value = plugin.n_tiles.get_value()
            if value is None:
                update("n_tiles", True, (None, image, None))
                return
            shape = get_data(image).shape
            try:
                value = tuple(value)
                len(value) == len(shape) or _raise(TypeError())
            except TypeError:
                raise ValueError(f"must be a tuple/list of length {len(shape)}")
            if not all(isinstance(t, int) and t >= 1 for t in value):
                raise ValueError(f"each value must be an integer >= 1")
            update("n_tiles", True, (value, image, None))
        except (ValueError, SyntaxError) as err:
            update("n_tiles", False, (None, image, err))

    # -> triggered by _image_change
    @change_handler(plugin.input_scale, init=False)
    def _input_scale_change(_value):
        image = plugin.image.value
        try:
            image is not None or _raise(ValueError("no image selected"))
            value = plugin.input_scale.get_value()
            if value is None:
                update("input_scale", True, (None, image, None))
                return
            shape = get_data(image).shape
            try:
                isinstance(value, numbers.Number) or len(tuple(value)) == len(
                    shape
                ) or _raise(TypeError())
            except TypeError:
                raise ValueError(
                    f"must be a scalar value or tuple/list of length {len(shape)}"
                )
            if not all(
                isinstance(t, numbers.Number) and t > 0
                for t in ((value,) if isinstance(value, numbers.Number) else value)
            ):
                raise ValueError(f"each value must be a number > 0")
            update("input_scale", True, (value, image, None))
        except (ValueError, SyntaxError) as err:
            update("input_scale", False, (None, image, err))

    # -------------------------------------------------------------------------

    # set thresholds to optimized values for chosen model
    @change_handler(plugin.set_thresholds, init=False)
    def _set_thresholds():
        if model_selected in model_threshs:
            thresholds = model_threshs[model_selected]
            plugin.nms_thresh.value = thresholds["nms"]
            plugin.prob_thresh.value = thresholds["prob"]

    # output type changed
    @change_handler(plugin.output_type, init=False)
    def _output_type_change(_value):
        update._update()

    # restore defaults
    @change_handler(plugin.defaults_button, init=False)
    def restore_defaults():
        for k, v in DEFAULTS_WIDGET.items():
            getattr(plugin, k).value = v

    # -------------------------------------------------------------------------

    # allow some widgets to shrink because their size depends on user input
    # int() -> cf. https://github.com/stardist/stardist-napari/actions/runs/3291915518/jobs/5426658873
    w = int(240)
    plugin.image.native.setMinimumWidth(w)
    plugin.model2d.native.setMinimumWidth(w)
    plugin.model3d.native.setMinimumWidth(w)
    plugin.timelapse_opts.native.setMinimumWidth(w)

    plugin.label_head.native.setOpenExternalLinks(True)
    # make reset button smaller
    # plugin.defaults_button.native.setMaximumWidth(150)

    # plugin.model_axes.native.setReadOnly(True)
    plugin.model_axes.enabled = False

    # reduce vertical spacing and fontsize
    layout = plugin.native.layout()
    layout.setSpacing(5)
    # for i in range(len(layout)) :
    #     w = layout.itemAt(i).widget()
    #     w.setStyleSheet("""QWidget {font-size:11px;}""")

    # push 'call_button' to bottom
    layout.insertStretch(layout.count() - 1)

    # for i in range(layout.count()):
    #     print(i, layout.itemAt(i).widget())

    # necessary since napari 0.4.17 to trigger change event
    # if there are existing image layers on plugin launch
    if plugin.image.value is not None:
        plugin.image.changed(plugin.image.value)

    if DEBUG:
        print("BUILT plugin\n")

    return plugin, plugin_function

    # endregion


def plugin_dock_widget():
    return _plugin_wrapper()[0]


def plugin_function():
    return _plugin_wrapper()[1]
