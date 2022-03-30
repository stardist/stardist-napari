"""
TODO:
- ability to cancel running stardist computation
- run only on field of view
  - https://forum.image.sc/t/how-could-i-get-the-viewed-coordinates/49709
  - https://napari.zulipchat.com/#narrow/stream/212875-general/topic/corner.20pixels.20and.20dims.20displayed
- add general tooltip help/info messages
- option to use CPU or GPU, limit tensorflow GPU memory ('allow_growth'?)
- try progress bar via @thread_workers
"""

import functools
import numbers
import os
import time
from enum import Enum
from pathlib import Path
from typing import List, Union
from warnings import warn

import napari
import numpy as np
from magicgui import magicgui
from magicgui import widgets as mw
from magicgui.application import use_app
from napari.qt.threading import thread_worker
from napari.utils.colormaps import label_colormap
from psygnal import Signal
from qtpy.QtWidgets import QSizePolicy


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


def plugin_wrapper():
    # delay imports until plugin is requested by user
    # -> especially those importing tensorflow (csbdeep.models.*, stardist.models.*)
    from csbdeep.models.pretrained import get_model_folder, get_registered_models
    from csbdeep.utils import (
        _raise,
        axes_check_and_normalize,
        axes_dict,
        load_json,
        normalize,
    )
    from stardist.matching import group_matching_labels
    from stardist.models import StarDist2D, StarDist3D
    from stardist.utils import abspath

    DEBUG = os.environ.get("STARDIST_NAPARI_DEBUG", "").lower() in (
        "y",
        "yes",
        "t",
        "true",
        "on",
        "1",
    )

    def get_data(image):
        image = image.data[0] if image.multiscale else image.data
        # enforce dense numpy array in case we are given a dask array etc
        return np.asarray(image)

    def change_handler(*widgets, init=True, debug=DEBUG):
        def decorator_change_handler(handler):
            @functools.wraps(handler)
            def wrapper(*args):
                source = Signal.sender()
                emitter = Signal.current_emitter()
                if debug:
                    # print(f"{emitter}: {source} = {args!r}")
                    print(f"{str(emitter.name).upper()}: {source.name} = {args!r}")
                return handler(*args)

            for widget in widgets:
                widget.changed.connect(wrapper)
                if init:
                    widget.changed(widget.value)
            return wrapper

        return decorator_change_handler

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

    CUSTOM_MODEL = "CUSTOM_MODEL"
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
            config = model_configs[(model_type, model)]
            model_class = StarDist2D if config["n_dim"] == 2 else StarDist3D
            return model_class(None, name=path.name, basedir=str(path.parent))
        else:
            return model_type.from_pretrained(model)

    # -------------------------------------------------------------------------

    class Output(Enum):
        Labels = "Label Image"
        Polys = "Polygons / Polyhedra"
        Both = "Both"

    output_choices = [Output.Labels.value, Output.Polys.value, Output.Both.value]

    class TimelapseLabels(Enum):
        Match = "Match to previous frame (via overlap)"
        Unique = "Unique through time"
        Separate = "Separate per frame (no processing)"

    timelapse_opts = [
        TimelapseLabels.Match.value,
        TimelapseLabels.Unique.value,
        TimelapseLabels.Separate.value,
    ]

    # -------------------------------------------------------------------------

    DEFAULTS = dict(
        model_type=StarDist2D,
        model2d=models_reg[StarDist2D][0][1],
        model3d=models_reg[StarDist3D][0][1],
        norm_image=True,
        input_scale="None",
        perc_low=1.0,
        perc_high=99.8,
        norm_axes="ZYX",
        prob_thresh=0.5,
        nms_thresh=0.4,
        output_type=Output.Both.value,
        n_tiles="None",
        timelapse_opts=TimelapseLabels.Unique.value,
        cnn_output=False,
    )

    # -------------------------------------------------------------------------

    logo = abspath(__file__, "resources/stardist_logo_napari.png")

    @magicgui(
        label_head=dict(
            widget_type="Label", label=f'<h1><img src="{logo}">StarDist</h1>'
        ),
        image=dict(label="Input Image"),
        axes=dict(widget_type="LineEdit", label="Image Axes"),
        label_nn=dict(widget_type="Label", label="<br><b>Neural Network Prediction:</b>"),
        model_type=dict(
            widget_type="RadioButtons",
            label="Model Type",
            orientation="horizontal",
            choices=model_type_choices,
            value=DEFAULTS["model_type"],
        ),
        model2d=dict(
            widget_type="ComboBox",
            visible=False,
            label="Pre-trained Model",
            choices=models_reg[StarDist2D],
            value=DEFAULTS["model2d"],
        ),
        model3d=dict(
            widget_type="ComboBox",
            visible=False,
            label="Pre-trained Model",
            choices=models_reg[StarDist3D],
            value=DEFAULTS["model3d"],
        ),
        model_folder=dict(
            widget_type="FileEdit", visible=False, label="Custom Model", mode="d"
        ),
        model_axes=dict(widget_type="LineEdit", label="Model Axes", value=""),
        norm_image=dict(
            widget_type="CheckBox", text="Normalize Image", value=DEFAULTS["norm_image"]
        ),
        label_nms=dict(widget_type="Label", label="<br><b>NMS Postprocessing:</b>"),
        perc_low=dict(
            widget_type="FloatSpinBox",
            label="Percentile low",
            min=0.0,
            max=100.0,
            step=0.1,
            value=DEFAULTS["perc_low"],
        ),
        perc_high=dict(
            widget_type="FloatSpinBox",
            label="Percentile high",
            min=0.0,
            max=100.0,
            step=0.1,
            value=DEFAULTS["perc_high"],
        ),
        input_scale=dict(
            widget_type="LiteralEvalLineEdit",
            label="Input image scaling",
            value=DEFAULTS["input_scale"],
        ),
        norm_axes=dict(
            widget_type="LineEdit",
            label="Normalization Axes",
            value=DEFAULTS["norm_axes"],
        ),
        prob_thresh=dict(
            widget_type="FloatSpinBox",
            label="Probability/Score Threshold",
            min=0.0,
            max=1.0,
            step=0.05,
            value=DEFAULTS["prob_thresh"],
        ),
        nms_thresh=dict(
            widget_type="FloatSpinBox",
            label="Overlap Threshold",
            min=0.0,
            max=1.0,
            step=0.05,
            value=DEFAULTS["nms_thresh"],
        ),
        output_type=dict(
            widget_type="ComboBox",
            label="Output Type",
            choices=output_choices,
            value=DEFAULTS["output_type"],
        ),
        label_adv=dict(widget_type="Label", label="<br><b>Advanced Options:</b>"),
        n_tiles=dict(
            widget_type="LiteralEvalLineEdit",
            label="Number of Tiles",
            value=DEFAULTS["n_tiles"],
        ),
        timelapse_opts=dict(
            widget_type="ComboBox",
            label="Time-lapse Labels ",
            choices=timelapse_opts,
            value=DEFAULTS["timelapse_opts"],
        ),
        cnn_output=dict(
            widget_type="CheckBox", text="Show CNN Output", value=DEFAULTS["cnn_output"]
        ),
        set_thresholds=dict(
            widget_type="PushButton",
            text="Set optimized postprocessing thresholds (for selected model)",
        ),
        defaults_button=dict(widget_type="PushButton", text="Restore Defaults"),
        progress_bar=dict(label=" ", min=0, max=0, visible=False),
        layout="vertical",
        persist=True,
        call_button=True,
    )
    def plugin(
        viewer: napari.Viewer,
        label_head,
        image: napari.layers.Image,
        axes,
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
        progress_bar: mw.ProgressBar,
    ) -> List[napari.types.LayerDataTuple]:

        model = get_model(*model_selected)
        if model._is_multiclass():
            warn("multi-class mode not supported yet, ignoring classification output")

        lkwargs = {}
        x = get_data(image)
        axes = axes_check_and_normalize(axes, length=x.ndim)

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

        if not axes.replace("T", "").startswith(model._axes_out.replace("C", "")):
            warn(
                f"output images have different axes ({model._axes_out.replace('C','')}) than input image ({axes})"
            )
            # TODO: adjust image.scale according to shuffled axes

        if norm_image:
            axes_norm = axes_check_and_normalize(norm_axes)
            axes_norm = "".join(
                set(axes_norm).intersection(set(axes))
            )  # relevant axes present in input image
            assert len(axes_norm) > 0
            # always jointly normalize channels for RGB images
            if ("C" in axes and image.rgb == True) and ("C" not in axes_norm):
                axes_norm = axes_norm + "C"
                warn("jointly normalizing channels of RGB input image")
            ax = axes_dict(axes)
            _axis = tuple(sorted(ax[a] for a in axes_norm))
            # # TODO: address joint vs. channel/time-separate normalization properly (let user choose)
            # #       also needs to be documented somewhere
            # if 'T' in axes:
            #     if 'C' not in axes or image.rgb == True:
            #          # normalize channels jointly, frames independently
            #          _axis = tuple(i for i in range(x.ndim) if i not in (ax['T'],))
            #     else:
            #         # normalize channels independently, frames independently
            #         _axis = tuple(i for i in range(x.ndim) if i not in (ax['T'],ax['C']))
            # else:
            #     if 'C' not in axes or image.rgb == True:
            #          # normalize channels jointly
            #         _axis = None
            #     else:
            #         # normalize channels independently
            #         _axis = tuple(i for i in range(x.ndim) if i not in (ax['C'],))
            x = normalize(x, perc_low, perc_high, axis=_axis)

        # TODO: progress bar (labels) often don't show up. events not processed?
        if "T" in axes:
            app = use_app()
            t = axes_dict(axes)["T"]
            n_frames = x.shape[t]
            if n_tiles is not None:
                # remove tiling value for time axis
                n_tiles = tuple(v for i, v in enumerate(n_tiles) if i != t)

            def progress(it, **kwargs):
                progress_bar.label = "StarDist Prediction (frames)"
                progress_bar.range = (0, n_frames)
                progress_bar.value = 0
                progress_bar.show()
                app.process_events()
                for item in it:
                    yield item
                    progress_bar.increment()
                    app.process_events()
                app.process_events()

        elif n_tiles is not None and np.prod(n_tiles) > 1:
            n_tiles = tuple(n_tiles)
            app = use_app()

            def progress(it, **kwargs):
                progress_bar.label = "CNN Prediction (tiles)"
                progress_bar.range = (0, kwargs.get("total", 0))
                progress_bar.value = 0
                progress_bar.show()
                app.process_events()
                for item in it:
                    yield item
                    progress_bar.increment()
                    app.process_events()
                #
                progress_bar.label = "NMS Postprocessing"
                progress_bar.range = (0, 0)
                app.process_events()

        else:
            progress = False
            progress_bar.label = "StarDist Prediction"
            progress_bar.range = (0, 0)
            progress_bar.show()
            use_app().process_events()

        # semantic output axes of predictions
        assert model._axes_out[-1] == "C"
        axes_out = list(model._axes_out[:-1])

        if "T" in axes:
            x_reorder = np.moveaxis(x, t, 0)
            axes_reorder = axes.replace("T", "")
            axes_out.insert(t, "T")
            res = tuple(
                zip(
                    *tuple(
                        model.predict_instances(
                            _x,
                            axes=axes_reorder,
                            prob_thresh=prob_thresh,
                            nms_thresh=nms_thresh,
                            n_tiles=n_tiles,
                            scale=input_scale,
                            sparse=(not cnn_output),
                            return_predict=cnn_output,
                        )
                        for _x in progress(x_reorder)
                    )
                )
            )

            if cnn_output:
                labels, polys = tuple(zip(*res[0]))
                cnn_output = tuple(np.stack(c, t) for c in tuple(zip(*res[1])))
            else:
                labels, polys = res

            labels = np.asarray(labels)

            if len(polys) > 1:
                if timelapse_opts == TimelapseLabels.Match.value:
                    # match labels in consecutive frames (-> simple IoU tracking)
                    labels = group_matching_labels(labels)
                elif timelapse_opts == TimelapseLabels.Unique.value:
                    # make label ids unique (shift by offset)
                    offsets = np.cumsum([len(p["points"]) for p in polys])
                    for y, off in zip(labels[1:], offsets):
                        y[y > 0] += off
                elif timelapse_opts == TimelapseLabels.Separate.value:
                    # each frame processed separately (nothing to do)
                    pass
                else:
                    raise NotImplementedError(
                        f"unknown option '{timelapse_opts}' for time-lapse labels"
                    )

            labels = np.moveaxis(labels, 0, t)

            if isinstance(model, StarDist3D):
                # TODO poly output support for 3D timelapse
                polys = None
            else:
                polys = dict(
                    coord=np.concatenate(
                        tuple(
                            np.insert(p["coord"], t, _t, axis=-2)
                            for _t, p in enumerate(polys)
                        ),
                        axis=0,
                    ),
                    points=np.concatenate(
                        tuple(
                            np.insert(p["points"], t, _t, axis=-1)
                            for _t, p in enumerate(polys)
                        ),
                        axis=0,
                    ),
                )

            if cnn_output:
                pred = (labels, polys), cnn_output
            else:
                pred = labels, polys

        else:
            # TODO: possible to run this in a way that it can be canceled?
            pred = model.predict_instances(
                x,
                axes=axes,
                prob_thresh=prob_thresh,
                nms_thresh=nms_thresh,
                n_tiles=n_tiles,
                show_tile_progress=progress,
                scale=input_scale,
                sparse=(not cnn_output),
                return_predict=cnn_output,
            )
        progress_bar.hide()

        # determine scale for output axes
        scale_in_dict = dict(zip(axes, image.scale))
        scale_out = [scale_in_dict.get(a, 1.0) for a in axes_out]

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
                0.5 * (grid_dict.get(a, 1) / input_scale_dict.get(a, 1) - s)
                for a, s in zip(axes_out, scale_out)
            ]

            layers.append(
                (
                    dist,
                    dict(
                        name="StarDist distances",
                        scale=[1] + _scale,
                        translate=[0] + _translate,
                        **lkwargs,
                    ),
                    "image",
                )
            )
            layers.append(
                (
                    prob,
                    dict(
                        name="StarDist probability",
                        scale=_scale,
                        translate=_translate,
                        **lkwargs,
                    ),
                    "image",
                )
            )
        else:
            labels, polys = pred

        if output_type in (Output.Labels.value, Output.Both.value):
            layers.append(
                (
                    labels,
                    dict(name="StarDist labels", scale=scale_out, opacity=0.5, **lkwargs),
                    "labels",
                )
            )
        if output_type in (Output.Polys.value, Output.Both.value):
            n_objects = len(polys["points"])
            if isinstance(model, StarDist3D):
                surface = surface_from_polys(polys)
                layers.append(
                    (
                        surface,
                        dict(
                            name="StarDist polyhedra",
                            contrast_limits=(0, surface[-1].max()),
                            scale=scale_out,
                            colormap=label_colormap(n_objects),
                            **lkwargs,
                        ),
                        "surface",
                    )
                )
            else:
                # TODO: sometimes hangs for long time (indefinitely?) when returning many polygons (?)
                #       seems to be a known issue: https://github.com/napari/napari/issues/2015
                # TODO: coordinates correct or need offset (0.5 or so)?
                shapes = np.moveaxis(polys["coord"], -1, -2)
                layers.append(
                    (
                        shapes,
                        dict(
                            name="StarDist polygons",
                            shape_type="polygon",
                            scale=scale_out,
                            edge_width=0.75,
                            edge_color="yellow",
                            face_color=[0, 0, 0, 0],
                            **lkwargs,
                        ),
                        "shapes",
                    )
                )
        return layers

    # -------------------------------------------------------------------------

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

            # try to get a hold of the viewer (can be None when plugin starts)
            if self.viewer is None:
                # TODO: when is this not safe to do and will hang forever?
                # while plugin.viewer.value is None:
                #     time.sleep(0.01)
                if plugin.viewer.value is not None:
                    self.viewer = plugin.viewer.value
                    if DEBUG:
                        print("GOT viewer")

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
                    valid
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
                    plugin.axes.tooltip = "\n".join(
                        [f"{a} = {s}" for a, s in zip(axes, get_data(image).shape)]
                    )
                    return axes, image
                else:
                    if err is not None:
                        err = str(err)
                        err = err[:-1] if err.endswith(".") else err
                        plugin.axes.tooltip = err
                        # warn(err) # alternative to tooltip (gui doesn't show up in ipython)
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
                        # warn(err) # alternative to tooltip (gui doesn't show up in ipython)
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
                    f"valid ({all_valid}):",
                    ", ".join([f"{k}={v}" for k, v in vars(self.valid).items()]),
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
    def _perc_low_change():
        plugin.perc_high.value = max(plugin.perc_low.value + 0.01, plugin.perc_high.value)

    @change_handler(plugin.perc_high)
    def _perc_high_change():
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

            @thread_worker
            def _get_model_folder():
                return get_model_folder(*key)

            def _process_model_folder(path):
                try:
                    model_configs[key] = load_json(str(path / "config.json"))
                    try:
                        # not all models have associated thresholds
                        model_threshs[key] = load_json(str(path / "thresholds.json"))
                    except FileNotFoundError:
                        pass
                finally:
                    select_model(key)
                    plugin.progress_bar.hide()

            worker = _get_model_folder()
            worker.returned.connect(_process_model_folder)
            worker.start()

            # delay showing progress bar -> won't show up if model already downloaded
            # TODO: hacky -> better way to do this?
            time.sleep(0.1)
            plugin.call_button.enabled = False
            plugin.progress_bar.label = "Downloading model"
            plugin.progress_bar.show()

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
    @change_handler(plugin.image, init=False)
    def _image_change(image: napari.layers.Image):
        ndim = get_data(image).ndim
        plugin.image.tooltip = f"Shape: {get_data(image).shape}"

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
    def _n_tiles_change():
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
    def _input_scale_change():
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
    def _output_type_change():
        update._update()

    # restore defaults
    @change_handler(plugin.defaults_button, init=False)
    def restore_defaults():
        for k, v in DEFAULTS.items():
            getattr(plugin, k).value = v

    # -------------------------------------------------------------------------

    # allow some widgets to shrink because their size depends on user input
    plugin.image.native.setMinimumWidth(240)
    plugin.model2d.native.setMinimumWidth(240)
    plugin.model3d.native.setMinimumWidth(240)
    plugin.timelapse_opts.native.setMinimumWidth(240)

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

    # push 'call_button' and 'progress_bar' to bottom
    layout.insertStretch(layout.count() - 2)

    # for i in range(layout.count()):
    #     print(i, layout.itemAt(i).widget())

    return plugin
