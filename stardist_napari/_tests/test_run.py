import napari
import numpy as np
import pytest
from scipy.ndimage import rotate
from stardist.models import StarDist2D, StarDist3D

from stardist_napari._dock_widget import Output, TimelapseLabels


def test_fluo_2d(call_plugin, nuclei_2d):
    kwargs = dict(
        image=nuclei_2d,
        axes="YX",
        model=StarDist2D.from_pretrained("2D_versatile_fluo"),
    )

    for output_type, num_out in (
        (Output.Labels, 1),
        (Output.Polys, 1),
        (Output.Both, 2),
    ):
        out = call_plugin(
            **kwargs,
            output_type=output_type,
        )
        assert len(out) == num_out

    out = call_plugin(
        **kwargs,
        input_scale=0.75,
        output_type=Output.Both,
        cnn_output=True,
        n_tiles=(3, 2),
    )

    assert len(out) == 4

    return out, kwargs


def test_fluo_3d(call_plugin, nuclei_3d):
    kwargs = dict(
        image=nuclei_3d,
        axes="ZYX",
        model=StarDist3D.from_pretrained("3D_demo"),
    )

    for output_type, num_out in (
        (Output.Labels, 1),
        (Output.Polys, 1),
        (Output.Both, 2),
    ):
        out = call_plugin(
            **kwargs,
            output_type=output_type,
        )
        assert len(out) == num_out

    out = call_plugin(
        **kwargs,
        input_scale=0.75,
        cnn_output=True,
        output_type=Output.Both,
        n_tiles=(2, 1, 2),
    )
    assert len(out) == 4

    return out, kwargs


# def test_custom_model_2d(call_plugin, nuclei_2d):
#     from csbdeep.models.pretrained import get_model_folder

#     model_type = StarDist2D
#     model_name = "2D_versatile_fluo"
#     model_path = get_model_folder(model_type, model_name)

#     kwargs = dict(viewer=None, image=nuclei_2d, axes="YX")
#     labels1, polys1 = call_plugin(**kwargs, model_type=model_type, model2d=model_name)
#     labels2, polys2 = call_plugin(
#         **kwargs, model_type=CUSTOM_MODEL, model_folder=model_path
#     )

#     assert np.allclose(labels1[0], labels2[0]) and labels1[1] == labels2[1]
#     assert np.allclose(polys1[0], polys2[0]) and polys1[1] == polys2[1]

#     return kwargs


@pytest.mark.parametrize("timelapse", (False, True))
def test_multiclass_2d(call_plugin, he_2d, timelapse):

    if timelapse:
        image = np.stack([he_2d.data] * 2, axis=0)
        image = napari.layers.Image(image, name="timelapse")
    else:
        image = he_2d

    kwargs = dict(
        image=image,
        axes="TYXC" if timelapse else "YXC",
        model=StarDist2D.from_pretrained("2D_conic_he"),
    )

    # TODO: remove this line once there is a registered multiclass
    if kwargs["model"] is None:
        return

    # for output_type, num_out in (
    #     (Output.Labels.value, 2),
    #     (Output.Polys.value, 1),
    #     (Output.Both.value, 3),
    # ):
    #     out = plugin(
    #         **kwargs,
    #         output_type=output_type,
    #     )
    #     assert len(out) == num_out

    out = call_plugin(
        **kwargs,
        output_type=Output.Both,
        cnn_output=True,
        n_tiles=(1, 3, 2, 1) if timelapse else (3, 2, 1),
    )

    assert len(out) == 6

    return out, kwargs


def test_timelapse_2d(call_plugin, nuclei_2d):
    timelapse = np.stack(
        [
            rotate(nuclei_2d.data, deg, reshape=False, mode="reflect")
            for deg in np.linspace(0, 50, 3)
        ],
        axis=0,
    )
    timelapse = napari.layers.Image(timelapse, name="timelapse")
    kwargs = dict(
        image=timelapse,
        axes="TYX",
        model=StarDist2D.from_pretrained("2D_versatile_fluo"),
    )

    for t in TimelapseLabels:
        call_plugin(**kwargs, timelapse_opts=t)

    out = call_plugin(**kwargs, cnn_output=True, output_type=Output.Both)
    assert len(out) == 4

    out = call_plugin(**kwargs, n_tiles=(1, 2, 3), output_type=Output.Both)
    assert len(out) == 2

    return out, kwargs


def test_timelapse_3d(call_plugin, nuclei_3d):
    timelapse = np.stack([np.roll(nuclei_3d.data, n) for n in (0, 10)], axis=0)
    timelapse = napari.layers.Image(timelapse, name="timelapse")
    kwargs = dict(
        image=timelapse,
        axes="TZYX",
        model=StarDist3D.from_pretrained("3D_demo"),
    )

    with pytest.raises(NotImplementedError):
        call_plugin(**kwargs, output_type=Output.Polys)

    kwargs["output_type"] = Output.Labels

    for t in TimelapseLabels:
        call_plugin(**kwargs, timelapse_opts=t)

    out = call_plugin(**kwargs, cnn_output=True)
    assert len(out) == 3

    out = call_plugin(**kwargs, n_tiles=(1, 1, 2, 3))
    assert len(out) == 1

    return out, kwargs


def test_he_2d(call_plugin, he_2d):
    kwargs = dict(
        image=he_2d,
        axes="YXC",
        model=StarDist2D.from_pretrained("2D_versatile_he"),
    )

    out = call_plugin(**kwargs, output_type=Output.Both)
    assert len(out) == 2

    return out, kwargs


if __name__ == "__main__":
    from stardist import data

    from stardist_napari import make_dock_widget_function

    plugin, nuclei_2d = make_dock_widget_function(), napari.layers.Image(
        data.test_image_nuclei_2d()
    )
    out, kwargs = test_fluo_2d(plugin, nuclei_2d)
    # plugin, img = make_dock_widget_function(), napari.layers.Image(data.test_image_he_2d())
    # out, kwargs = test_multiclass_2d(plugin, img)

    # plugin, img = make_dock_widget_function(), napari.layers.Image(
    #     data.test_image_he_2d()[:128, :128]
    # )
    # out, kwargs = test_multiclass_2d(plugin, img, True)

    # plugin, img = make_dock_widget_function(), napari.layers.Image(data.test_image_nuclei_2d())
    # out, kwargs = test_timelapse_2d(plugin, img)
    # plugin, img = make_dock_widget_function(), napari.layers.Image(data.test_image_nuclei_3d())
    # out, kwargs = test_fluo_3d(plugin, img)

    # plugin, img = make_dock_widget_function(), napari.layers.Image(data.test_image_nuclei_3d())
    # out, kwargs = test_timelapse_3d(plugin, img)
