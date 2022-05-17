def _test_image_nuclei_2d():
    from stardist import data

    return [(data.test_image_nuclei_2d(), {"name": "nuclei_2d"})]


def _test_image_he_2d():
    from stardist import data

    return [(data.test_image_he_2d(), {"name": "he_2d"})]


def _test_image_nuclei_3d():
    from stardist import data

    return [(data.test_image_nuclei_3d(), {"name": "nuclei_3d"})]
