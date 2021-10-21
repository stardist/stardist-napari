# StarDist Napari Plugin

[![PyPI version](https://img.shields.io/pypi/v/stardist-napari.svg)](https://pypi.org/project/stardist-napari)
[![Image.sc forum](https://img.shields.io/badge/dynamic/json.svg?label=forum&url=https%3A%2F%2Fforum.image.sc%2Ftags%2Fstardist.json&query=%24.topic_list.tags.0.topic_count&colorB=brightgreen&suffix=%20topics&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA4AAAAOCAYAAAAfSC3RAAABPklEQVR42m3SyyqFURTA8Y2BER0TDyExZ+aSPIKUlPIITFzKeQWXwhBlQrmFgUzMMFLKZeguBu5y+//17dP3nc5vuPdee6299gohUYYaDGOyyACq4JmQVoFujOMR77hNfOAGM+hBOQqB9TjHD36xhAa04RCuuXeKOvwHVWIKL9jCK2bRiV284QgL8MwEjAneeo9VNOEaBhzALGtoRy02cIcWhE34jj5YxgW+E5Z4iTPkMYpPLCNY3hdOYEfNbKYdmNngZ1jyEzw7h7AIb3fRTQ95OAZ6yQpGYHMMtOTgouktYwxuXsHgWLLl+4x++Kx1FJrjLTagA77bTPvYgw1rRqY56e+w7GNYsqX6JfPwi7aR+Y5SA+BXtKIRfkfJAYgj14tpOF6+I46c4/cAM3UhM3JxyKsxiOIhH0IO6SH/A1Kb1WBeUjbkAAAAAElFTkSuQmCC)](https://forum.image.sc/tags/stardist)

This project provides the [napari](https://napari.org/) plugin for [StarDist](https://github.com/stardist/stardist), a deep learning based 2D and 3D object detection method with star-convex shapes. StarDist has originally been developed (see [papers](https://github.com/stardist/stardist#stardist---object-detection-with-star-convex-shapes)) for the segmentation of densely packed cell nuclei in challenging images with low signal-to-noise ratios. The plugin allows to apply pretrained and custom trained models from within napari.

![Screenshot](https://github.com/stardist/stardist-napari/raw/main/images/stardist_napari_screenshot_small.png)

## Installation & Usage

Install the plugin with `pip install stardist-napari` or from within napari via `Plugins > Install/Uninstall Package(s)…`. If you want GPU-accelerated prediction, please read the more detailed [installation instructions](https://github.com/stardist/stardist#installation) for StarDist.

You can activate the plugin in napari via `Plugins > StarDist: StarDist`. Example images for testing are provided via `File > Open Sample > StarDist`.

For a more detailed demonstration of the plugin, please [watch this short video](https://www.youtube.com/watch?v=Km1_TnUQ4FM&list=PLilvrWT8aLuZCaOkjucLjvDu2YRtCS-JT&index=5).

There's no dedicated documentation yet, but the most important parameters are identical to those of our [StarDist ImageJ/Fiji plugin](https://github.com/stardist/stardist-imagej), which are documented [here](https://imagej.net/plugins/stardist#usage).

If you use this plugin for your research, please [cite us](https://github.com/stardist/stardist#how-to-cite).

## Troubleshooting & Support

- The [image.sc forum](https://forum.image.sc/tag/stardist) is the best place to start getting help and support. Make sure to use the tag `stardist`, since we are monitoring all questions with this tag.
- For general questions about StarDist, it's worth taking a look at the [frequently asked questions (FAQ)]( https://stardist.net/docs/faq.html).
- If you have technical questions or found a bug, feel free to [open an issue](https://github.com/stardist/stardist-napari/issues).
