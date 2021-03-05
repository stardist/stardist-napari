from setuptools import setup, find_packages
from os import path


_dir = path.dirname(__file__)

with open(path.join(_dir,'stardist_napari','_version.py'), encoding="utf-8") as f:
    exec(f.read())

with open(path.join(_dir,'README.md'), encoding="utf-8") as f:
    long_description = f.read()


setup(
    name='stardist-napari',
    version=__version__,
    description='StarDist for Napari',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/stardist/stardist-napari',
    author='Uwe Schmidt, Martin Weigert',
    author_email='research@uweschmidt.org, martin.weigert@epfl.ch',
    license='BSD 3-Clause License',
    packages=find_packages(),
    python_requires='>=3.6',

    entry_points={'napari.plugin': 'StarDist = stardist_napari'},

    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: BSD License',

        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',

        'Framework :: napari',
    ],

    install_requires=[
        'stardist>=0.7.0',
        'tensorflow',
        'napari>=0.4.8',
        # 'magicgui>0.2.9'
    ],
)