#!/usr/bin/env python
from setuptools import setup

setup(
    name="pyAudioAnalysis",
    version="1.0",
    description="",
    author="",
    author_email="",
    url="",
    include_package_data=True,
    zip_safe=True,
    install_requires=[
        'scipy',
        'numpy',
        'matplotlib',
        'sklearn',
        'hmmlearn',
        'simplejson',
        'eyed3',
        'pydub',
    ]
)