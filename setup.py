from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

setup(
    name='pyAudioAnalysis',
    version='0.1.3',

    description='Python Audio Analysis Library: Feature Extraction, Classification, Segmentation and Applications',
    long_description='pyAudioAnalysis is a Python library covering a wide range of audio analysis tasks, including: feature extraction, classification, segmentation and visualization.',
    url='https://github.com/tyiannak/pyAudioAnalysis',
    author='Theodoros Giannakopoulos',
    author_email='tyiannak@gmail.com',
    license='Apache 2.0',

    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Topic :: Multimedia :: Sound/Audio :: Analysis',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7'
    ],

    keywords='audio analysis feature extraction classification segmentation visualization',

    packages=find_packages(exclude=['contrib', 'docs', 'tests']),

    install_requires=['numpy', 'matplotlib', 'scipy', 'sklearn', 'hmmlearn', 'simplejson', 'eyed3', 'pydub'],

    #package_data={
    #    'pyAudioAnalysis': ['data/*'],
    #},
)

