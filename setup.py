import os
from setuptools import setup

def read(fname):
    try:
        with open(os.path.join(os.path.dirname(__file__), fname)) as fh:
            return fh.read()
    except IOError:
        return ''

requirements = read('requirements.txt').splitlines()

setup(name='pyAudioAnalysis',
      version='0.3.0',
      description='Python Audio Analysis Library: Feature Extraction, Classification, Segmentation and Applications',
      url='https://github.com/tyiannak/pyAudioAnalysis',
      author='Theodoros Giannakopoulos',
      author_email='tyiannak@gmail.com',
      license='Apache License, Version 2.0',
      packages=['pyAudioAnalysis'],
      zip_safe=False,
      install_requires=requirements,
      )
