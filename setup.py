try:
	from setuptools import setup
except ImportError:
	from distutils.core import setup

try:
	from distutils.command.build_py import build_py_2to3 as build_py
except ImportError:
	from distutils.command.build_py import build_py

package = 'pyAudioAnalysis'
version = '1.0'

setup(name=package,
	version=version,
	packages = ['pyaudio_analysis','data'],
        scripts = ['pyaudio_analysis/audioAnalysis.py'],
	cmdclass={'build_py': build_py},
	author='Theodoros Giannakopoulos',
	author_email='tyiannak@gmail.com',
	long_description=open('README.md').read(),
	url='https://github.com/tyiannak/pyAudioAnalysis',
	install_requires=[
		"hmmlearn",
		"simplejson",
		"eyed3"
	]
)
