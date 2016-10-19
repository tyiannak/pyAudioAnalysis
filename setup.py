try:
	from setuptools import setup
except ImportError:
	from distutils.core import setup

package = 'pyAudioAnalysis'
version = '1.0'

setup(name=package,
	version=version,
	packages = ["pyaudio_analysis"],
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
