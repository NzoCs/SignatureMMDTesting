from setuptools import setup, Extension

module = Extension ('cython_backend', sources=['cython_backend.pyx'])

setup(
    name='cython_backend',
    version='1.0',
    author='AndrewAlden',
    ext_modules=[module]
)