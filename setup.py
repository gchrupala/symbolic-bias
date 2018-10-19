# encoding: utf-8
from setuptools import setup

setup(name='symbolic_bias',
      version='0.3',
      description='Visually grounded speech models in Pytorch',
      url='https://github.com/gchrupala/vgs',
      author='Grzegorz Chrupała',
      author_email='g.chrupala@uvt.nl',
      license='MIT',
      packages=['onion','vg', 'vg.defn'],
      zip_safe=False,
      install_requires=[
          'torch==0.3.1',
          'torchvision',
          'sklearn',
          'scipy'
      ])
