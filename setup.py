# encoding: utf-8
from setuptools import setup

setup(name='symbolic_bias',
      version='0.4',
      description='Visually grounded speech models in Pytorch',
      url='https://github.com/gchrupala/vgs',
      author='Grzegorz Chrupała',
      author_email='g.chrupala@uvt.nl',
      license='MIT',
      packages=['onion','vg', 'vg.defn'],
      zip_safe=False,
      install_requires=[
          'torch == 1.0.0',
          'torchvision == 0.2.1',
          'scikit-learn == 0.20.1',
          'scipy == 1.10.0',
          'python-Levenshtein == 0.12.0'
          
      ])
