import io
import os

from pathlib import Path

from setuptools import setup,find_packages

#metadata

NAME='prediction_model'
DESCRIPTION='student dropout'
REQUIRES_PYTHON ='>=3.8.0'

pwd=os.path.abspath(os.path.dirname(__file__))

#encoding='UTF-16LE'
def list_reqs(fname='requirements.txt'):
    with io.open(os.path.join(pwd,fname),encoding='UTF-8') as f:
        return f.read().splitlines()

try:
    with io.open(os.path.join(pwd,'README.md')) as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    long_description=DESCRIPTION

ROOT_DIR=Path(__file__).resolve().parent
PACKAGE_DIR =ROOT_DIR/NAME

about={}

with open(PACKAGE_DIR/'VERSION') as f:
    _version =f.read().strip()
    about['__version__']=_version

setup(name=NAME,
      version=about['__version__'],
      description=DESCRIPTION,
      long_description='',
      python_requires=REQUIRES_PYTHON,
      packages=find_packages(exclude=('tests',)),
      package_data={'prediction_model':['VERSION']},
      install_requires=list_reqs(),
      include_package_data=True

      )