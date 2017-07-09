#!/bin/bash -x

# http://conda.pydata.org/docs/travis.html#the-travis-yml-file
if [[ "$TRAVIS_OS_NAME" == "osx" ]]; then
  wget https://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-x86_64.sh -O miniconda.sh;
else
  wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh;
fi
bash miniconda.sh -b -p $HOME/miniconda
export PATH="$HOME/miniconda/bin:$PATH"

# Conda Python
hash -r
conda config --set always_yes yes --set changeps1 no
conda update -q conda
conda info -a
conda create --yes -n test python=$PYTHON_VERSION
source activate test
conda install numpy=$NUMPY_VERSION scipy setuptools pytest pytest-cov jinja2 pyyaml pip
conda install -c conda-forge pybind11
pip install coveralls

# Build the extension
if [[ "$TRAVIS_OS_NAME" == "osx" ]]; then
  python setup.py install
else
  CXX=g++-4.9 CC=gcc-4.9 python setup.py build_ext $BUILD_ARGS install
fi
