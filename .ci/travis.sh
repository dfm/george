#!/bin/bash -x

# If building the paper, do that here
if [[ $TEST_LANG == paper ]]
then
  if git diff --name-only $TRAVIS_COMMIT_RANGE | grep 'paper/'
  then
    echo "Building the paper..."
    export GEORGE_BUILDING_PAPER=true
    mkdir -p ~/.config/Tectonic
    cp .ci/tectonic.config.toml ~/.config/Tectonic/config.toml
    source "$( dirname "${BASH_SOURCE[0]}" )"/setup-tectonic.sh
    return
  fi
  export GEORGE_BUILDING_PAPER=false
  return
fi

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
python setup.py develop
