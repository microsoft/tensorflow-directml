#!/bin/sh -ex

unset PYTHONPATH
python3 -m venv build/venv
. build/venv/bin/activate

pip install --upgrade pip
pip install tox
tox -e "py3$(python -c 'import sys; print(sys.version_info.minor)')"
