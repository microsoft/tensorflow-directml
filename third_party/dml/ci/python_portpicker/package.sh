#!/bin/sh -ex

unset PYTHONPATH
python3 -m venv build/venv
. build/venv/bin/activate

pip install --upgrade build twine
python -m build
twine check dist/*

echo 'When ready, upload to PyPI using: build/venv/bin/twine upload dist/*'
