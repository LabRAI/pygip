# PyHazards Documentation

## Build

cd docs
sphinx-build -b html source build/html
cp -r build/html/* .

## clean

sphinx-build -M clean source build

## test

open docs/build/html/index.html

