# Photon Generator

## Introduction

The library is used to generate a variety synthetic data representing the photons of Astronomical sources hitting a CCD detector 2D input array. Several types of source can be modelled:

- Flat field sources 
- 2D Gaussian source based on the corresponding PDF function
- 2D Moffat source based on the corresponding PDF function
- Diffraction-limited source
- Seeing limited source, where the speckle patterns are simulated from Komologorov phase screens

## License

The code is licensed under the Apache 2.0 license. See license file for details

## Installing

This requires python 3.6 +

```shell script
git clone https://github.com/fatmac78/photon_generator.git
cd photon_generator
pip install -r requirements.txt
```

## Usage

An example of the usage of the function in the library may be found in the included test cases.  The tests may be run as follow:

```shell script
python -m pytest
```

