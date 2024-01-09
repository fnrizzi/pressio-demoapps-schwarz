
# Overview

This repository provides an interface for applying domain-decomposed solutions of fluid flow ODEs via the Schwarz alternating method through the [pressio-demoapps](https://github.com/Pressio/pressio-demoapps) solver and sample problem suite. This serves as a launching point for exploring Schwarz coupling for advection-dominated systems, as well as coupling full-order ("high-fidelity," FOM) solvers to data-driven projection-based reduced-order models (PROMs) via [Pressio](https://github.com/Pressio/pressio). The framework exemplified in the test cases (in ```tests_cpp/```) should be easily extensible to any sample case provided by **pressio-demoapps**.

# Building and Running Tests

Executing the test cases requires a copy of the **pressio-demoapps** source (which has bundled the **Eigen** library)
and the **pressio**source. Building and executing the test cases can be performed as

```
git clone git@github.com:cwentland0/pressio-demoapps-schwarz.git
export CXX=<path-to-your-CXX-compiler>
export PDA_ROOT=<path-to-pressio-demoapps-root>
export PRESSIO_ROOT=<path-to-pressio>
cd pressio-demoapps-schwarz && mkdir build && cd build
cmake -DPDA_SOURCE=${PDA_ROOT} -DPRESSIO_SOURCE=${PRESSIO_ROOT} ..
make -j4
ctest -j4
```

# Postprocessing

Python utilities for data extraction, visualization, PROM preparation, and error measurement can be found in the ```python/``` directory. Refer to the README there for instructions on installing and using the associated local package.