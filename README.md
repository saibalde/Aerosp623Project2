# Aerosp 623 Project 2

Finite volume solver for 2D Euler flow

## Requirements

1.  To compile the code, a compiler supporting C++11 standard is required.

2.  You need to use the [CMake](https://cmake.org/) build system to build this
    code.
    
3.  The code also depends on the [Armadillo](http://arma.sourceforge.net/)
    library. It needs to be installed and searchable through `find_package`
    command from CMake.

## Compilation

To compile the code, create a new directory inside the main code base file:
```sh
cd /project/root/directory
mkdir build
cd build
cmake ..
make
```

## Usage

The compilation process builds five executables inside the `build` directory:

-   `app/RoeFlux` to run the Roe flux tests
-   `app/FirstOrderPreserve` to run the free stream preservation test
-   `app/FirstOrderSolver` to run the solver with full boundary condition
-   `app/SecondOrderPreserve` to run the free stream preservation test
-   `app/SecondOrderSolver` to run the solver with full boundary condition

Each of these programs requires an integer between 0 and 4 (both inclusive)
to indicate which mesh should be used (e.g. 0 for `bump0.gri` etc.).

Note that, before running `SecondOrderSolver`, the `FirstOrderSolver` should
be run to generate the initial state.
