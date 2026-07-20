# Rotating BEMTool Fork

[![Tests](https://github.com/0xBachmann/BemTool/actions/workflows/tests.yaml/badge.svg?branch=master)](https://github.com/0xBachmann/BemTool/actions/workflows/tests.yaml?query=branch%3Amaster)

This repository is the software companion to the MSc thesis
**Boundary Element Method for Electromagnetic Scattering from Rotating Bodies**
by Jonas Bachmann, ETH Zürich, April 2026.

The repository contains a research fork of
[BEMTool](https://github.com/xclaeys/BemTool). It keeps the original BEMTool
infrastructure for meshes, finite-element spaces, quadrature, operator
assembly, and potential evaluation, and adds rotation-aware Helmholtz kernels
and cylinder-scattering examples used in the thesis.

The mathematical derivations, notation, modelling assumptions, and discussion
of the numerical results are given in the thesis. This repository is intended
to make the implementation and numerical experiments easier to inspect,
build, test, and reproduce.

## Citation

Please cite the thesis if you use this fork or the rotating-kernel
implementation:

> Jonas Bachmann. *Boundary Element Method for Electromagnetic Scattering from
> Rotating Bodies*. MSc thesis, ETH Zürich, Seminar for Applied Mathematics,
> April 2026.

```bibtex
@mastersthesis{bachmann2026rotatingbem,
  author = {Bachmann, Jonas},
  title  = {Boundary Element Method for Electromagnetic Scattering from Rotating Bodies},
  school = {ETH Zurich},
  year   = {2026},
  month  = apr,
  note   = {MSc thesis, Seminar for Applied Mathematics}
}
```

## Scope

The implementation focuses on two-dimensional scalar Helmholtz models obtained
from the reduction of electromagnetic scattering by an infinitely long cylinder
aligned with the z-axis. The main model problems are:

- scattering by a perfectly electrically conducting cylinder,
- transmission through a homogeneous dielectric cylinder,
- comparison of stationary and rotating dielectric-cylinder cases.

Rotation is introduced at the kernel level. In the rotating interior problem,
the stationary free-space Helmholtz Green's function is replaced by a
first-order rotation-aware Green's function. The surrounding BEMTool assembly
machinery is reused as much as possible.

The present implementation is a proof-of-concept for the scalar reduced
problem. The full horizontal 1-form equation, non-circular rotating geometries,
higher-order rotational corrections, and fast BEM acceleration are discussed as
future extensions in Chapter 5 of the thesis.

## Relation to the thesis

The table below gives the main correspondence between code concepts and thesis
sections.

| Repository concept | Main code location | Thesis reference |
| --- | --- | --- |
| Reduced scalar rotating model | `bemtool/potential/rotating_helmholtz_fast_pot.hpp`, `bemtool/operator/rotating_helmholtz_fast_op.hpp` | Chapter 2, especially Sections 2.6.2, 2.7, 2.8, and 2.9.1 |
| Rotating Green's function | `bemtool/potential/rotating_helmholtz_fast_pot.hpp` | Sections 2.9 and 2.9.1 |
| First-order phase-factor approximation | `rotating_helmholtz_fast_*` files | Sections 2.6.2 and 2.9.1, with supporting estimates in Appendix A.6 |
| Layer potentials and boundary integral operators | `bemtool/potential/`, `bemtool/operator/` | Sections 3.2--3.5 and Appendix A.9 |
| Weak treatment of the hypersingular operator | weak hypersingular operator code | Appendix A.10 |
| PEC cylinder formulation | `rotation/pec_cylinder.cpp` | Sections 3.6 and 4.2 |
| Dielectric transmission formulation | `rotation/dielectric_cylinder.cpp` | Sections 3.7 and 4.3 |
| Galerkin discretization and BEMTool conventions | `rotation/helpers.hpp`, BEMTool assembly code | Sections 3.8 and 3.9 |
| Numerical cylinder experiments | `rotation/`, `rotation/plotting/` | Sections 4.1--4.3 |

## Main additions

The rotation-specific code is concentrated in two header files:

```text
bemtool/potential/rotating_helmholtz_fast_pot.hpp
bemtool/operator/rotating_helmholtz_fast_op.hpp
```

These files define the rotating Helmholtz kernel tag `RH` and the associated
layer potentials and boundary integral operators.

The suffix `fast` refers to the Steinberg-type first-order phase-factor
approximation of the rotating Green's function. It does not mean that the
dense BEM matrices are accelerated by a fast multipole method, hierarchical
matrices, or another fast-summation technique.

Convenience aliases are provided so that the rotating kernel can be used in
the same style as the built-in BEMTool Helmholtz kernels. Examples include:

- `RH_SL_2D_P0`, `RH_SL_2D_P1` for rotating single-layer potentials,
- `RH_DL_2D_P0`, `RH_DL_2D_P1` for rotating double-layer potentials,
- `RH_SL_2D_P1xP1` for rotating single-layer boundary operators,
- `RH_DL_2D_P1xP1` for rotating double-layer boundary operators,
- `RH_TDL_2D_P1xP1` for rotating adjoint double-layer boundary operators,
- `RH_HS_2D_P1xP1` for rotating hypersingular boundary operators.

This makes it possible to switch between stationary and rotating Helmholtz
formulations with minimal changes at the assembly level.

## Repository layout

```text
├── bemtool/                  # Core BEMTool headers and rotating-kernel extensions
│   ├── calculus/             # Small vectors, matrices, and expression utilities
│   ├── fem/                  # Degrees of freedom, shape functions, interpolation
│   ├── mesh/                 # Mesh data structures, elements, normals, adjacency
│   ├── miscellaneous/        # Output, wrappers, reference data, helper utilities
│   ├── operator/             # Boundary integral operators
│   ├── potential/            # Layer potentials
│   └── quadrature/           # BEM and potential quadrature rules
├── mesh/                     # Example Gmsh geometries and meshes
├── rotation/                 # Rotating-cylinder experiments and post-processing
│   ├── dielectric_cylinder.cpp
│   ├── pec_cylinder.cpp
│   ├── helpers.hpp
│   ├── plotting/             # Python plotting and post-processing scripts
│   └── scripts/              # Cluster / sbatch helper scripts
├── tests/                    # Small consistency and regression tests
├── CMakeLists.txt
├── build_and_test.sh         # Configure, build, and run the tests
└── README.md
```

## Building and testing

The simplest way to configure, build, and test the project is to run:

```bash
./build_and_test.sh
```

This creates a `build` directory, builds the available targets, and runs the
automated tests. Depending on the machine and enabled targets, the full build
can take several minutes.

A manual build is also possible:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

The main dependencies are those required by BEMTool and the added examples:
a C++17-capable compiler, CMake, Eigen, and Boost.

## Automated tests

The `tests/` directory contains small consistency checks for the rotating
implementation. They are intended to be run after installation and after code
changes.

The current tests check:

- the non-rotating limit of the rotating Helmholtz implementation,
- consistency of Green-function derivatives with finite differences.

Run the tests with:

```bash
./build_and_test.sh
```

or, after a manual build, with:

```bash
ctest --test-dir build --output-on-failure
```

The larger numerical examples in `rotation/` are thesis experiments and should
not be treated as a replacement for the automated test suite.

## Numerical examples

The main example drivers are located in `rotation/`. They reproduce the
proof-of-concept experiments from Chapter 4 of the thesis. The table below is
organized by repository file or directory: each item is listed once, together
with the thesis material it is connected to.

| File or directory | Related thesis material | Role in the repository |
| --- | --- | --- |
| `rotation/pec_cylinder.cpp` | PEC boundary integral formulation in Section 3.6 and stationary PEC cylinder experiments in Section 4.2, including Figure 4.1 and Table 4.1. | Driver for the PEC cylinder test case. It compares TM-like Dirichlet and TE-like Neumann branches, first- and second-kind formulations, and condition-number diagnostics. |
| `rotation/dielectric_cylinder.cpp` | Dielectric transmission formulation in Section 3.7 and dielectric cylinder experiments in Section 4.3. This includes the stationary comparisons in Figures 4.2--4.3 and the rotating-cylinder experiments in Figures 4.4--4.7. | Driver for the homogeneous dielectric cylinder test case. It is used for stationary transmission tests, rotating-field comparisons, isocontour studies, and the even--odd scaling experiment with respect to the interior angular velocity. |
| `rotation/helpers.hpp` | Galerkin discretization and BEMTool implementation conventions in Sections 3.8--3.9. | Shared helper routines for setting up spaces, assembling systems, evaluating fields, and writing output used by the cylinder examples. |
| `rotation/plotting/` | Post-processing of the numerical experiments in Chapter 4, especially the field plots, isocontours, and scaling comparisons in Section 4.3. | Python scripts for turning generated field data into the plots discussed in the thesis. |
| `rotation/scripts/` | Reproducibility support for the numerical experiments in Chapter 4. | Shell and batch scripts documenting the command-line arguments and run configurations used for the thesis experiments. |

After building, run the example executables from the build directory. The
scripts in `rotation/scripts/` document the arguments used for the thesis
runs. Generated field data can be post-processed with the Python scripts in
`rotation/plotting/`.

## License

This repository is based on BEMTool and keeps the original GPL license terms.
See `LICENSE` for details.