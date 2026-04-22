# Rotating BEMTool Fork

This repository is a fork of **BemTool** used for the numerical experiments in the thesis *Boundary Element Method for Electromagnetic Scattering from Rotating Bodies*. For the general design, usage, and core boundary-element infrastructure of BemTool, please refer to the original project and its documentation. In this fork, BemTool is used as the generic BEM backbone: mesh handling, basis functions, quadrature, operator assembly, and potential evaluation remain in the BemTool style, while the main extension is the introduction of rotation-aware kernels for the reduced scalar scattering model.

## Purpose of this fork

The goal of this fork is to extend the standard 2D Helmholtz BEM workflow by replacing the stationary free-space Green's function with a **rotation-aware Green's function** for scattering by uniformly rotating cylindrical media. This follows the thesis framework and the project description: the rotating problem is treated as a modification of the kernel, while the surrounding BEM machinery is kept as close as possible to standard BemTool.

In particular, this repository was used to produce the numerical results for:

- scattering by a **PEC cylinder**
- scattering by a **dielectric cylinder**
- comparison between the **stationary** and **rotating** cases

as discussed in the thesis chapters on the rotating cylinder test cases.

## Main additions

The main implementation relevant to the rotating-medium model is contained in the following files:

- `bemtool/potential/rotating_helmholtz_fast_pot.hpp`
- `bemtool/operator/rotating_helmholtz_fast_op.hpp`

These files implement the **Green's function of a rotating medium** and the associated boundary integral operators in the first-order rotating approximation used in the thesis. The implementation follows the Steinberg-type approximation in which the rotating Green kernel is written as a correction of the stationary kernel by a rotation-dependent phase factor. In the thesis, this is the approximation

\[
G_{\mathrm{sc}} \approx G_0 \exp\!\left(i\,\frac{\Omega}{nc}\,\hat{k}\,(\mathbf r \times \mathbf r')\right),
\]

together with the corresponding reversed-rotation kernel where the sign of \(\Omega\) is flipped.

These two files are the main entry points if you want to understand how the rotating kernel is plugged into BemTool's usual operator and potential framework.

## Relation to the thesis

This code accompanies the thesis *Boundary Element Method for Electromagnetic Scattering from Rotating Bodies*. The mathematical model, boundary integral formulation, and the test problems implemented here are derived in the thesis, in particular:

- the reduced scalar rotating equation and its Green's function
- the boundary integral operators built from the modified kernel
- the PEC exterior problem
- the dielectric transmission problem
- the Galerkin discretization in BemTool
- the rotating-cylinder numerical tests

The repository should therefore be read together with the thesis if the analytical background is needed.

## Repository structure

A simplified overview of the relevant structure is:

    .
    ├── bemtool
    │   ├── operator
    │   │   ├── rotating_helmholtz_fast_op.hpp
    │   │   ├── rotating_helmholtz_op.hpp
    │   │   ├── rotating_helmholtz_singrem_op.hpp
    │   │   └── rotating_helmholtz_singrem_real_op.hpp
    │   ├── potential
    │   │   ├── rotating_helmholtz_fast_pot.hpp
    │   │   ├── rotating_helmholtz_singrem_pot.hpp
    │   │   └── rotating_helmholtz_singrem_real_pot.hpp
    │   └── ...
    ├── rotation
    │   ├── dielectric_cylinder.cpp
    │   ├── pec_cylinder.cpp
    │   ├── helpers.hpp
    │   ├── plot.py
    │   ├── plotting_isocontour.py
    │   ├── plotting_pec.py
    │   ├── plotting_results.py
    │   ├── plotting_scaling.py
    │   └── plotting_stationary.py
    ├── CMakeLists.txt
    └── README.md

## Numerical examples

The directory `rotation/` contains the example programs and plotting scripts used for the thesis experiments.

### `rotation/pec_cylinder.cpp`

Implements the **PEC cylinder** scattering tests. This corresponds to the exterior boundary-value problem discussed in the thesis for the perfectly conducting cylinder. Depending on the polarization and formulation, the code solves Dirichlet- or Neumann-type boundary integral equations and evaluates the scattered and total fields.

### `rotation/dielectric_cylinder.cpp`

Implements the **dielectric cylinder** transmission tests. This corresponds to the coupled interior-exterior problem in which the interface conditions connect the traces across the boundary.

### Plotting scripts

The Python files in `rotation/` were used to post-process the field data and generate the numerical plots shown in the thesis, including stationary and rotating comparisons, scaling studies, and isocontours.

## Older experimental files

Besides the `fast` rotating implementation, the repository also contains older prototype files such as

- `rotating_helmholtz_op.hpp`
- `rotating_helmholtz_singrem_op.hpp`
- `rotating_helmholtz_singrem_real_op.hpp`
- `rotating_helmholtz_singrem_pot.hpp`
- `rotating_helmholtz_singrem_real_pot.hpp`

These belong to earlier implementation attempts and experiments. In particular, some of them follow older ideas based more directly on **Fourier-expansion-style constructions** rather than the final fast Green-function implementation used for the numerical results. They are kept for reference and comparison, but the files with the `fast` suffix are the relevant ones for reproducing the thesis computations.

## Connection to the equations in the thesis

This fork is intended as the implementation counterpart of the reduced rotating BEM formulation developed in the thesis. At a high level, the mapping is:

- **Chapter 2**: derivation of the rotating reduced scalar operator and its Green's function
- **Chapter 3**: layer potentials, boundary integral operators, and Galerkin discretization
- **Chapter 4**: numerical experiments for PEC and dielectric cylinders

The code in `bemtool/operator` and `bemtool/potential` corresponds to the kernel and operator level, while the code in `rotation/` corresponds to the concrete test problems and numerical experiments built from those operators.

## Building

A `CMakeLists.txt` file was added in this fork to make building the examples simpler. This was not part of the original minimal setup and is included here for convenience so the rotating test programs can be compiled more easily.

A typical out-of-source build is:

```bash
mkdir -p build
cd build
cmake ..
make -j
```

Depending on your environment, you may also need Eigen and the usual dependencies required by BemTool.

## Notes

- This repository is a **research codebase** used for a master's thesis.
- The focus is on the reduced scalar rotating model for cylindrical geometries.
- The implementation is primarily a **proof of concept** for the boundary element treatment of the rotating kernel, rather than a polished general-purpose package.