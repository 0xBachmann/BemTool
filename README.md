# Rotating BEMTool Fork

This repository is a fork of **BemTool** used for the numerical experiments in the thesis *Boundary Element Method for Electromagnetic Scattering from Rotating Bodies*. For the general design, usage, and core boundary-element infrastructure of BemTool, please refer to the original project and its documentation. In this fork, BemTool is used as the generic BEM backbone: mesh handling, basis functions, quadrature, operator assembly, and potential evaluation remain in the BemTool style, while the main extension is the introduction of rotation-aware kernels for the reduced scalar scattering model.

## Purpose of this fork

The goal of this fork is to extend the standard 2D Helmholtz BEM workflow by replacing the stationary free-space Green's function with a **rotation-aware Green's function** for scattering by uniformly rotating cylindrical media. This follows the thesis framework and the project description: the rotating problem is treated as a modification of the kernel, while the surrounding BEM machinery is kept as close as possible to standard BemTool.

In particular, this repository was used to produce the numerical results for:

- scattering by a **PEC cylinder**
- scattering by a **dielectric cylinder**
- comparison between the **stationary** and **rotating** cases
## Main additions

The main implementation relevant to the rotating-medium model is contained in the following files:
TODO fast refers to taking steinbergs approximation instead of a truncated sum
- `bemtool/potential/rotating_helmholtz_fast_pot.hpp`
- `bemtool/operator/rotating_helmholtz_fast_op.hpp`

These files implement the **Green's function of a rotating medium** and the associated boundary integral operators in the first-order rotating approximation used in the thesis. The implementation follows the Steinberg-type approximation in which the rotating Green kernel is written as a correction of the stationary kernel by a rotation-dependent phase factor. In the thesis, this is the approximation

\[
G_{\mathrm{sc}} \approx G_0 \exp\!\left(i\,\frac{\Omega}{nc}\,\hat{k}\,(\mathbf r \times \mathbf r')\right),
\]

together with the corresponding reversed-rotation kernel where the sign of \(\Omega\) is flipped. This matches the thesis goal of introducing a rotation-aware Green kernel as a first-order correction to the standard Helmholtz kernel. :contentReference[oaicite:0]{index=0}

In the code, this rotating Helmholtz kernel is identified by the tag `RH` (for **Rotating Helmholtz**). The same tag is used consistently for both potentials and boundary integral operators.

A few convenient typedefs / aliases are introduced so the rotating kernel can be used in the same style as the standard BEMTool kernels. In particular, the potential file provides aliases such as

- `RH_SL_2D_P0`, ... for the rotating single-layer potential,
- `RH_DL_2D_P1`, ... for the rotating double-layer potential,

while the operator file introduces aliases such as

- `RH_SL_2D_P1xP0`, ... for the single-layer operator,
- `RH_DL_2D_P1xP1`, ... for the double-layer operator,
- `RH_TDL_2D_P0xP0`, ... for the "adjoint" double-layer operator,
- `RH_HS_2D_P0xP1`, ... for the hypersingular operator.

This makes it possible to swap the stationary Helmholtz kernel for the rotating one with only minimal changes at the assembly level, while keeping the surrounding BEMTool workflow unchanged.
## Repository structure

A simplified overview of the relevant structure is:

    .
    ├── bemtool
    │   ├── operator
    │   │   └── rotating_helmholtz_fast_op.hpp
    │   ├── potential
    │   │   └── rotating_helmholtz_fast_pot.hpp
    │   └── ...
    ├── rotation
    │   ├── dielectric_cylinder.cpp
    │   ├── pec_cylinder.cpp
    │   ├── helpers.hpp
    │   └── ...
    ├── CMakeLists.txt
    └── README.md

## Numerical examples

The directory `rotation/` contains the example programs and plotting scripts used for the thesis experiments.

### `rotation/pec_cylinder.cpp`

Implements the **PEC cylinder** scattering tests. This corresponds to the exterior boundary-value problem discussed in Section 3.6 of the thesis for the perfectly conducting cylinder. Depending on the polarization and formulation, the code solves Dirichlet- or Neumann-type boundary integral equations and evaluates the scattered and total fields.

### `rotation/dielectric_cylinder.cpp`

Implements the **dielectric cylinder** transmission tests. This corresponds to the coupled interior-exterior problem in which the interface conditions connect the traces across the boundary, discussed Section 3.7.

### Plotting scripts

The Python files in `rotation/` were used to post-process the field data and generate the numerical plots shown in the thesis, including stationary and rotating comparisons, scaling studies, and isocontours.

## Building

A `CMakeLists.txt` file was added in this fork to make building the examples simpler. This was not part of the original minimal setup and is included here for convenience so the rotating test programs can be compiled more easily.

A typical out-of-source build is:

```bash
mkdir -p build
cd build
cmake ..
make -j
```

Depending on your environment, you may also need Eigen, Boost and the usual dependencies required by BemTool.


In the non-rotating limit, the rotating Helmholtz kernel reduces to
\[
G_{\mathrm{rot},0}(x,y)
=
\frac{1}{4i}H_0^{(1)}(\kappa |x-y|)
=
-\frac{i}{4}H_0^{(1)}(\kappa |x-y|)
=
-\,G_{\mathrm{BT}}(x,y),
\]
where BemTool's Helmholtz implementation uses
\[
G_{\mathrm{BT}}(x,y)
=
\frac{i}{4}H_0^{(1)}(\kappa |x-y|).
\]
Thus, for operators depending linearly on the Green kernel and using the
same derivative conventions, the corresponding BEM matrices should agree
up to this global sign:
\[
A_{\mathrm{rot},0}
\approx - A_{\mathrm{BT}}.
\]
Consequently, the non-rotating consistency test should compare
\[
A_{\mathrm{rot},0}+A_{\mathrm{BT}}
\]
rather than
\[
A_{\mathrm{rot},0}-A_{\mathrm{BT}}.
\]
The observed equality of norms together with entries satisfying
\[
B_{ij}=-A_{ij}
\]
is therefore consistent with the Green-function convention difference, not a
discretization error.