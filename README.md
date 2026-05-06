# Rotating BEMTool Fork

This repository is a research fork of [BEMTool](https://github.com/xclaeys/BemTool) used for the MSc thesis project **Boundary Element Method for Electromagnetic Scattering from Rotating Bodies**.

The fork keeps the original BEMTool structure for meshes, finite-element spaces, quadrature, operator assembly, and potential evaluation. The main additions are rotation-aware Helmholtz kernels and example drivers for scattering by circular cylinders.

## Scope

The implementation targets two-dimensional scalar Helmholtz models obtained from the reduction of electromagnetic scattering by a cylinder aligned with the z-axis. The main examples are:

- scattering by a perfectly electrically conducting (PEC) cylinder,
- scattering by a homogeneous dielectric cylinder,
- comparison of stationary and rotating cases.

The rotating model is implemented as a kernel-level modification: the stationary free-space Helmholtz Green's function is replaced by a first-order rotation-aware Green's function, while the surrounding BEMTool assembly machinery is reused.

## Main additions

The rotation-specific implementation is concentrated in:

```text
bemtool/potential/rotating_helmholtz_fast_pot.hpp
bemtool/operator/rotating_helmholtz_fast_op.hpp
```

These files define the rotating Helmholtz kernel tag `RH` and the corresponding potentials and boundary integral operators.

The suffix `fast` refers to the Steinberg-type first-order phase-factor approximation of the rotating Green's function, not to the truncated Sum.

Convenience aliases are provided so that the rotating kernel can be used in the same style as the built-in BEMTool kernels. Examples include:

- `RH_SL_2D_P0`, `RH_SL_2D_P1` for rotating single-layer potentials,
- `RH_DL_2D_P0`, `RH_DL_2D_P1` for rotating double-layer potentials,
- `RH_SL_2D_P1xP1` for single-layer operators,
- `RH_DL_2D_P1xP1` for double-layer operators,
- `RH_TDL_2D_P1xP1` for adjoint double-layer operators,
- `RH_HS_2D_P1xP1` for hypersingular operators.

This makes it possible to switch between stationary and rotating Helmholtz formulations with minimal changes at the assembly level.

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
├── tests/                    # Consistency and regression tests
├── CMakeLists.txt
├── build_and_test.sh         # Build and test the project
└── README.md
```

## Numerical examples

The main example drivers are in `rotation/`.

### `rotation/pec_cylinder.cpp`

Implements scattering tests for a PEC circular cylinder. The scalar model corresponds to the usual TM/TE reductions of the electromagnetic cylinder problem:

- TM-like Dirichlet formulation for the total field,
- TE-like Neumann formulation for the total field.

### `rotation/dielectric_cylinder.cpp`

Implements transmission tests for a homogeneous dielectric cylinder. The formulation couples interior and exterior fields through interface conditions involving the material parameters.

### Plotting

The scripts in `rotation/plotting/` post-process numerical output and generate plots for:

- scattered and total fields,
- stationary versus rotating comparisons,
- isocontours,
- scaling experiments.

## Building

This fork provides a top-level `CMakeLists.txt` for building the examples and tests.

You can also use the helper script from the repository root:

```bash
./build_and_test.sh
```

To build the project into a directory `build` and run the tests (may take up to 30min).

Dependencies are those required by BEMTool and the added examples, in particular a C++ compiler with C++17 support, CMake, Eigen, and Boost.

## Running examples

After building, run the example executables from the build directory. See `rotation/scripts/` on what arguments to pass the them.
Generated field data can be post-processed with the Python scripts in `rotation/plotting/`.

## Tests

The `tests/` directory contains consistency checks for the rotating implementation. See the README in `tests` for more information.

## License

This repository is based on BEMTool and keeps the original GPL license terms. See `LICENSE` for details.