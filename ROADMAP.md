# Roadmap

This file collects possible follow-up work for the rotating BEMTool fork. These items are not required to build the repository or reproduce the proof-of-concept experiments from the thesis, but they would improve validation, robustness, and scope.

## Implementation and tests

- [ ] Add end-to-end PEC and dielectric cylinder tests.
  - Verify boundary conditions on the circular interface: Dirichlet for the TM-type branch and Neumann for the TE-type branch.
  - Use the weak hypersingular implementation, for example `HS_WEAK` / `RH_HS_WEAK_*`, in the PEC and dielectric cylinder examples whenever the hypersingular operator is needed.
  - For the dielectric case, compare stationary results with analytical Fourier--Bessel reference data.
  - Track condition numbers and residuals for each formulation.

## Numerical validation

- [ ] Run a small convergence study on circular meshes.
  - Compare field values, boundary traces, and residuals under mesh refinement.
  - Separate discretization error from post-processing error.

- [ ] Investigate conditioning of the assembled systems.
  - Record condition numbers for first-kind and second-kind formulations.
  - Test whether the weak hypersingular formulation improves stability in TE/Neumann-type experiments.
  - Compare the weak hypersingular formulation against the current strong-form hypersingular assembly where possible, then avoid the strong-form variant in the main PEC and dielectric examples.

## Future extensions

- [ ] Treat the full horizontal 1-form equation.
  - The current implementation focuses on the scalar reduced equation.
  - A future implementation should handle the component coupling in the transverse 1-form problem; see thesis Section 5.1.

- [ ] Generalize beyond centered circular cylinders.
  - For non-circular boundaries, the extra boundary term involving the rotational drift generally no longer vanishes.
  - The missing boundary contribution should be added before claiming support for general shapes; see thesis Section 5.2.

- [ ] Explore higher-order rotational corrections.
  - The current kernel uses the first-order / slow-rotation approximation.
  - Higher-order terms may be relevant for larger angular velocities, larger radii, or sensitive observables; see thesis Section 5.3.

- [ ] Consider fast BEM acceleration.
  - The current priority is correctness and validation.
  - Once the kernels and weak operators are tested, investigate H-matrix or fast-multipole style acceleration; see thesis Section 5.5.