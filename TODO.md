# TODO

Short list of follow-up tasks for the rotating BEMTool fork.

## Implementation and tests

- [ ] Add end-to-end PEC and dielectric cylinder tests.
    - Verify boundary conditions on the circle: Dirichlet for the TM-type branch and Neumann for the TE-type branch.
    - Use the `HS_WEAK` / `RH_HS_WEAK_*` hypersingular implementation in the PEC and dielectric cylinder examples where the hypersingular operator is needed, instead of relying on the strong-form `HS_OP`.
    - For the dielectric case, compare stationary results with analytical Fourier--Bessel reference data.
    - Track condition numbers and residuals for each formulation.

## Numerical validation

- [ ] Run a small convergence study on circular meshes.
    - Compare field values, boundary traces, and residuals under mesh refinement.
    - Separate discretization error from post-processing error.

- [ ] Investigate conditioning of the assembled systems.
    - Record condition numbers for first-kind and second-kind formulations.
    - Test whether the weak hypersingular formulation `HS_WEAK` improves stability in TE/Neumann-type experiments.
    - Compare `HS_WEAK` against the current strong-form hypersingular assembly where possible, then remove or avoid strong-form usage in the main PEC and dielectric examples.

## Future extensions

- [ ] Treat the full horizontal 1-form equation.
    - The current implementation focuses on the scalar reduced equation.
    - A future implementation should handle the component coupling in the transverse 1-form problem.

- [ ] Generalize beyond centered circular cylinders.
    - For non-circular boundaries, the extra boundary term involving the rotational drift generally no longer vanishes.
    - Add the missing boundary contribution before claiming support for general shapes.

- [ ] Explore higher-order rotational corrections.
    - The current kernel uses the first-order / slow-rotation approximation.
    - Higher-order terms may be relevant for larger `Omega`, larger radii, or sensitive observables.

- [ ] Consider fast BEM acceleration later.
    - The current priority is correctness and validation.
    - Once the kernels and weak operators are tested, investigate H-matrix or fast-multipole style acceleration.