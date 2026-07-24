# Roadmap

This file collects the principal open questions and follow-up tasks for the
rotating BEMTool fork. The current implementation is a proof of concept for the
scalar reduced problem developed in the thesis. The rotation-aware kernels and
the weak regularization of the hypersingular operator are implemented and
documented, but the complete formulation has not yet been validated
systematically.

The tasks below are ordered according to their logical dependencies. In
particular, the modelling framework and interface conditions should be settled
before the rotating boundary integral formulation is revised and subjected to
systematic numerical validation.

## 1. Clarify the modelling framework

- [ ] Decide how the exterior vacuum region should be described.
  - Formulate the exterior problem in the inertial frame, where the incident
    field is a simple plane wave but the interface conditions may be more
    involved.
  - Formulate the exterior problem in the rotating-observer framework, where
    the interface conditions may be simpler but the incident field must be
    transformed.
  - Determine to which approximation order the two descriptions are
    equivalent.
  - Compare their analytical consistency, implementation complexity, and
    numerical advantages and disadvantages.
  - Document the selected formulation and the assumptions under which it is
    valid.

## 2. Analyse the interface conditions

- [ ] Derive consistent transmission conditions at the material interface.
  - Distinguish discontinuities in material parameters from discontinuities in
    the observer structure.
  - Clarify the roles of material motion, observer motion, and motion of the
    interface itself.
  - Determine whether the standard local transmission conditions remain valid
    when the observer field is discontinuous.
  - Investigate whether additional, transformed, or non-local interface terms
    are required.
  - Verify that the derived conditions reduce to the standard stationary
    conditions in the appropriate limit.

## 3. Update the numerical formulation

- [ ] Adapt the boundary integral formulation to the conclusions of Sections 1
  and 2.
  - Update the interior and exterior representation formulas, trace variables,
    and transmission system.
  - Ensure that the incident field and interface data are expressed in the
    selected observer framework.
  - Revisit the Calderón systems and the scaling of the Dirichlet and conormal
    traces.
  - Update the PEC and dielectric cylinder examples accordingly.

- [ ] Systematically validate the weak hypersingular formulation.
  - Test self-, neighbouring-, near-singular, and well-separated element
    interactions.
  - Verify the relevant symmetry and adjoint identities in the stationary and
    rotating cases.
  - Perform mesh-refinement and convergence studies.
  - Compare against analytical or independently computed reference values.
  - Document the supported trial and test spaces.
  - Use the weak hypersingular implementation, for example
    `HS_WEAK` / `RH_HS_WEAK_*`, in the main PEC and dielectric examples
    whenever the hypersingular operator is required.

## 4. Investigate conditioning

- [ ] Determine the origin of the observed condition numbers.
  - Separate effects caused by the continuous formulation, the discrete spaces,
    matrix scaling, quadrature, and implementation.
  - Record condition numbers and residuals as functions of mesh size,
    wavenumber, angular velocity, and material contrast.
  - Compare first-kind and second-kind formulations.
  - Compare the weak hypersingular formulation with the existing strong-form
    assembly where this comparison is meaningful.
  - Check the influence of basis normalization and mass-matrix scaling.
  - Investigate suitable preconditioners or alternative integral formulations
    where necessary.

## 5. Validation -- consistency limits

- [ ] Verify the stationary limit.
  - Confirm that the rotating kernels and operators converge to their standard
    Helmholtz counterparts as the angular velocity tends to zero.
  - Compare boundary traces, fields, residuals, and assembled matrices.

- [ ] Verify the equal-material limit.
  - Confirm that no artificial scattering occurs when the material parameters
    are identical on both sides of the interface.
  - Repeat the corresponding test for the rotating reference case once the
    modelling framework has been clarified.
  - Use these tests to detect inconsistent interface conditions or trace
    scalings.

- [ ] Run convergence studies on circular meshes.
  - Compare field values, boundary traces, and residuals under mesh refinement.
  - Separate discretization, quadrature, and post-processing errors.

## 6. Validation -- reference solutions

- [ ] Validate the stationary cylinder problems.
  - Compare the PEC and dielectric results with analytical Fourier--Bessel
    solutions.
  - Test both Dirichlet/TM-type and Neumann/TE-type branches where applicable.
  - Report quantitative field, trace, and far-field errors.

- [ ] Validate the rotating formulation.
  - Identify analytical rotating-cylinder solutions or reliable perturbative
    results.
  - Compare the numerical solution with first-order predictions in the
    slow-rotation regime.
  - Verify the expected even and odd dependence on the angular velocity.
  - Establish the range of parameters for which the first-order rotating kernel
    remains accurate.

## 7. Further mathematical and numerical extensions

These extensions should be considered after the modeling and validation tasks
above have established a reliable baseline.

- [ ] Treat the full horizontal 1-form equation.
  - The current implementation focuses on the scalar reduced equation.
  - A future implementation should handle the component coupling in the
    transverse 1-form problem; see thesis Section 5.1.

- [ ] Generalize beyond centred circular cylinders.
  - For non-circular boundaries, the additional boundary term involving the
    rotational drift generally no longer vanishes.
  - Add the missing boundary contribution before claiming support for general
    shapes; see thesis Section 5.2.

- [ ] Explore higher-order rotational corrections.
  - The current kernel uses the first-order slow-rotation approximation.
  - Higher-order terms may be relevant for larger angular velocities, larger
    radii, or sensitive observables; see thesis Section 5.3.

- [ ] Consider fast BEM acceleration.
  - The current priority is correctness and validation.
  - Once the kernels and weak operators are validated, investigate hierarchical
    matrices, fast multipole methods, or related acceleration techniques; see
    thesis Section 5.5.