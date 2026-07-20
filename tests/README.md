# Rotating Helmholtz Consistency Tests

This directory contains small automated tests for the rotating Helmholtz implementation used in the thesis software companion.

The tests are not intended to reproduce the full numerical experiments from Chapter 4 of the thesis. Instead, they check local consistency properties that should remain true after installation or code changes.

## Test files

| Test | Purpose |
| --- | --- |
| `rh_equals_he_nonrotating.cpp` | Compares rotating Helmholtz boundary integral operators with BEMTool's standard Helmholtz operators in the non-rotating limit. |
| `normal_deivative_G.cpp` | Compares analytical normal derivatives of the rotating Green's function with finite-difference approximations. |

The relevant mathematical background is discussed in the thesis in Sections 2.9 and 3.2--3.5, with implementation conventions described in Section 3.9 and derivative formulas collected in Appendix A.9.

## Running the tests

From the repository root, run:

```bash
./build_and_test.sh
```

or build manually and then run CTest:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

## Green-function convention

BEMTool's two-dimensional Helmholtz implementation uses the outgoing Green's function

$$
G_{\mathrm{HE}}(x,y)
= \frac{i}{4} H_0^{(1)}\!\left(\kappa |x-y|\right).
$$

In the non-rotating limit, the rotating Helmholtz kernel used in the thesis reduces to

$$
G_{\mathrm{RH},0}(x,y)
= \frac{1}{4i} H_0^{(1)}\!\left(\kappa |x-y|\right)
= -\frac{i}{4} H_0^{(1)}\!\left(\kappa |x-y|\right)
= -G_{\mathrm{HE}}(x,y).
$$

Equivalently,

$$
G_{\mathrm{RH},0} + G_{\mathrm{HE}} = 0.
$$

The non-rotating consistency test therefore checks the rotating implementation against the standard Helmholtz implementation with the sign convention used in the code.

## Derivative convention for double-layer operators

BEMTool's double-layer convention differentiates the Green kernel with respect to the argument $x-y$:

$$
\Psi_{\mathrm{DL}} p(x)
= \int_\Gamma n(y) \cdot \nabla G(x-y)\,p(y)\,d\sigma(y).
$$

This differs by a sign from differentiating with respect to the source coordinate $y$, because

$$
\nabla_y G(x-y) = -\nabla_{x-y} G(x-y).
$$

For this reason, the tests should always be interpreted together with the sign conventions in `rh_equals_he_nonrotating.cpp` and `normal_deivative_G.cpp`. The purpose of the tests is not merely to compare raw matrix entries, but to verify that the implemented rotating kernels, normal derivatives, and BEMTool operator conventions agree in the non-rotating limit.
