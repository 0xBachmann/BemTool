# Non-rotating Rotating-Helmholtz Consistency Tests

This directory contains consistency tests comparing the rotating Helmholtz
boundary integral operators against BemTool's standard Helmholtz operators in
the non-rotating limit (`rh_equals_he_nonrotating.cpp`), as well as comparing derivatives of the Green's
function to finite differences (`normal_deivative_G.cpp`).

The main purpose of these tests is to verify that the rotating implementation
reduces to the Helmholtz implementation when the rotation parameter is set to
zero, modulo the sign conventions of the Green function and the derivative
conventions used by BemTool.

## Green-function convention

BemTool's two-dimensional Helmholtz implementation uses the outgoing Green
function (see `doc/solutions-analytiques.pdf`)

$$
G_{\mathrm{HE}}(x,y)
=
\frac{i}{4}H_0^{(1)}(\kappa |x-y|).
$$

In the non-rotating limit, the rotating Helmholtz kernel used in this thesis
reduces to

$$
G_{\mathrm{RH},0}(x,y)
=
\frac{1}{4i}H_0^{(1)}(\kappa |x-y|)
=-\frac{i}{4}H_0^{(1)}(\kappa |x-y|)
=-\,G_{\mathrm{HE}}(x,y).
$$

Thus,

$$
G_{\mathrm{RH},0}=-G_{\mathrm{HE}}.
$$

Thus we expect
```cpp
V_rh == V_he
Kp_rh === Kp_he
```

## Derivative convention for the double-layer operator

BemTool's double-layer convention differentiates the Green kernel with respect
to the argument $x-y$:

$$
DL(p)(x)
=
\int_\Gamma n(y)\cdot \nabla G(x-y)\,p(y)\,d\sigma(y).
$$

This is not the same as differentiating with respect to the source coordinate
$y$, since

$$
\nabla_y G(x-y)
=-\nabla_{x-y}G(x-y).
$$

Thus we expect
```cpp
K_rh == K_he
```

and the according signs for the single and double layer potentials.