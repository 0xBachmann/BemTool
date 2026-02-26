// tests/test_rh_converges_to_he.cpp
#include <gtest/gtest.h>

#include "../bemtool/tools.hpp"

// HE kernels live here in BemTool
#include "../bemtool/operator/helmholtz_op.hpp"

// Your new RH kernels
#include "../bemtool/operator/rotating_helmholtz_op.hpp"

#include <complex>
#include <vector>
#include <cmath>
#include <algorithm>

using namespace bemtool;

static inline Real rel_err(const Cplx& a, const Cplx& b)
{
  const Real na = std::abs(a);
  const Real nb = std::abs(b);
  const Real denom = std::max<Real>({na, nb, (Real)1e-30});
  return std::abs(a - b) / denom;
}

TEST(RotatingHelmholtz, RH_SL_converges_to_HE_SL_when_Omega0)
{
  // ---- Load a standard boundary mesh (same as test2D) ----
  Geometry node("mesh/circle.msh");
  Mesh1D mesh;
  mesh.Load(node, /*physical_tag=*/1);
  Orienting(mesh);

  const int nb_elt = NbElt(mesh);
  ASSERT_GT(nb_elt, 4);

  // Pick two well-separated elements (avoid near-singular behavior)
  const int ix = 0;
  const int iy = nb_elt / 2;

  // Choose basis types (must exist in your BemTool build).
  // P1xP1 matches the standard HE_SL_2D_P1xP1 you showed earlier.
  using PhiX = P1;
  using PhiY = P1;

  // Instantiate the HE single-layer kernel (reference)
  const Real kappa = 6.0; // choose moderate k; too large may need bigger M
  BIOpKernel<HE, SL_OP, 2, PhiX, PhiY> he(mesh, mesh, kappa);
  he.Assign(ix, iy);

  // Choose quadrature points in reference element coordinates.
  // In BemTool, Trait::Rdx/Rdy are small vectors (dimension 1 for 1D elements).
  using TraitHE = BIOpKernelTraits<HE, SL_OP, 2, PhiX, PhiY>;
  typename TraitHE::Rdx tx;
  typename TraitHE::Rdy ty;
  tx[0] = 0.37;
  ty[0] = 0.61;

  // Evaluate a representative entry (kx=0, ky=0).
  // Since both kernels multiply by basis functions the same way,
  // comparing any fixed entry is fine.
  const Cplx he_val = he(tx, ty, /*kx=*/0, /*ky=*/0);

  // RH parameters: Omega=0 so kappa_m should be constant and the mode sum should reproduce HE as M grows.
  const Real Omega = 0.0;
  const Real c = 1.0;
  const bool keep_D2 = false;

  // IMPORTANT: Use khat that makes RH match HE when Omega=0.
  // If your theory uses kappa_m = kappa when Omega=0, set khat=kappa here.
  // (If your theory instead uses kappa_m^2 = -khat^2, then khat must be chosen accordingly,
  // and HE reference should use the matching complex kappa.)
  const Real khat = kappa;

  std::vector<int> Ms = {10, 20, 40, 80, 160};

  Real prev_err = std::numeric_limits<Real>::infinity();
  Real last_err = prev_err;

  for (int M : Ms)
  {
    BIOpKernel<RH, SL_OP, 2, PhiX, PhiY> rh(mesh, mesh, khat, Omega, c, M, keep_D2);
    rh.Assign(ix, iy);

    const Cplx rh_val = rh(tx, ty, /*kx=*/0, /*ky=*/0);
    const Real e = rel_err(rh_val, he_val);

    // We don't demand strict monotonicity (truncation can wiggle),
    // but error should trend down.
    // A weak check: by M>=40 it should not be *worse* than at M=10 by a lot.
    if (M >= 40)
    {
      EXPECT_LT(e, prev_err * 1.5) << "Error did not decrease reasonably at M=" << M;
    }

    prev_err = std::min(prev_err, e);
    last_err = e;
  }

  // Final accuracy target (adjust if your kappa or point choice changes)
  EXPECT_LT(last_err, 1e-8) << "RH did not converge close enough to HE at largest M.";
}
