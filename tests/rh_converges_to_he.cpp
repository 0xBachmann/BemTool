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

static constexpr auto mesh_file = "mesh/circle.msh";

static inline Real rel_err(const Cplx& a, const Cplx& b)
{
  const Real na = std::abs(a);
  const Real nb = std::abs(b);
  const Real denom = std::max<Real>({na, nb, 1e-30});
  return std::abs(a - b) / denom;
}

// Observed convergence order when the control parameter is halved each step:
// If e(Omega) ~ C * Omega^p, then p = log(e_i/e_{i+1}) / log(2).
static inline Real observed_order_halving(const Real e_i, const Real e_ip1)
{
  const Real ei = std::max<Real>(e_i, 1e-300);
  const Real eip = std::max<Real>(e_ip1, 1e-300);
  return std::log(ei / eip) / std::log(2.0);
}

TEST(RotatingHelmholtz, RH_SL_converges_to_HE_SL_when_Omega0)
{
  // ---- Load a standard boundary mesh (same as test2D) ----
  Geometry node(mesh_file);
  Mesh1D mesh;
  mesh.Load(node);
  Orienting(mesh);

  const int nb_elt = NbElt(mesh);
  ASSERT_GT(nb_elt, 4);

  // Pick two well-separated elements (avoid near-singular behavior)
  const int ix = 0;
  const int iy = nb_elt / 2;

  // Choose basis types (must exist in your BemTool build).
  // P1xP1 matches the standard HE_SL_2D_P1xP1 you showed earlier.
  using PhiX = P1_2D;
  using PhiY = P1_2D;

  // Instantiate the HE single-layer kernel (reference)
  const Real kappa = 1.0; // choose moderate k; too large may need bigger M
  HE_SL_2D_P1xP1 he(mesh, mesh, kappa);
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

  std::vector<int> Ms = {10, 20, 40, 80};

  Real prev_err = std::numeric_limits<Real>::infinity();
  Real last_err = prev_err;

  for (int M : Ms)
  {
    RH_SL_2D_P1xP1 rh(mesh, mesh, khat, Omega, c, M, keep_D2);
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
  EXPECT_LT(last_err, 1e-8) << "RH SL did not converge close enough to HE SL at largest M.";
}

TEST(RotatingHelmholtz, RH_DL_converges_to_HE_DL_when_Omega0)
{
  // ---- Load a standard boundary mesh (same as test2D) ----
  Geometry node(mesh_file);
  Mesh1D mesh;
  mesh.Load(node);
  Orienting(mesh);

  const int nb_elt = NbElt(mesh);
  ASSERT_GT(nb_elt, 4);

  // Pick two well-separated elements (avoid near-singular behavior)
  const int ix = 0;
  const int iy = nb_elt / 2;

  // Choose basis types (must exist in your BemTool build).
  // P1xP1 matches the standard HE_DL_2D_P1xP1 you showed earlier.
  using PhiX = P1_2D;
  using PhiY = P1_2D;

  // Instantiate the HE single-layer kernel (reference)
  const Real kappa = 1.0; // choose moderate k; too large may need bigger M
  HE_DL_2D_P1xP1 he(mesh, mesh, kappa);
  he.Assign(ix, iy);

  // Choose quadrature points in reference element coordinates.
  // In BemTool, Trait::Rdx/Rdy are small vectors (dimension 1 for 1D elements).
  using TraitHE = BIOpKernelTraits<HE, DL_OP, 2, PhiX, PhiY>;
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

  std::vector<int> Ms = {10, 20, 40, 80};

  Real prev_err = std::numeric_limits<Real>::infinity();
  Real last_err = prev_err;

  for (int M : Ms)
  {
    RH_DL_2D_P1xP1 rh(mesh, mesh, khat, Omega, c, M, keep_D2);
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
  EXPECT_LT(last_err, 1e-8) << "RH DL did not converge close enough to HE DL at largest M.";
}

TEST(RotatingHelmholtz, RH_SL_converges_to_HE_SL_as_Omega_halves)
{
  // ---- Load a standard boundary mesh (same as test2D) ----
  Geometry node(mesh_file);
  Mesh1D mesh;
  mesh.Load(node);
  Orienting(mesh);

  const int nb_elt = NbElt(mesh);
  ASSERT_GT(nb_elt, 4);

  // Pick two well-separated elements (avoid near-singular behavior)
  const int ix = 0;
  const int iy = nb_elt / 2;

  using PhiX = P1_2D;
  using PhiY = P1_2D;

  // Reference (Omega = 0) kernel
  const Real kappa = 1.0;
  HE_SL_2D_P1xP1 he(mesh, mesh, kappa);
  he.Assign(ix, iy);

  using TraitHE = BIOpKernelTraits<HE, SL_OP, 2, PhiX, PhiY>;
  typename TraitHE::Rdx tx;
  typename TraitHE::Rdy ty;
  tx[0] = 0.37;
  ty[0] = 0.61;

  const Cplx he_val = he(tx, ty, /*kx=*/0, /*ky=*/0);

  // RH parameters
  const Real c = 1.0;
  const bool keep_D2 = false;
  const Real khat = kappa;

  // Make truncation error negligible so we're measuring Omega -> 0 convergence.
  // (If this is too slow in CI, reduce M but keep it fixed across Omega.)
  const int M = 20;

  // Omega sequence: 1, 1/2, 1/4, ...
  std::vector<Real> Omegas;
  {
    Real Om = 1.0 / (2 * M + 1);
    for (int i = 0; i < 32; ++i)
    {
      Omegas.push_back(Om);
      Om *= 0.5;
    }
  }

  std::vector<Real> errs;
  errs.reserve(Omegas.size());

  for (Real Omega : Omegas)
  {
    RH_SL_2D_P1xP1 rh(mesh, mesh, khat, Omega, c, M, keep_D2);
    rh.Assign(ix, iy);

    const Cplx rh_val = rh(tx, ty, /*kx=*/0, /*ky=*/0);

    const Real e = rel_err(rh_val, he_val);
    errs.push_back(e);
  }

  // Expect a clear decreasing trend as Omega -> 0.
  // Allow a little wiggle due to floating point + any residual truncation.
  for (std::size_t i = 0; i + 1 < errs.size(); ++i)
  {
    EXPECT_LE(errs[i + 1], errs[i] * 1.25)
      << "Error did not decrease when halving Omega: Omega=" << Omegas[i]
      << " -> " << Omegas[i + 1] << ", err=" << errs[i] << " -> " << errs[i + 1];
  }

  // Estimate observed order p in e ~ C * Omega^p.
  // Use the last few steps (small Omega) for the most asymptotic behavior.
  std::vector<Real> ps;
  for (std::size_t i = 0; i + 1 < errs.size(); ++i)
  {
    ps.push_back(observed_order_halving(errs[i], errs[i + 1]));
  }
  // Robust summary: median of the last 3 orders
  const std::size_t n = ps.size();
  ASSERT_GE(n, 4u);
  std::vector<Real> tail = {ps[n - 3], ps[n - 2], ps[n - 1]};
  std::sort(tail.begin(), tail.end());
  const Real p_med = tail[1];
  std::cout << "Observed Omega->0 convergence order " << p_med << std::endl;

  // Theoretical order depends on your exact RH formulation; in most consistent
  // observer expansions, the leading correction is at least O(Omega).
  // This asserts you see *at least* first-order convergence.
  EXPECT_GT(p_med, 0.8)
    << "Observed Omega->0 convergence order (median of last 3 halvings) too low: p_med=" << p_med;

  // Also ensure we're actually close for the smallest Omega.
  EXPECT_LT(errs.back(), 1e-6)
    << "RH SL not sufficiently close to HE SL at smallest Omega=" << Omegas.back() << ".";
}

TEST(RotatingHelmholtz, RH_DL_converges_to_HE_DL_as_Omega_halves)
{
  // ---- Load a standard boundary mesh (same as test2D) ----
  Geometry node(mesh_file);
  Mesh1D mesh;
  mesh.Load(node);
  Orienting(mesh);

  const int nb_elt = NbElt(mesh);
  ASSERT_GT(nb_elt, 4);

  // Pick two well-separated elements (avoid near-singular behavior)
  const int ix = 0;
  const int iy = nb_elt / 2;

  using PhiX = P1_2D;
  using PhiY = P1_2D;

  // Reference (Omega = 0) kernel
  const Real kappa = 1.0;
  HE_DL_2D_P1xP1 he(mesh, mesh, kappa);
  he.Assign(ix, iy);

  using TraitHE = BIOpKernelTraits<HE, DL_OP, 2, PhiX, PhiY>;
  typename TraitHE::Rdx tx;
  typename TraitHE::Rdy ty;
  tx[0] = 0.37;
  ty[0] = 0.61;

  const Cplx he_val = he(tx, ty, /*kx=*/0, /*ky=*/0);

  // RH parameters
  const Real c = 1.0;
  const bool keep_D2 = false;
  const Real khat = kappa;
  const int M = 20;

  std::vector<Real> Omegas;
  {
    Real Om = 1.0 / (2 * M + 1);
    for (int i = 0; i <32; ++i)
    {
      Omegas.push_back(Om);
      Om *= 0.5;
    }
  }

  std::vector<Real> errs;
  errs.reserve(Omegas.size());

  for (Real Omega : Omegas)
  {
    RH_DL_2D_P1xP1 rh(mesh, mesh, khat, Omega, c, M, keep_D2);
    rh.Assign(ix, iy);

    const Cplx rh_val = rh(tx, ty, /*kx=*/0, /*ky=*/0);
    const Real e = rel_err(rh_val, he_val);
    errs.push_back(e);
  }

  for (std::size_t i = 0; i + 1 < errs.size(); ++i)
  {
    EXPECT_LE(errs[i + 1], errs[i] * 1.25)
      << "Error did not decrease when halving Omega: Omega=" << Omegas[i]
      << " -> " << Omegas[i + 1] << ", err=" << errs[i] << " -> " << errs[i + 1];
  }

  std::vector<Real> ps;
  for (std::size_t i = 0; i + 1 < errs.size(); ++i)
  {
    ps.push_back(observed_order_halving(errs[i], errs[i + 1]));
  }
  const std::size_t n = ps.size();
  ASSERT_GE(n, 4u);
  std::vector<Real> tail = {ps[n - 3], ps[n - 2], ps[n - 1]};
  std::sort(tail.begin(), tail.end());
  const Real p_med = tail[1];

  std::cout << "Observed Omega->0 convergence order " << p_med << std::endl;

  EXPECT_GT(p_med, 0.8)
    << "Observed Omega->0 convergence order (median of last 3 halvings) too low: p_med=" << p_med;

  EXPECT_LT(errs.back(), 1e-6)
    << "RH DL not sufficiently close to HE DL at smallest Omega=" << Omegas.back() << ".";
}

