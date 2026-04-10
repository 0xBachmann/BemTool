#include <gtest/gtest.h>
#include "../bemtool/tools.hpp"

#include <vector>
#include <complex>
#include <cmath>
#include <algorithm>

using namespace bemtool;

static constexpr auto mesh_file = "mesh/circle.msh";

static inline Real rel_l2(const std::vector<Cplx>& a,
                          const std::vector<Cplx>& b)
{
  Real num = 0.0, den = 0.0;
  for (size_t i=0;i<a.size();++i){
    num += std::norm(a[i]-b[i]);
    den += std::norm(a[i]);
  }
  return std::sqrt(num / std::max(den, 1e-30));
}

// ------------------------------------------------------------------
//  Small helper: Potential wrapper that forwards kernel constructor
//  arguments (BemTool's Potential<> only forwards (mesh,k)).
// ------------------------------------------------------------------
template <typename KernelType>
class PotentialEx {
public:
  static const int dimy = KernelType::Trait::dimy;
  using KernelTypeTrait = typename KernelType::Trait;
  using MeshY = typename KernelTypeTrait::MeshY;
  using RdY   = typename KernelTypeTrait::Rdy;
  using MatType = typename KernelTypeTrait::MatType;
  using QuadType = QuadPot<dimy>;

  template <class... Args>
  PotentialEx(const MeshY& my, Args&&... args)
    : meshy(my)
    , nodey(GeometryOf(my))
    , ker(my, std::forward<Args>(args)...)
    , qr(10) {}

  const MatType& operator()(const R3& x, const int& jy) {
    const std::vector<RdY>&  t  = qr.GetPoints();
    const std::vector<Real>& w  = qr.GetWeights();
    mat = 0;
    ker.Assign(x, jy);
    for (size_t j = 0; j < w.size(); ++j) mat += w[j] * ker(x, t[j]);
    return mat;
  }

  const Cplx& operator()(const R3& x, const N2& Iy) {
    const std::vector<RdY>&  t  = qr.GetPoints();
    const std::vector<Real>& w  = qr.GetWeights();
    val = 0;
    ker.Assign(x, Iy[0]);
    for (size_t j = 0; j < w.size(); ++j) val += w[j] * ker(x, t[j], Iy[1]);
    return val;
  }

private:
  const MeshY& meshy;
  const Geometry& nodey;
  KernelType ker;
  QuadType qr;
  MatType mat{};
  Cplx val{};
};

// ------------------------------------------------------------------
//  Evaluate SL potential at an off-boundary point using BemTool kernels.
//  This matches the P1 assembly convention: sum element-local basis
//  contributions weighted by the global dofs.
// ------------------------------------------------------------------
static Cplx eval_sl_potential_HE(const Mesh1D& mesh, const Dof<P1_1D>& dof,
                                 const std::vector<Cplx>& mu, Real kappa,
                                 const R3& x)
{
  PotentialEx<HE_SL_2D_P1> pot(mesh, kappa);
  Cplx u = 0.0;
  for (int iy = 0; iy < NbElt(mesh); ++iy) {
    const N2& gd = dof[iy];
    u += pot(x, N2_(iy, 0)) * mu[gd[0]];
    u += pot(x, N2_(iy, 1)) * mu[gd[1]];
  }
  return u;
}

static Cplx eval_sl_potential_RH(const Mesh1D& mesh, const Dof<P1_1D>& dof,
                                 const std::vector<Cplx>& mu,
                                 Real khat, Real Omega,
                                 Real c, int M, bool keep_D2,
                                 const R3& x)
{
  PotentialEx<RH_SL_2D_P1> pot(mesh, khat, Omega, c, M, keep_D2);
  Cplx u = 0.0;
  for (int iy = 0; iy < NbElt(mesh); ++iy) {
    const N2& gd = dof[iy];
    u += pot(x, N2_(iy, 0)) * mu[gd[0]];
    u += pot(x, N2_(iy, 1)) * mu[gd[1]];
  }
  return u;
}

// Assemble a dense matrix using the same pattern you used in test2D.cpp:
// A(j,k) += Op(dof.ToElt(j), dof.ToElt(k));
template<class OpType>
static EigenDense assemble_dense(const Mesh1D& mesh, const Dof<P1_1D>& dof,
                                 OpType& Op)
{
  const int n = NbDof(dof);
  EigenDense A(n,n); Clear(A);
  progress bar("Assemblage", n);
  std::cout << "n = " << n << std::endl;
  for(int j=0; j<n; ++j){
    std::cout << j << std::endl;
    bar++;
    for(int k=0; k<n; ++k){
      A(j,k) += Op(dof.ToElt(j), dof.ToElt(k));
    }
  }
  bar.end();
  return A;
}

TEST(RotatingHelmholtz, RH_solution_matches_HE_solution_off_boundary_when_Omega0)
{
  Geometry node(mesh_file);
  Mesh1D mesh; mesh.Load(node,1);
  Orienting(mesh);

  Dof<P1_1D> dof(mesh);
  const int n = NbDof(dof);
  ASSERT_GT(n, 8);

  const Real kappa = 1.0;

  // RH parameters
  const Real Omega = 0.0;
  const Real c = 1.0;
  const bool keep_D2 = false;
  const Real khat = kappa;
  const int  M = 5; // big truncation for the “solution-level” test

  // Choose a *stable* operator to invert for this test.
  // Easiest: A = I + V (single-layer). (You can swap to CFIE later.)
  using HE_V = HE_SL_2D_P1xP1;
  using RH_V = RH_SL_2D_P1xP1;

  BIOp<HE_V> Vhe(mesh, mesh, kappa);
  BIOp<RH_V> Vrh(mesh, mesh, khat, Omega, c, M, keep_D2);

  EigenDense Ahe = assemble_dense(mesh, dof, Vhe);
  EigenDense Arh = assemble_dense(mesh, dof, Vrh);

  // A := I + V  (simple, deterministic)
  for(int i=0;i<n;++i){ Ahe(i,i) += 1.0; Arh(i,i) += 1.0; }

  // Manufactured solution on boundary (deterministic, smooth-ish)
  std::vector<Cplx> mu_ref(n);
  for(int j=0;j<NbElt(mesh);++j){
    const N2& jdof = dof[j];
    const array<2,R3> xdof = dof(j);
    for(int a=0;a<2;++a){
      // example: (x + i y)^2
      mu_ref[jdof[a]] = std::pow(xdof[a][0] + iu*xdof[a][1], 2);
    }
  }

  // rhs = A_he * mu_ref
  EigenDense::VectorType rhs(n);
  for(int i=0;i<n;++i){
    Cplx s = 0.0;
    for(int j=0;j<n;++j) s += Ahe(i,j) * mu_ref[j];
    rhs[i] = s;
  }

  // Solve both
  EigenDense::VectorType mu_he_ev(n), mu_rh_ev(n);
  lu_solve(Ahe, rhs, mu_he_ev);
  lu_solve(Arh, rhs, mu_rh_ev);

  std::vector<Cplx> mu_he(n), mu_rh(n);
  for(int i=0;i<n;++i){ mu_he[i] = mu_he_ev[i]; mu_rh[i] = mu_rh_ev[i]; }

  // Boundary unknowns should match
  EXPECT_LT(rel_l2(mu_he, mu_rh), 1e-8);

  // Off-boundary field comparison on a ring of target points
  const int Nt = 128;
  const Real R = 2.0; // keep away from boundary to avoid near-singular quadrature
  std::vector<Cplx> u_he(Nt), u_rh(Nt);

  for(int p=0;p<Nt;++p){
    Real t = 2.0*M_PI*p / Nt;
    R3 x = R3_(R*std::cos(t), R*std::sin(t), 0.0);

    u_he[p] = eval_sl_potential_HE(mesh, dof, mu_he, kappa, x);
    u_rh[p] = eval_sl_potential_RH(mesh, dof, mu_rh, khat, Omega, c, M, keep_D2, x);
  }

  EXPECT_LT(rel_l2(u_he, u_rh), 1e-8);
}