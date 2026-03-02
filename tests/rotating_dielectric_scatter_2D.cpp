#include <iostream>
#include <vector>
#include <cmath>

#include "../bemtool/tools.hpp"
// Use BemTool Helmholtz outside
#include "../bemtool/operator/helmholtz_op.hpp"

using namespace bemtool;

// Assemble a P1 mass matrix on a 1D mesh (Galerkin identity operator)
static EigenDense AssembleMassP1(const Mesh1D& mesh, const Dof<P1_1D>& dof, const Real radius)
{
  const int nb_dof = NbDof(dof);
  EigenDense M(nb_dof, nb_dof);
  Clear(M);

  const int nb_elt = NbElt(mesh);
  for (int e = 0; e < nb_elt; ++e)
  {
    const N2& edof = dof[e];
    const auto& elt = mesh[e];

    // element length on unit mesh
    const Real L_unit = norm2(elt[1] - elt[0]);
    const Real L = radius * L_unit;

    // local P1 mass matrix on segment: L/6 * [[2,1],[1,2]]
    const Real a = L / 6.0;
    M(edof[0], edof[0]) += 2.0 * a;
    M(edof[0], edof[1]) += 1.0 * a;
    M(edof[1], edof[0]) += 1.0 * a;
    M(edof[1], edof[1]) += 2.0 * a;
  }
  return M;
}

// Incident plane wave u_inc(x) = exp(i k d·x), d = (1,0)
static inline Cplx u_inc_plane(const R3& x_phys, const Real kappa_out)
{
  return std::exp(iu * kappa_out * x_phys[0]);
}

int main(int argc, char* argv[])
{
  // --- user parameters (easy to swap to CLI later) ---
  const Real radius = (argc > 1) ? std::atof(argv[1]) : 0.7;   // cylinder radius (<1 if mesh is unit circle)
  const Real k0     = (argc > 2) ? std::atof(argv[2]) : 3.0;   // vacuum wavenumber
  const Real eps_in = (argc > 3) ? std::atof(argv[3]) : 2.25;  // relative permittivity inside
  const Real mu_in  = (argc > 4) ? std::atof(argv[4]) : 1.0;   // relative permeability inside
  const Real Omega  = (argc > 5) ? std::atof(argv[5]) : 0.2;   // rotation rate (your model units)
  const int  Mtrunc = (argc > 6) ? std::atoi(argv[6]) : 5;    // modal truncation

  // outside medium (vacuum-like)
  const Real eps_out = 1.0;
  const Real mu_out  = 1.0;

  const Real kappa_out = k0 * std::sqrt(eps_out * mu_out);
  const Real kappa_in  = k0 * std::sqrt(eps_in  * mu_in);

  // For your RH operator we pass khat (your effective k) and c.
  // In your thesis derivation you often use \hat{k} = k n / c. Here we keep it simple:
  // set khat := kappa_in and c := 1, so Ω is effectively in the same units as khat.
  const Real khat = kappa_in;
  const Real c    = 1.0;

  Geometry node("mesh/circle.msh");
  Mesh1D mesh; mesh.Load(node, 1);
  Orienting(mesh);

  Dof<P1_1D> dof(mesh);
  const int nb_dof = NbDof(dof);

  std::cout << "nb_dof:\t" << nb_dof << "\n";
  std::cout << "radius:\t" << radius << "\n";
  std::cout << "kappa_out:\t" << kappa_out << "\n";
  std::cout << "kappa_in (Helmholtz equiv):\t" << kappa_in << "\n";
  std::cout << "Omega:\t" << Omega << "\n";
  std::cout << "Mtrunc:\t" << Mtrunc << "\n";

  // --- Operators ---
  // Outside: standard Helmholtz
  BIOp<HE_SL_2D_P1xP1> Vout(mesh, mesh, kappa_out);
  BIOp<HE_DL_2D_P1xP1> Kout(mesh, mesh, kappa_out);

  // Inside: rotating Helmholtz (RH) with singular+remainder stabilization and radius scaling
  BIOp<RH_SL_2D_P1xP1> Vin(mesh, mesh, khat, Omega, c, Mtrunc, /*keep_D2=*/false);
  BIOp<RH_DL_2D_P1xP1> Kin(mesh, mesh, khat, Omega, c, Mtrunc, /*keep_D2=*/false);

  // Mass matrix for the identity term (Galerkin)
  EigenDense Mass = AssembleMassP1(mesh, dof, radius);

  // --- Assemble block system for transmission ---
  // Unknowns: phi = trace(u) on Gamma, q = (1/mu) * d_n u on Gamma.
  // Representation (using BemTool DL convention):
  // (  +1/2 I + K_out ) phi  - V_out( mu_out q ) = u_inc|_Gamma
  // (  -1/2 I + K_in  ) phi  - V_in ( mu_in  q ) = 0

  EigenDense A11(nb_dof, nb_dof); Clear(A11);
  EigenDense A12(nb_dof, nb_dof); Clear(A12);
  EigenDense A21(nb_dof, nb_dof); Clear(A21);
  EigenDense A22(nb_dof, nb_dof); Clear(A22);

  progress bar("Assemble (blocks)", nb_dof);
  for (int j = 0; j < nb_dof; ++j)
  {
    std::cout << j << std::endl;
    bar++;
    const auto& ej = dof.ToElt(j);
    for (int k = 0; k < nb_dof; ++k)
    {
      const auto& ek = dof.ToElt(k);

      // K and V contributions
      const Cplx kout_jk = Kout(ej, ek);
      const Cplx vout_jk = Vout(ej, ek);
      const Cplx kin_jk  = Kin(ej, ek);
      const Cplx vin_jk  = Vin(ej, ek);

      A11(j, k) += kout_jk;
      A12(j, k) += -mu_out * vout_jk;
      A21(j, k) += kin_jk;
      A22(j, k) += -mu_in * vin_jk;
    }
  }
  bar.end();

  // Add the ±1/2 I terms in Galerkin form
  // Add ±1/2 I in Galerkin form
  for (int j = 0; j < nb_dof; ++j)
    for (int k = 0; k < nb_dof; ++k)
    {
      A11(j, k) += (0.5)  * Mass(j, k);
      A21(j, k) += (-0.5) * Mass(j, k);
    }

  // Build full 2n x 2n matrix without .block()
  const int N = 2 * nb_dof;
  EigenDense A(N, N);
  Clear(A);

  for (int j = 0; j < nb_dof; ++j)
  {
    for (int k = 0; k < nb_dof; ++k)
    {
      A(j, k)                 = A11(j, k);
      A(j, k + nb_dof)        = A12(j, k);
      A(j + nb_dof, k)        = A21(j, k);
      A(j + nb_dof, k+nb_dof) = A22(j, k);
    }
  }


  // RHS
  EigenDense::VectorType b(2 * nb_dof);

  // u_inc trace sampled at dof nodes (physical coords = radius * x_unit)
  for (int e = 0; e < NbElt(mesh); ++e)
  {
    const N2& edof = dof[e];
    const array<2, R3> xdof = dof(e);
    for (int a = 0; a < 2; ++a)
    {
      const int I = edof[a];
      const R3 x_phys = radius * xdof[a];
      b[I] = u_inc_plane(x_phys, kappa_out);
    }
  }

  // Solve
  std::cout << "Solving...\n";
  EigenDense::VectorType s(2 * nb_dof);
  lu_solve(A, b, s);

  EigenDense::EigenVectorType phi(nb_dof);
  EigenDense::EigenVectorType q(nb_dof);
  for (unsigned i = 0; i < nb_dof; ++i)
  {
    phi(i) = s[i];
    q(i) = s[nb_dof + i];
  }

  std::cout << "||phi||_2 = " << phi.norm() << "\n";
  std::cout << "||q||_2   = " << q.norm() << "\n";

  // (Optional) evaluate scattered field at a point outside, e.g. x=(2,0)
  const R3 x_eval = R3_(2.0, 0.0, 0.0);
  Cplx u_sc = 0.0;
  for (int e = 0; e < NbElt(mesh); ++e)
  {
    // crude: midpoint collocation-like evaluation using element midpoint
    const auto& elt = mesh[e];
    const R3 y_unit = 0.5 * (elt[0] + elt[1]);
    const R3 y_phys = radius * y_unit;

    const Real R = norm2(x_eval - y_phys);
    const Cplx G = (iu / 4.0) * Hankel0(kappa_out * R);

    // Use element length as weight
    const Real ds = radius * norm2(elt[1] - elt[0]);

    // nearest dof indices (just to get something quick)
    const N2& edof = dof[e];
    const Cplx phi_e = 0.5 * (phi(edof[0]) + phi(edof[1]));
    const Cplx q_e   = 0.5 * (q(edof[0]) + q(edof[1]));

    // representation: u_sc = D phi - S (mu_out q)
    // Here we only include the S term as a quick demo; full D needs normal derivative.
    u_sc += -G * (mu_out * q_e) * ds;
  }

  std::cout << "u_sc(x=(2,0)) (rough) = " << u_sc << "\n";

  return 0;
}
