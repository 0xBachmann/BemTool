#include <iostream>
#include <vector>
#include <cmath>

#include "../bemtool/tools.hpp"


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

// Incident plane wave u_inc(x) = exp(i k_hat_out d·x), d = (1,0)
static inline Cplx u_inc_plane(const R3& x_phys, const Real khat_out)
{
  return std::exp(iu * khat_out * x_phys[0]);
}

template <class Pot, class DofType>
Cplx eval_potential_at_point(Pot& pot, const DofType& dof, const std::vector<Cplx>& sigma, const R3& x)
{
  Cplx u = 0.;
  const int nd = NbDof(dof);
  for (int j = 0; j < nd; ++j)
  {
    u += sigma[j] * pot(x, dof.ToElt(j)); // key: sums elem contributions of global dof j
  }
  return u;
}

inline Cplx plane_wave_inc(const R3& x, Real khat, Real dx, Real dy)
{
  return std::exp(iu * khat * (dx * x[0] + dy * x[1]));
}

int main(int argc, char* argv[])
{
  // --- user parameters (easy to swap to CLI later) ---
  const Real radius = (argc > 1) ? std::atof(argv[1]) : 0.7; // cylinder radius (<1 if mesh is unit circle)
  const Real k0 = (argc > 2) ? std::atof(argv[2]) : 10.0; // vacuum wavenumber (k0 = omega/c)
  const Real eps_in = (argc > 3) ? std::atof(argv[3]) : 2.25; // relative permittivity inside
  const Real mu_in = (argc > 4) ? std::atof(argv[4]) : 1.0; // relative permeability inside
  const Real Omega = (argc > 5) ? std::atof(argv[5]) : 0.; // rotation rate (same units as your kernel)

  // polarization flag: 1 = TMz (u=Ez, flux scaling 1/mu), 0 = TEz (u=Hz, flux scaling 1/eps)
  const int polTM = (argc > 6) ? std::atoi(argv[6]) : 1;
  const bool TMz = (polTM != 0);

  // outside medium (vacuum-like)
  const Real eps_out = 1.0;
  const Real mu_out = 1.0;

  const Real khat_out = k0 * std::sqrt(eps_out * mu_out);
  const Real khat_in = k0 * std::sqrt(eps_in * mu_in);

  // Interface scaling: (1/alpha) dn u continuous
  // TMz: alpha = mu, TEz: alpha = eps
  const Real alpha_out = TMz ? mu_out : eps_out;
  const Real alpha_in = TMz ? mu_in : eps_in;

  Geometry node("mesh/circle.msh");
  Mesh1D mesh;
  mesh.Load(node, 1);
  Orienting(mesh);

  Dof<P1_1D> dof(mesh);
  const int nb_dof = NbDof(dof);

  std::cout << "nb_dof:\t" << nb_dof << "\n";
  std::cout << "radius:\t" << radius << "\n";
  std::cout << "k0:\t" << k0 << "\n";
  std::cout << "khat_out:\t" << khat_out << "\n";
  std::cout << "khat_in:\t" << khat_in << "\n";
  std::cout << "Omega:\t" << Omega << "\n";
  std::cout << "polarization:\t" << (TMz ? "TMz (u=Ez, alpha=mu)" : "TEz (u=Hz, alpha=eps)") << "\n";

  // --- Operators ---
  // Use the rotating fast Green's function in BOTH regions (same Omega, different eps/mu).
  // Constructor is: BIOp<RH_*_2D_P1xP1>(mesh, mesh, k0, eps, mu, Omega)
  BIOp<RH_SL_2D_P1xP1> Vout(mesh, mesh, k0, eps_out, mu_out, Omega);
  BIOp<RH_DL_2D_P1xP1> Kout(mesh, mesh, k0, eps_out, mu_out, Omega);

  BIOp<RH_SL_2D_P1xP1> Vin(mesh, mesh, k0, eps_in, mu_in, Omega);
  BIOp<RH_DL_2D_P1xP1> Kin(mesh, mesh, k0, eps_in, mu_in, Omega);

  // Mass matrix for the identity term (Galerkin)
  EigenDense Mass = AssembleMassP1(mesh, dof, radius);

  // --- Assemble block system for transmission ---
  // Unknowns:
  //   phi = trace(u) on Gamma
  //   q   = (1/alpha) * d_n u on Gamma   (so alpha*q = d_n u)
  // Using direct representation limits:
  //   ( +1/2 I + K_out ) phi  - V_out( alpha_out q ) = u_inc|_Gamma
  //   ( -1/2 I + K_in  ) phi  - V_in ( alpha_in  q ) = 0

  EigenDense A11(nb_dof, nb_dof);
  Clear(A11);
  EigenDense A12(nb_dof, nb_dof);
  Clear(A12);
  EigenDense A21(nb_dof, nb_dof);
  Clear(A21);
  EigenDense A22(nb_dof, nb_dof);
  Clear(A22);

  progress bar("Assemble (blocks)", nb_dof);
  for (int j = 0; j < nb_dof; ++j)
  {
    bar++;
    const auto& ej = dof.ToElt(j);
    for (int k = 0; k < nb_dof; ++k)
    {
      const auto& ek = dof.ToElt(k);

      const Cplx kout_jk = Kout(ej, ek);
      const Cplx vout_jk = Vout(ej, ek);
      const Cplx kin_jk = Kin(ej, ek);
      const Cplx vin_jk = Vin(ej, ek);

      A11(j, k) += kout_jk;
      A12(j, k) += -alpha_out * vout_jk;

      A21(j, k) += kin_jk;
      A22(j, k) += -alpha_in * vin_jk;
    }
  }
  bar.end();

  // Add the \pm 1/2 I terms in Galerkin form
  for (int j = 0; j < nb_dof; ++j)
    for (int k = 0; k < nb_dof; ++k)
    {
      A11(j, k) += (0.5) * Mass(j, k);
      A21(j, k) += (-0.5) * Mass(j, k);
    }

  // Build full 2n x 2n matrix
  const int N = 2 * nb_dof;
  EigenDense A(N, N);
  Clear(A);

  for (int j = 0; j < nb_dof; ++j)
  {
    for (int k = 0; k < nb_dof; ++k)
    {
      A(j, k) = A11(j, k);
      A(j, k + nb_dof) = A12(j, k);
      A(j + nb_dof, k) = A21(j, k);
      A(j + nb_dof, k + nb_dof) = A22(j, k);
    }
  }

  // RHS
  EigenDense::VectorType b(2 * nb_dof);
  for (int i = 0; i < 2 * nb_dof; ++i) b[i] = Cplx(0., 0.);

  // u_inc trace sampled at dof nodes (physical coords = radius * x_unit)
  for (int e = 0; e < NbElt(mesh); ++e)
  {
    const N2& edof = dof[e];
    const array<2, R3> xdof = dof(e);
    for (int a = 0; a < 2; ++a)
    {
      const int I = edof[a];
      const R3 x_phys = radius * xdof[a];
      b[I] = u_inc_plane(x_phys, khat_out);
    }
  }

  // Solve
  std::cout << "Solving...\n";
  EigenDense::VectorType s(2 * nb_dof);
  lu_solve(A, b, s);

  EigenDense::EigenVectorType phi(nb_dof);
  EigenDense::EigenVectorType q(nb_dof);
  for (unsigned i = 0; i < (unsigned)nb_dof; ++i)
  {
    phi(i) = s[i];
    q(i) = s[nb_dof + i];
  }

  std::cout << "||phi||_2 = " << phi.norm() << "\n";
  std::cout << "||q||_2   = " << q.norm() << "\n";


  // after you solved: phi (nb_dof), q (nb_dof)
  std::vector<Real> phi_re(nb_dof), phi_im(nb_dof), phi_abs(nb_dof);
  std::vector<Real> q_re(nb_dof), q_im(nb_dof), q_abs(nb_dof);
  std::vector<Real> aq_in_abs(nb_dof), aq_out_abs(nb_dof);

  for (int i = 0; i < nb_dof; ++i)
  {
    phi_re[i] = std::real(phi(i));
    phi_im[i] = std::imag(phi(i));
    phi_abs[i] = std::abs(phi(i));

    q_re[i] = std::real(q(i));
    q_im[i] = std::imag(q(i));
    q_abs[i] = std::abs(q(i));

    aq_out_abs[i] = std::abs(alpha_out * q(i));
    aq_out_abs[i] = std::abs(alpha_in * q(i));

  }

  WriteMeshParaview(dof, "bnd.geo");

  WritePointValParaview(dof, "phi_abs.scl", phi_abs);
  WriteCaseParaview("phi_abs.case", "bnd.geo", "phi_abs", "phi_abs.scl");

  WritePointValParaview(dof, "phi_re.scl", phi_re);
  WriteCaseParaview("phi_re.case", "bnd.geo", "phi_re", "phi_re.scl");

  WritePointValParaview(dof, "phi_im.scl", phi_im);
  WriteCaseParaview("phi_im.case", "bnd.geo", "phi_im", "phi_im.scl");

  // normal derivative on each side:
  WritePointValParaview(dof, "dn_out_abs.scl", aq_out_abs);
  WriteCaseParaview("dn_out_abs.case", "bnd.geo", "dn_out_abs", "dn_out_abs.scl");

  WritePointValParaview(dof, "dn_in_abs.scl", aq_in_abs);
  WriteCaseParaview("dn_in_abs.case", "bnd.geo", "dn_in_abs", "dn_in_abs.scl");

  Geometry node_vis("mesh/disc.msh");
  Mesh2D mesh_vis;
  mesh_vis.Load(node_vis, /*phys=*/1); // or whatever you use
  Orienting(mesh_vis);

  const int nn = NbNode(node_vis);

  // --- potentials built from your FAST rotating kernel ---
  Potential<RH_SL_2D_P1> SLout(mesh, k0, eps_out, mu_out, Omega);
  Potential<RH_DL_2D_P1> DLout(mesh, k0, eps_out, mu_out, Omega);

  // densities for evaluation
  std::vector<Cplx> sigma_sl(nb_dof), sigma_dl(nb_dof);
  for (int j = 0; j < nb_dof; ++j)
  {
    sigma_dl[j] = phi[j];
    sigma_sl[j] = alpha_out * q[j]; // because q = (1/alpha)*dn u
  }

  std::vector<Real> u_abs(nn), u_re(nn), u_im(nn);

  // plane wave direction
  const Real dx = 1.0, dy = 0.0;

  progress bar2("Evaluate", nn);
  for (int i = 0; i < nn; ++i)
  {
    bar2++;
    const R3 x = node_vis[i];

    if (std::abs(norm2(x) - 1.) < 1e-4){
          continue;
    }

    const Cplx u_sc =
      eval_potential_at_point(DLout, dof, sigma_dl, x)
      - eval_potential_at_point(SLout, dof, sigma_sl, x);

    const Cplx u_tot = plane_wave_inc(x, khat_out, dx, dy) + u_sc;

    u_abs[i] = std::abs(u_tot);
    u_re[i] = std::real(u_tot);
    u_im[i] = std::imag(u_tot);
  }
  bar2.end();

  // write to ParaView/Ensight using your helper
  WriteMeshParaview(mesh_vis, "vis.geo");
  WritePointValParaview(mesh_vis, "u_abs.scl", u_abs);
  WriteCaseParaview("u_abs.case", "vis.geo", "u_abs", "u_abs.scl");

  WritePointValParaview(mesh_vis, "u_re.scl", u_re);
  WriteCaseParaview("u_re.case", "vis.geo", "u_re", "u_re.scl");

  WritePointValParaview(mesh_vis, "u_im.scl", u_im);
  WriteCaseParaview("u_im.case", "vis.geo", "u_im", "u_im.scl");

  return 0;
}
