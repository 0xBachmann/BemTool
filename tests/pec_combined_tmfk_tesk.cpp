#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <fstream>
#include <string>
#include <algorithm>
#include <functional>

#include "../bemtool/tools.hpp"

using namespace bemtool;

namespace
{
    struct SolveResult
    {
        std::string name;
        Eigen::VectorXcd density;
        Eigen::MatrixXcd A;
        Eigen::VectorXcd rhs;
        bool uses_sl_potential = true;
        Real relative_bie_residual = 0.0;
        bool available = false;
    };

    struct FieldSample
    {
        Real theta = 0.0;
        R3 x;
        Cplx uinc = 0.0;
        Cplx usca = 0.0;
        Cplx utot = 0.0;
    };

    Cplx incident_plane_wave(const R3& x, Real kx, Real ky)
    {
        return std::exp(iu * (kx * x[0] + ky * x[1]));
    }

    Cplx dirichlet_trace_incident(const R3& x, Real kx, Real ky)
    {
        return incident_plane_wave(x, kx, ky);
    }

    Cplx neumann_trace_incident_circle(const R3& x, Real kx, Real ky)
    {
        const Real r = std::sqrt(x[0] * x[0] + x[1] * x[1]);
        const Real nx = x[0] / r;
        const Real ny = x[1] / r;
        const Cplx uinc = incident_plane_wave(x, kx, ky);
        return iu * (kx * nx + ky * ny) * uinc;
    }

    template <typename Space>
    Eigen::VectorXcd interpolate_to_dofs(const Mesh1D& mesh,
                                         const Dof<Space>& dof,
                                         const std::function<Cplx(const R3&)>& f)
    {
        Interpolator<Space> interp(mesh);
        Eigen::VectorXcd coeffs = Eigen::VectorXcd::Zero(NbDof(dof));

        const int nb_elt = NbElt(mesh);
        for (int e = 0; e < nb_elt; ++e)
        {
            const auto local_vals = interp(f, e);
            const auto& edof = dof[e];
            for (int k = 0; k < Space::nb_dof_loc; ++k)
            {
                const int I = edof[k];
                coeffs(I) += local_vals[k];
            }
        }

        return coeffs;
    }

    template <typename FUNC>
Eigen::VectorXcd L2WithP1Basis(FUNC g,
                               const Mesh1D& mesh,
                               const Dof<P1_1D>& dof_p1)
    {
        // 8-point Gauss rule on [0,1] via mapped points from [-1,1]
        // This is usually more than enough for smooth incident data.
        static const int order = 8;
        static const Real X[8] = {
            -0.9602898564975363, -0.7966664774136267,
            -0.5255324099163290, -0.1834346424956498,
             0.1834346424956498,  0.5255324099163290,
             0.7966664774136267,  0.9602898564975363
        };
        static const Real W[8] = {
            0.1012285362903763, 0.2223810344533745,
            0.3137066458778873, 0.3626837833783620,
            0.3626837833783620, 0.3137066458778873,
            0.2223810344533745, 0.1012285362903763
        };

        Eigen::VectorXcd out = Eigen::VectorXcd::Zero(NbDof(dof_p1));

        const int nb_elt = NbElt(mesh);
        for (int e = 0; e < nb_elt; ++e)
        {
            const Elt1D& elt = mesh[e];
            const R3 a = elt[0];
            const R3 b = elt[1];
            const R3 tau = b - a;
            const Real h = std::sqrt((tau, tau));

            const auto& edof = dof_p1[e];

            Cplx local0 = 0.0;
            Cplx local1 = 0.0;

            // reference interval [-1,1]
            // gamma(t) = 0.5*(a+b) + 0.5*t*(b-a)
            // |gamma'(t)| = h/2
            for (int q = 0; q < order; ++q)
            {
                const Real t = X[q];
                const Real w = W[q];

                const R3 x = 0.5 * (a + b) + 0.5 * t * tau;

                const Cplx gx = g(x);

                const Real phi0 = 0.5 * (1.0 - t);
                const Real phi1 = 0.5 * (1.0 + t);

                local0 += 0.5 * h * w * gx * phi0;
                local1 += 0.5 * h * w * gx * phi1;
            }

            out(edof[0]) += local0;
            out(edof[1]) += local1;
        }

        return out;
    }

    template <typename Space>
    Eigen::VectorXcd assemble_rhs_from_interpolation(const Mesh1D& mesh,
                                                     const Dof<Space>& dof,
                                                     const Eigen::MatrixXcd& M,
                                                     const std::function<Cplx(const R3&)>& f)
    {
        const Eigen::VectorXcd coeffs = interpolate_to_dofs(mesh, dof, f);
        return M * coeffs;
    }

    template <typename Op, typename DofType>
    Cplx eval_layer_field(Potential<Op>& pot,
                          const Mesh1D& mesh,
                          const DofType& dof,
                          const Eigen::VectorXcd& dens,
                          const R3& x)
    {
        const int nb_elt = NbElt(mesh);
        Cplx u = 0.0;
        for (int e = 0; e < nb_elt; ++e)
        {
            const auto& edof = dof[e];
            for (int k = 0; k < DofType::ShapeFct::nb_dof_loc; ++k)
            {
                u += pot(x, N2_(e, k)) * dens(edof[k]);
            }
        }
        return u;
    }

// -----------------------------------------------------------------------------
// Geometric helpers for 1D boundary elements in 2D
// -----------------------------------------------------------------------------
inline R3 element_midpoint(const Elt<1>& e) {
  return 0.5 * (e[0] + e[1]);
}

inline double element_length(const Elt<1>& e) {
  return norm2(e[1] - e[0]);
}

// Outward normal from oriented tangent.
// For a positively oriented boundary, this gives one of the two normals.
// If the sign comes out wrong in your tests, flip it once globally.
inline R3 element_unit_normal(const Elt<1>& e) {
  R3 t = e[1] - e[0];
  const double len = norm2(t);
  if (len == 0.0) {
    throw std::runtime_error("Degenerate boundary element with zero length.");
  }

  R3 n;
  n[0] =  t[1] / len;
  n[1] = -t[0] / len;
  n[2] =  0.0;
  return n;
}

// -----------------------------------------------------------------------------
// gamma_D : trace to continuous P1 space
//
// Returns one coefficient per global P1 dof.
// In the usual lowest-order setting on a boundary curve, those are nodal values.
// -----------------------------------------------------------------------------
template <typename DirichletFunctor>
Eigen::VectorXcd DirichletTraceToP1(const Mesh1D& mesh,
                                    const Dof<P1_1D>& dof,
                                    DirichletFunctor&& g) {
  const int nb_dof = NbDof(dof);
  Eigen::VectorXcd rhs = Eigen::VectorXcd::Zero(nb_dof);

  // We also count how many times each global dof is hit, in case the access
  // pattern below visits shared vertices multiple times.
  Eigen::VectorXi counts = Eigen::VectorXi::Zero(nb_dof);

  const int nb_elt = NbElt(mesh);
  for (int j = 0; j < nb_elt; ++j) {
    const Elt<1>& e = mesh[j];

    // Local P1 has two endpoint dofs.
    for (int a = 0; a < 2; ++a) {
      const int I = dof[j][a];
      rhs(I) += g(e[a]);
      counts(I) += 1;
    }
  }

  for (int i = 0; i < nb_dof; ++i) {
    if (counts(i) > 0) {
      rhs(i) /= static_cast<double>(counts(i));
    }
  }

  return rhs;
}

// -----------------------------------------------------------------------------
// gamma_D with optional L2 projection to P1
//
// This is the more variationally faithful version.
// It solves M * u_h = b with
//   M_ij = (phi_j, phi_i)
//   b_i  = (g, phi_i)
// Use this if you want the trace in the same sense as your L2 projection code.
// -----------------------------------------------------------------------------
template <typename DirichletFunctor, typename MassAssembler, typename LoadAssembler>
Eigen::VectorXcd DirichletTraceToP1_L2(const Mesh1D& mesh,
                                       MassAssembler&& assemble_mass_p1p1,
                                       LoadAssembler&& assemble_load_with_p1,
                                       DirichletFunctor&& g) {
  Eigen::MatrixXcd M = assemble_mass_p1p1(mesh);
  Eigen::VectorXcd b = assemble_load_with_p1(mesh, std::forward<DirichletFunctor>(g));
  return M.fullPivLu().solve(b);
}

// -----------------------------------------------------------------------------
// gamma_N : trace to discontinuous P0 space
//
// Returns one coefficient per boundary element.
// The coefficient is the elementwise average of the normal derivative.
// -----------------------------------------------------------------------------
template <typename GradFunctor>
Eigen::VectorXcd NeumannTraceToP0(const Mesh1D& mesh, GradFunctor&& grad_u) {
  const int nb_elt = NbElt(mesh);
  Eigen::VectorXcd rhs(nb_elt);

  for (int j = 0; j < nb_elt; ++j) {
    const Elt<1>& e = mesh[j];
    const R3 xm = element_midpoint(e);
    const R3 n  = element_unit_normal(e);

    const auto gu = grad_u(xm);   // expected to return gradient vector
    rhs(j) = gu[0] * n[0] + gu[1] * n[1] + gu[2] * n[2];
  }

  return rhs;
}

// -----------------------------------------------------------------------------
// gamma_N : slightly more accurate P0 projection by 2-point Gauss rule
//
// Computes the element average
//   (1/|T|) \int_T grad(u)·n ds
// -----------------------------------------------------------------------------
template <typename GradFunctor>
Eigen::VectorXcd NeumannTraceToP0_Average(const Mesh1D& mesh, GradFunctor&& grad_u) {
  const int nb_elt = NbElt(mesh);
  Eigen::VectorXcd rhs(nb_elt);

  // Gauss points on [0,1]
  static const double s1 = 0.5 * (1.0 - 1.0 / std::sqrt(3.0));
  static const double s2 = 0.5 * (1.0 + 1.0 / std::sqrt(3.0));
  static const double w1 = 0.5;
  static const double w2 = 0.5;

  for (int j = 0; j < nb_elt; ++j) {
    const Elt<1>& e = mesh[j];
    const R3 x0 = e[0];
    const R3 x1 = e[1];
    const R3 n  = element_unit_normal(e);

    const R3 xg1 = (1.0 - s1) * x0 + s1 * x1;
    const R3 xg2 = (1.0 - s2) * x0 + s2 * x1;

    const auto g1 = grad_u(xg1);
    const auto g2 = grad_u(xg2);

    const Cplx dn1 = g1[0] * n[0] + g1[1] * n[1] + g1[2] * n[2];
    const Cplx dn2 = g2[0] * n[0] + g2[1] * n[1] + g2[2] * n[2];

    // This is already the average on the reference segment [0,1].
    rhs(j) = w1 * dn1 + w2 * dn2;
  }

  return rhs;
}

// -----------------------------------------------------------------------------
// Example incident plane wave data
// u_inc(x) = exp(i k·x)
// grad u_inc(x) = i k exp(i k·x)
// -----------------------------------------------------------------------------
inline Cplx plane_wave(const R3& x, double kx, double ky) {
  const double phase = kx * x[0] + ky * x[1];
  return std::exp(Cplx(0.0, phase));
}

inline C3 plane_wave_grad(const R3& x, double kx, double ky) {
  const Cplx u = plane_wave(x, kx, ky);
  C3 g;
  g[0] = Cplx(0.0, kx) * u;
  g[1] = Cplx(0.0, ky) * u;
  g[2] = Cplx(0.0, 0.0);
  return g;
}

// -----------------------------------------------------------------------------
// Convenience wrappers for your PEC tests
// -----------------------------------------------------------------------------
inline Eigen::VectorXcd assemble_rhs_dirichlet_p1(const Mesh1D& mesh,
                                                  const Dof<P1_1D>& dof,
                                                  double kx,
                                                  double ky) {
  return DirichletTraceToP1(
      mesh, dof,
      [=](const R3& x) -> Cplx {
        return plane_wave(x, kx, ky);
      });
}

inline Eigen::VectorXcd assemble_rhs_neumann_p0(const Mesh1D& mesh,
                                                double kx,
                                                double ky) {
  return NeumannTraceToP0_Average(
      mesh,
      [=](const R3& x) -> C3 {
        return plane_wave_grad(x, kx, ky);
      });
}



    template <typename SLPot, typename DLPot, typename DofType>
    std::vector<FieldSample> sample_total_field_on_circle(const Mesh1D& mesh,
                                                          const DofType& dof,
                                                          Potential<SLPot>& sl_pot,
                                                          Potential<DLPot>& dl_pot,
                                                          const SolveResult& result,
                                                          Real radius,
                                                          int n_samples,
                                                          Real kx,
                                                          Real ky)
    {
        std::vector<FieldSample> samples;
        samples.reserve(n_samples);

        for (int m = 0; m < n_samples; ++m)
        {
            const Real theta = 2.0 * M_PI * static_cast<Real>(m) / static_cast<Real>(n_samples);
            R3 x;
            x[0] = radius * std::cos(theta);
            x[1] = radius * std::sin(theta);
            x[2] = 0.0;

            FieldSample s;
            s.theta = theta;
            s.x = x;
            s.uinc = incident_plane_wave(x, kx, ky);

            if (result.uses_sl_potential)
            {
                s.usca = -eval_layer_field(sl_pot, mesh, dof, result.density, x);
            }
            else
            {
                s.usca = eval_layer_field(dl_pot, mesh, dof, result.density, x);
            }

            s.utot = s.uinc + s.usca;
            samples.push_back(s);
        }

        return samples;
    }

    template <class OperatorType, class TestDofType, class TrialDofType>
    Eigen::MatrixXcd assemble_biop_matrix(const TestDofType& test_dof,
                                          const TrialDofType& trial_dof,
                                          OperatorType& op,
                                          const std::string& label)
    {
        const int nb_row = NbDof(test_dof);
        const int nb_col = NbDof(trial_dof);
        Eigen::MatrixXcd A = Eigen::MatrixXcd::Zero(nb_row, nb_col);

        progress bar(label.c_str(), nb_row);
        for (int i = 0; i < nb_row; ++i)
        {
            bar++;
            for (int j = 0; j < nb_col; ++j)
            {
                A(i, j) += op(test_dof.ToElt(i), trial_dof.ToElt(j));
            }
        }
        bar.end();
        return A;
    }

    template <class OperatorType, class DofType>
    Eigen::MatrixXcd assemble_square_biop_matrix(const DofType& dof,
                                                 OperatorType& op,
                                                 const std::string& label)
    {
        return assemble_biop_matrix(dof, dof, op, label);
    }

    void write_field_samples_csv(const std::string& filename,
                                 const std::vector<FieldSample>& samples)
    {
        std::ofstream fout(filename);
        fout << "theta,x,y,real_uinc,imag_uinc,real_usca,imag_usca,real_utot,imag_utot,abs_utot\n";
        for (const auto& s : samples)
        {
            fout << s.theta << ',' << s.x[0] << ',' << s.x[1] << ','
                 << std::real(s.uinc) << ',' << std::imag(s.uinc) << ','
                 << std::real(s.usca) << ',' << std::imag(s.usca) << ','
                 << std::real(s.utot) << ',' << std::imag(s.utot) << ','
                 << std::abs(s.utot) << '\n';
        }
    }

    void print_total_field_stats(const std::string& label,
                                 const std::vector<FieldSample>& samples,
                                 Real radius)
    {
        Real max_abs_utot = 0.0;
        Real sum_sq_abs_utot = 0.0;

        for (const auto& s : samples)
        {
            const Real a = std::abs(s.utot);
            max_abs_utot = std::max(max_abs_utot, a);
            sum_sq_abs_utot += a * a;
        }

        const Real rms_abs_utot = std::sqrt(sum_sq_abs_utot / static_cast<Real>(samples.size()));
        std::cout << label << " on observation circle r = " << radius << "\n";
        std::cout << "max |u_tot| = " << max_abs_utot << "\n";
        std::cout << "rms |u_tot| = " << rms_abs_utot << "\n";
    }

    void compare_scattered_fields(const std::string& label,
                                  const std::vector<FieldSample>& a,
                                  const std::vector<FieldSample>& b,
                                  const std::string& filename)
    {
        std::ofstream fout(filename);
        fout << "theta,real_usca_a,imag_usca_a,real_usca_b,imag_usca_b,abs_diff\n";

        Real max_abs_diff = 0.0;
        Real sum_sq_abs_diff = 0.0;
        Real sum_sq_abs_a = 0.0;

        const int n = static_cast<int>(std::min(a.size(), b.size()));
        for (int i = 0; i < n; ++i)
        {
            const Cplx diff = a[i].usca - b[i].usca;
            const Real ad = std::abs(diff);
            const Real aa = std::abs(a[i].usca);

            max_abs_diff = std::max(max_abs_diff, ad);
            sum_sq_abs_diff += ad * ad;
            sum_sq_abs_a += aa * aa;

            fout << a[i].theta << ','
                 << std::real(a[i].usca) << ',' << std::imag(a[i].usca) << ','
                 << std::real(b[i].usca) << ',' << std::imag(b[i].usca) << ','
                 << ad << '\n';
        }

        const Real rms_abs_diff = std::sqrt(sum_sq_abs_diff / static_cast<Real>(n));
        const Real rel_rms_diff = std::sqrt(sum_sq_abs_diff / std::max(sum_sq_abs_a, Real(1.0e-30)));

        std::cout << label << "\n";
        std::cout << "max |u_sca^A - u_sca^B| = " << max_abs_diff << "\n";
        std::cout << "rms |u_sca^A - u_sca^B| = " << rms_abs_diff << "\n";
        std::cout << "relative rms difference = " << rel_rms_diff << "\n";
    }
} // namespace

int main(int argc, char* argv[])
{
    constexpr bool do_tm_fk = true;
    constexpr bool do_tm_sk = true;
    constexpr bool do_te_fk = false;
    constexpr bool do_te_sk = false;

    constexpr Real k0 = 2.0;
    constexpr Real eps = 1.0;
    constexpr Real mu = 1.0;
    constexpr Real Omega = 0.0; // stationary PEC benchmark

    constexpr Real theta_inc = 0.0;
    constexpr Real r_obs = 2.0;
    constexpr int n_obs = 720;

    const Real kx = k0 * std::cos(theta_inc);
    const Real ky = k0 * std::sin(theta_inc);

    Geometry node("../mesh/circle.msh");
    Mesh1D mesh;
    mesh.Load(node, 1);
    Orienting(mesh);

    Dof<P0_1D> dof_p0(mesh);
    Dof<P1_1D> dof_p1(mesh);
    const int nb_elt = NbElt(mesh);

    std::cout << "solving ";
    if (do_te_fk) std::cout << "te_fk, ";
    if (do_te_sk) std::cout << "te_sk, ";
    if (do_tm_fk) std::cout << "tm_fk, ";
    if (do_tm_sk) std::cout << "tm_sk, ";
    std::cout << "\n";

    std::cout << "nb_elt = " << nb_elt << "\n";
    std::cout << "nb_dof(P0) = " << NbDof(dof_p0) << "\n";
    std::cout << "nb_dof(P1) = " << NbDof(dof_p1) << "\n";

    BIOp<CST_1D_P0xP0> M_00_op(mesh, mesh);
    BIOp<CST_1D_P1xP0> M_10_op(mesh, mesh);
    BIOp<CST_1D_P0xP1> M_01_op(mesh, mesh);
    BIOp<CST_1D_P1xP1> M_11_op(mesh, mesh);
    const Eigen::MatrixXcd M_00 = assemble_square_biop_matrix(dof_p0, M_00_op, "Assembling M(P0xP0)");
    const Eigen::MatrixXcd M_10 = assemble_biop_matrix(dof_p1, dof_p0, M_10_op, "Assembling M(P1xP0)");
    const Eigen::MatrixXcd M_01 = assemble_biop_matrix(dof_p0, dof_p1, M_01_op, "Assembling M(P0xP1)");
    const Eigen::MatrixXcd M_11 = assemble_square_biop_matrix(dof_p1, M_11_op, "Assembling M(P1xP1)");

    Potential<RH_SL_2D_P0> sl_pot_p0(mesh, k0, eps, mu, Omega);
    Potential<RH_SL_2D_P1> sl_pot_p1(mesh, k0, eps, mu, Omega);
    Potential<RH_DL_2D_P1> dl_pot_p1(mesh, k0, eps, mu, Omega);

    auto g_dir = [&](const R3& x) -> Cplx {
        return dirichlet_trace_incident(x, kx, ky);
    };

    auto g_neu = [&](const R3& x) -> Cplx {
        return neumann_trace_incident_circle(x, kx, ky);
    };

    SolveResult tm_fk, tm_sk, te_sk, te_fk;

    if (do_tm_fk)
    {
        tm_fk.name = "tm_fk";
        tm_fk.uses_sl_potential = true;
        tm_fk.available = true;

        BIOp<RH_SL_2D_P1xP0> V(mesh, mesh, k0, eps, mu, Omega);
        tm_fk.A = assemble_biop_matrix(dof_p1, dof_p0, V, "Assembling V(P1xP0)");
        tm_fk.rhs = assemble_rhs_from_interpolation(mesh, dof_p0, M_10, g_dir);
        // tm_fk.rhs = M_11 * assemble_rhs_dirichlet_p1(mesh, dof_p1, kx, ky);

        tm_fk.density = tm_fk.A.fullPivLu().solve(tm_fk.rhs);
        const Eigen::VectorXcd res = tm_fk.A * tm_fk.density - tm_fk.rhs;
        tm_fk.relative_bie_residual = res.norm() / std::max(tm_fk.rhs.norm(), Real(1.0e-30));

        std::cout << "relative BIE residual (tm_fk) = " << tm_fk.relative_bie_residual << "\n";
    }

    if (do_tm_sk)
    {
        tm_sk.name = "tm_sk";
        tm_sk.uses_sl_potential = false;
        tm_sk.available = true;

        BIOp<RH_DL_2D_P1xP1> K(mesh, mesh, k0, eps, mu, Omega);
        const Eigen::MatrixXcd Kmat = assemble_square_biop_matrix(dof_p1, K, "Assembling K(P1xP1)");
        tm_sk.A = 0.5 * M_11 + Kmat;
        tm_sk.rhs = -assemble_rhs_from_interpolation(mesh, dof_p1, M_11, g_dir);
        // tm_fk.rhs = -M_11*assemble_rhs_dirichlet_p1(mesh, dof_p1, kx, ky);


        tm_sk.density = tm_sk.A.fullPivLu().solve(tm_sk.rhs);
        const Eigen::VectorXcd res = tm_sk.A * tm_sk.density - tm_sk.rhs;
        tm_sk.relative_bie_residual = res.norm() / std::max(tm_sk.rhs.norm(), Real(1.0e-30));

        std::cout << "relative BIE residual (tm_sk) = " << tm_sk.relative_bie_residual << "\n";
    }

    if (do_te_sk)
    {
        te_sk.name = "te_sk";
        te_sk.uses_sl_potential = true;
        te_sk.available = true;

        BIOp<RH_TDL_2D_P0xP0> Kp(mesh, mesh, k0, eps, mu, Omega);
        const Eigen::MatrixXcd Kpmat = assemble_square_biop_matrix(dof_p0, Kp, "Assembling Kp(P0xP0)");
        te_sk.A = -0.5 * M_00 + Kpmat;
        te_sk.rhs = assemble_rhs_from_interpolation(mesh, dof_p0, M_00, g_neu);

        te_sk.density = te_sk.A.fullPivLu().solve(te_sk.rhs);
        const Eigen::VectorXcd res = te_sk.A * te_sk.density - te_sk.rhs;
        te_sk.relative_bie_residual = res.norm() / std::max(te_sk.rhs.norm(), Real(1.0e-30));

        std::cout << "relative BIE residual (te_sk) = " << te_sk.relative_bie_residual << "\n";
    }

    if (do_te_fk)
    {
        te_fk.name = "te_fk";
        te_fk.uses_sl_potential = false;
        te_fk.available = true;

        BIOp<RH_HS_2D_P0xP1> W(mesh, mesh, k0, eps, mu, Omega);
        te_fk.A = assemble_biop_matrix(dof_p0, dof_p1, W, "Assembling W(P0xP1)");
        te_fk.rhs = assemble_rhs_from_interpolation(mesh, dof_p0, M_00, g_neu);

        te_fk.density = te_fk.A.fullPivLu().solve(te_fk.rhs);
        const Eigen::VectorXcd res = te_fk.A * te_fk.density - te_fk.rhs;
        te_fk.relative_bie_residual = res.norm() / std::max(te_fk.rhs.norm(), Real(1.0e-30));

        std::cout << "relative BIE residual (te_fk) = " << te_fk.relative_bie_residual << "\n";
    }

    if (tm_fk.available)
    {
        const auto samples = sample_total_field_on_circle(mesh, dof_p0, sl_pot_p0, dl_pot_p1, tm_fk, r_obs, n_obs, kx, ky);
        write_field_samples_csv("pec_tm_fk_obs_r2.csv", samples);
        print_total_field_stats("TM first-kind total-field check", samples, r_obs);
        std::cout << "Wrote observations to pec_tm_fk_obs_r2.csv\n";
    }

    if (tm_sk.available)
    {
        const auto samples = sample_total_field_on_circle(mesh, dof_p1, sl_pot_p1, dl_pot_p1, tm_sk, r_obs, n_obs, kx, ky);
        write_field_samples_csv("pec_tm_sk_obs_r2.csv", samples);
        print_total_field_stats("TM second-kind total-field check", samples, r_obs);
        std::cout << "Wrote observations to pec_tm_sk_obs_r2.csv\n";
    }

    if (te_sk.available)
    {
        const auto samples = sample_total_field_on_circle(mesh, dof_p0, sl_pot_p0, dl_pot_p1, te_sk, r_obs, n_obs, kx, ky);
        write_field_samples_csv("pec_te_sk_obs_r2.csv", samples);
        print_total_field_stats("TE second-kind total-field observations", samples, r_obs);
        std::cout << "Wrote observations to pec_te_sk_obs_r2.csv\n";
    }

    if (te_fk.available)
    {
        const auto samples = sample_total_field_on_circle(mesh, dof_p1, sl_pot_p1, dl_pot_p1, te_fk, r_obs, n_obs, kx, ky);
        write_field_samples_csv("pec_te_fk_obs_r2.csv", samples);
        print_total_field_stats("TE first-kind total-field observations", samples, r_obs);
        std::cout << "Wrote observations to pec_te_fk_obs_r2.csv\n";
    }

    if (tm_fk.available && tm_sk.available)
    {
        const auto a = sample_total_field_on_circle(mesh, dof_p0, sl_pot_p0, dl_pot_p1, tm_fk, r_obs, n_obs, kx, ky);
        const auto b = sample_total_field_on_circle(mesh, dof_p1, sl_pot_p1, dl_pot_p1, tm_sk, r_obs, n_obs, kx, ky);
        compare_scattered_fields("TM formulation agreement on observation circle",
                                 a, b, "pec_tm_formulation_comparison_r2.csv");
        std::cout << "Wrote comparison to pec_tm_formulation_comparison_r2.csv\n";
    }

    if (te_sk.available && te_fk.available)
    {
        const auto a = sample_total_field_on_circle(mesh, dof_p0, sl_pot_p0, dl_pot_p1, te_sk, r_obs, n_obs, kx, ky);
        const auto b = sample_total_field_on_circle(mesh, dof_p1, sl_pot_p1, dl_pot_p1, te_fk, r_obs, n_obs, kx, ky);
        compare_scattered_fields("TE formulation agreement on observation circle",
                                 a, b, "pec_te_formulation_comparison_r2.csv");
        std::cout << "Wrote comparison to pec_te_formulation_comparison_r2.csv\n";
    }

    return 0;
}
