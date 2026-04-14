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

        // Direct traces:
        // beta  = Dirichlet trace, discretized in P1
        // gamma = Neumann   trace, discretized in P0
        Eigen::VectorXcd beta_p1;
        Eigen::VectorXcd gamma_p0;

        Eigen::MatrixXcd A;
        Eigen::VectorXcd rhs;
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
                coeffs(edof[k]) = local_vals[k];
            }
        }
        return coeffs;
    }

    template <typename FUNC>
    Eigen::VectorXcd L2WithP1Basis(FUNC g,
                                   const Mesh1D& mesh,
                                   const Dof<P1_1D>& dof_p1)
    {
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

    template <typename FUNC>
    Eigen::VectorXcd L2ProjectionToP1(FUNC g, const Mesh1D& mesh,
                                   const Dof<P1_1D>& dof_p1, const Eigen::MatrixXcd& M11) {
        // L2 projection of Dirichlet data g
        // Idea: projection g_h can be written in terms of P1 basis functions and satisfies
        // the orthogonality relation: (g - g_h, b_i) = 0 for all i.
        // Collecting coefficients of g_h^j in terms of the basis b_j gives the linear system
        // \sum_{j=1..N} g_h^j (b_j,b_i) = (g, b_i)
        // (b_j, b_i) is the matrix M11

        // Computing (g, b_i) for all basis functions b_i in the space P1
        // Can be done by simply using Gauss quadrature
        const Eigen::VectorXcd g_b = L2WithP1Basis(g, mesh, dof_p1);

        // Projecting the dirichlet data by solving the linear system
        // Computing the coefficients g_h^j by solving the linear system
        const Eigen::VectorXd g_projected = M11.fullPivLu().solve(g_b);

        return g_projected;
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

    Cplx eval_tm_scattered_field(Potential<RH_SL_2D_P0>& sl_pot_p0,
                                 Potential<RH_DL_2D_P1>& dl_pot_p1,
                                 const Mesh1D& mesh,
                                 const Dof<P0_1D>& dof_p0,
                                 const Dof<P1_1D>& dof_p1,
                                 const SolveResult& result,
                                 const Eigen::VectorXcd& beta_inc_p1,
                                 const R3& x)
    {
        // Section 3.6, eq. (3.89) / (3.90):
        //   u_sca = -Psi_SL f_N + Psi_DL g_D
        // with g_D = -beta_inc.
        const Cplx sl = eval_layer_field(sl_pot_p0, mesh, dof_p0, result.gamma_p0, x);
        const Cplx dl = eval_layer_field(dl_pot_p1, mesh, dof_p1, beta_inc_p1, x);
        return -sl - dl;
    }

    Cplx eval_te_scattered_field(Potential<RH_SL_2D_P0>& sl_pot_p0,
                                 Potential<RH_DL_2D_P1>& dl_pot_p1,
                                 const Mesh1D& mesh,
                                 const Dof<P0_1D>& dof_p0,
                                 const Dof<P1_1D>& dof_p1,
                                 const SolveResult& result,
                                 const Eigen::VectorXcd& gamma_inc_p0,
                                 const R3& x)
    {
        // Section 3.6, eq. (3.111) / (3.112):
        //   u_sca = -Psi_SL g_N + Psi_DL f_D
        // with g_N = -gamma_inc.
        const Cplx sl = eval_layer_field(sl_pot_p0, mesh, dof_p0, gamma_inc_p0, x);
        const Cplx dl = eval_layer_field(dl_pot_p1, mesh, dof_p1, result.beta_p1, x);
        return sl + dl;
    }

    template <typename EvalScattered>
    std::vector<FieldSample> sample_total_field_on_circle(const EvalScattered& eval_scattered,
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
            s.usca = eval_scattered(x);
            s.utot = s.uinc + s.usca;
            samples.push_back(s);
        }

        return samples;
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
    constexpr bool do_te_fk = true;
    constexpr bool do_te_sk = true;

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

    Dof<P0_1D> dof_p0(mesh); // Neumann trace gamma
    Dof<P1_1D> dof_p1(mesh); // Dirichlet trace beta
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

    auto beta_inc_fun = [&](const R3& x) -> Cplx {
        return dirichlet_trace_incident(x, kx, ky);
    };

    auto gamma_inc_fun = [&](const R3& x) -> Cplx {
        return neumann_trace_incident_circle(x, kx, ky);
    };

    const Eigen::VectorXcd beta_inc_p1 = L2WithP1Basis(beta_inc_fun, mesh, dof_p1);
    const Eigen::VectorXcd gamma_inc_p0 = assemble_rhs_from_interpolation(mesh, dof_p0, M_00, gamma_inc_fun);

    Potential<RH_SL_2D_P0> sl_pot_p0(mesh, k0, eps, mu, Omega);
    Potential<RH_DL_2D_P1> dl_pot_p1(mesh, k0, eps, mu, Omega);

    BIOp<RH_SL_2D_P1xP0> V(mesh, mesh, k0, eps, mu, Omega);      // P0 -> P1
    BIOp<RH_DL_2D_P1xP1> K(mesh, mesh, k0, eps, mu, Omega);      // P1 -> P1
    BIOp<RH_TDL_2D_P0xP0> Kp(mesh, mesh, k0, eps, mu, Omega);    // P0 -> P0
    BIOp<RH_HS_2D_P0xP1> W(mesh, mesh, k0, eps, mu, Omega);      // P1 -> P0

    const Eigen::MatrixXcd V_mat  = assemble_biop_matrix(dof_p1, dof_p0, V,  "Assembling V(P1xP0)");
    const Eigen::MatrixXcd K_mat  = assemble_biop_matrix(dof_p1, dof_p1, K,  "Assembling K(P1xP1)");
    const Eigen::MatrixXcd Kp_mat = assemble_biop_matrix(dof_p0, dof_p0, Kp, "Assembling Kp(P0xP0)");
    const Eigen::MatrixXcd W_mat  = assemble_biop_matrix(dof_p0, dof_p1, W,  "Assembling W(P0xP1)");

    SolveResult tm_fk, tm_sk, te_fk, te_sk;

    if (do_tm_fk)
    {
        // Eq. (3.86): V f_N = (1/2 I - K) beta_inc
        tm_fk.name = "tm_fk";
        tm_fk.available = true;
        tm_fk.A = V_mat;
        tm_fk.rhs = (0.5 * M_11 - K_mat) * beta_inc_p1;
        tm_fk.gamma_p0 = tm_fk.A.fullPivLu().solve(tm_fk.rhs);

        const Eigen::VectorXcd res = tm_fk.A * tm_fk.gamma_p0 - tm_fk.rhs;
        tm_fk.relative_bie_residual = res.norm() / std::max(tm_fk.rhs.norm(), Real(1.0e-30));
        std::cout << "relative BIE residual (tm_fk) = " << tm_fk.relative_bie_residual << "\n";
    }

    if (do_tm_sk)
    {
        // Eq. (3.88): (1/2 I + K') f_N = W beta_inc
        tm_sk.name = "tm_sk";
        tm_sk.available = true;
        tm_sk.A = 0.5 * M_00 + Kp_mat;
        tm_sk.rhs = W_mat * beta_inc_p1;
        tm_sk.gamma_p0 = tm_sk.A.fullPivLu().solve(tm_sk.rhs);

        const Eigen::VectorXcd res = tm_sk.A * tm_sk.gamma_p0 - tm_sk.rhs;
        tm_sk.relative_bie_residual = res.norm() / std::max(tm_sk.rhs.norm(), Real(1.0e-30));
        std::cout << "relative BIE residual (tm_sk) = " << tm_sk.relative_bie_residual << "\n";
    }

    if (do_te_fk)
    {
        // Eq. (3.108): W f_D = (1/2 I + K') gamma_inc
        te_fk.name = "te_fk";
        te_fk.available = true;
        te_fk.A = W_mat;
        te_fk.rhs = (0.5 * M_00 + Kp_mat) * gamma_inc_p0;
        te_fk.beta_p1 = te_fk.A.fullPivLu().solve(te_fk.rhs);

        const Eigen::VectorXcd res = te_fk.A * te_fk.beta_p1 - te_fk.rhs;
        te_fk.relative_bie_residual = res.norm() / std::max(te_fk.rhs.norm(), Real(1.0e-30));
        std::cout << "relative BIE residual (te_fk) = " << te_fk.relative_bie_residual << "\n";
    }

    if (do_te_sk)
    {
        // Eq. (3.110): (1/2 I - K) f_D = V gamma_inc
        te_sk.name = "te_sk";
        te_sk.available = true;
        te_sk.A = 0.5 * M_11 - K_mat;
        te_sk.rhs = V_mat * gamma_inc_p0;
        te_sk.beta_p1 = te_sk.A.fullPivLu().solve(te_sk.rhs);

        const Eigen::VectorXcd res = te_sk.A * te_sk.beta_p1 - te_sk.rhs;
        te_sk.relative_bie_residual = res.norm() / std::max(te_sk.rhs.norm(), Real(1.0e-30));
        std::cout << "relative BIE residual (te_sk) = " << te_sk.relative_bie_residual << "\n";
    }

    if (tm_fk.available)
    {
        const auto samples = sample_total_field_on_circle(
            [&](const R3& x) {
                return eval_tm_scattered_field(sl_pot_p0, dl_pot_p1,
                                               mesh, dof_p0, dof_p1,
                                               tm_fk, beta_inc_p1, x);
            },
            r_obs, n_obs, kx, ky);

        write_field_samples_csv("pec_tm_fk_obs_r2.csv", samples);
        print_total_field_stats("TM direct first-kind total-field check", samples, r_obs);
        std::cout << "Wrote observations to pec_tm_fk_obs_r2.csv\n";
    }

    if (tm_sk.available)
    {
        const auto samples = sample_total_field_on_circle(
            [&](const R3& x) {
                return eval_tm_scattered_field(sl_pot_p0, dl_pot_p1,
                                               mesh, dof_p0, dof_p1,
                                               tm_sk, beta_inc_p1, x);
            },
            r_obs, n_obs, kx, ky);

        write_field_samples_csv("pec_tm_sk_obs_r2.csv", samples);
        print_total_field_stats("TM direct second-kind total-field check", samples, r_obs);
        std::cout << "Wrote observations to pec_tm_sk_obs_r2.csv\n";
    }

    if (te_fk.available)
    {
        const auto samples = sample_total_field_on_circle(
            [&](const R3& x) {
                return eval_te_scattered_field(sl_pot_p0, dl_pot_p1,
                                               mesh, dof_p0, dof_p1,
                                               te_fk, gamma_inc_p0, x);
            },
            r_obs, n_obs, kx, ky);

        write_field_samples_csv("pec_te_fk_obs_r2.csv", samples);
        print_total_field_stats("TE direct first-kind total-field observations", samples, r_obs);
        std::cout << "Wrote observations to pec_te_fk_obs_r2.csv\n";
    }

    if (te_sk.available)
    {
        const auto samples = sample_total_field_on_circle(
            [&](const R3& x) {
                return eval_te_scattered_field(sl_pot_p0, dl_pot_p1,
                                               mesh, dof_p0, dof_p1,
                                               te_sk, gamma_inc_p0, x);
            },
            r_obs, n_obs, kx, ky);

        write_field_samples_csv("pec_te_sk_obs_r2.csv", samples);
        print_total_field_stats("TE direct second-kind total-field observations", samples, r_obs);
        std::cout << "Wrote observations to pec_te_sk_obs_r2.csv\n";
    }

    if (tm_fk.available && tm_sk.available)
    {
        const auto a = sample_total_field_on_circle(
            [&](const R3& x) {
                return eval_tm_scattered_field(sl_pot_p0, dl_pot_p1,
                                               mesh, dof_p0, dof_p1,
                                               tm_fk, beta_inc_p1, x);
            },
            r_obs, n_obs, kx, ky);

        const auto b = sample_total_field_on_circle(
            [&](const R3& x) {
                return eval_tm_scattered_field(sl_pot_p0, dl_pot_p1,
                                               mesh, dof_p0, dof_p1,
                                               tm_sk, beta_inc_p1, x);
            },
            r_obs, n_obs, kx, ky);

        compare_scattered_fields("TM direct formulations agreement on observation circle",
                                 a, b, "pec_tm_formulation_comparison_r2.csv");
        std::cout << "Wrote comparison to pec_tm_formulation_comparison_r2.csv\n";
    }

    if (te_fk.available && te_sk.available)
    {
        const auto a = sample_total_field_on_circle(
            [&](const R3& x) {
                return eval_te_scattered_field(sl_pot_p0, dl_pot_p1,
                                               mesh, dof_p0, dof_p1,
                                               te_fk, gamma_inc_p0, x);
            },
            r_obs, n_obs, kx, ky);

        const auto b = sample_total_field_on_circle(
            [&](const R3& x) {
                return eval_te_scattered_field(sl_pot_p0, dl_pot_p1,
                                               mesh, dof_p0, dof_p1,
                                               te_sk, gamma_inc_p0, x);
            },
            r_obs, n_obs, kx, ky);

        compare_scattered_fields("TE direct formulations agreement on observation circle",
                                 a, b, "pec_te_formulation_comparison_r2.csv");
        std::cout << "Wrote comparison to pec_te_formulation_comparison_r2.csv\n";
    }

    return 0;
}
