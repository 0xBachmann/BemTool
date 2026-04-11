
#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <fstream>
#include <string>
#include <algorithm>

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

    void assemble_rhs_dirichlet_p1(const Mesh1D& mesh, const Dof<P1_1D>& dof,
                                   Real kx, Real ky, Eigen::VectorXcd& rhs)
    {
        static const Real gp[2] = {
            0.5 * (1.0 - 1.0 / std::sqrt(3.0)),
            0.5 * (1.0 + 1.0 / std::sqrt(3.0))
        };
        static const Real gw[2] = {0.5, 0.5};

        const int nb_elt = NbElt(mesh);
        for (int e = 0; e < nb_elt; ++e)
        {
            const auto& elt = mesh[e];
            const R3 a = elt[0];
            const R3 b = elt[1];
            const R3 tau = b - a;
            const Real h = std::sqrt(tau[0] * tau[0] + tau[1] * tau[1] + tau[2] * tau[2]);

            const N2& edof = dof[e];

            for (int q = 0; q < 2; ++q)
            {
                const Real t = gp[q];
                const R3 x = a + t * tau;
                const Cplx g = incident_plane_wave(x, kx, ky);
                const Real phi0 = 1.0 - t;
                const Real phi1 = t;

                rhs(edof[0]) += gw[q] * h * g * phi0;
                rhs(edof[1]) += gw[q] * h * g * phi1;
            }
        }
    }

    void assemble_rhs_neumann_p1(const Mesh1D& mesh, const Dof<P1_1D>& dof,
                                 Real kx, Real ky, Eigen::VectorXcd& rhs)
    {
        static const Real gp[2] = {
            0.5 * (1.0 - 1.0 / std::sqrt(3.0)),
            0.5 * (1.0 + 1.0 / std::sqrt(3.0))
        };
        static const Real gw[2] = {0.5, 0.5};

        const std::vector<R3>& normals = NormalTo(mesh);
        const int nb_elt = NbElt(mesh);

        for (int e = 0; e < nb_elt; ++e)
        {
            const auto& elt = mesh[e];
            const R3 a = elt[0];
            const R3 b = elt[1];
            const R3 tau = b - a;
            const Real h = std::sqrt(tau[0] * tau[0] + tau[1] * tau[1] + tau[2] * tau[2]);
            const R3 n = normals[e];

            const N2& edof = dof[e];

            for (int q = 0; q < 2; ++q)
            {
                const Real t = gp[q];
                const R3 x = a + t * tau;
                const Cplx uinc = incident_plane_wave(x, kx, ky);
                const Cplx g = iu * (kx * n[0] + ky * n[1]) * uinc;
                const Real phi0 = 1.0 - t;
                const Real phi1 = t;

                rhs(edof[0]) += gw[q] * h * g * phi0;
                rhs(edof[1]) += gw[q] * h * g * phi1;
            }
        }
    }

    template <typename Op>
    Cplx eval_layer_field(Potential<Op>& pot,
                          const Mesh1D& mesh,
                          const Dof<P1_1D>& dof,
                          const Eigen::VectorXcd& dens,
                          const R3& x)
    {
        const int nb_elt = NbElt(mesh);
        Cplx u = 0.0;
        for (int e = 0; e < nb_elt; ++e)
        {
            const N2& edof = dof[e];
            u += pot(x, N2_(e, 0)) * dens(edof[0]);
            u += pot(x, N2_(e, 1)) * dens(edof[1]);
        }
        return u;
    }

    template <class OperatorType>
    Eigen::MatrixXcd assemble_biop_matrix(const Mesh1D& mesh,
                                          const Dof<P1_1D>& dof,
                                          OperatorType& op,
                                          const std::string& label)
    {
        const int nb_dof = NbDof(dof);
        Eigen::MatrixXcd A = Eigen::MatrixXcd::Zero(nb_dof, nb_dof);

        progress bar(label.c_str(), nb_dof);
        for (int i = 0; i < nb_dof; ++i)
        {
            bar++;
            for (int j = 0; j < nb_dof; ++j)
            {
                A(i, j) += op(dof.ToElt(i), dof.ToElt(j));
            }
        }
        bar.end();
        return A;
    }

    std::vector<FieldSample> sample_total_field_on_circle(const Mesh1D& mesh,
                                                          const Dof<P1_1D>& dof,
                                                          Potential<HE_SL_2D_P1>& sl_pot,
                                                          Potential<HE_DL_2D_P1>& dl_pot,
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

    Dof<P1_1D> dof(mesh);
    const int nb_dof = NbDof(dof);
    const int nb_elt = NbElt(mesh);

    std::cout << "solving ";
    if (do_te_fk) std::cout << "te_fk, ";
    if (do_te_sk) std::cout << "te_sk, ";
    if (do_tm_fk) std::cout << "tm_fk, ";
    if (do_tm_sk) std::cout << "tm_sk, ";
    std::cout << "\n";

    std::cout << "nb_elt = " << nb_elt << "\n";
    std::cout << "nb_dof = " << nb_dof << "\n";

    BIOp<CST_1D_P1xP1> M_op(mesh, mesh);
    const Eigen::MatrixXcd M = assemble_biop_matrix(mesh, dof, M_op, "Assembling M");

    Potential<HE_SL_2D_P1> sl_pot(mesh, k0/*, eps, mu, Omega*/);
    Potential<HE_DL_2D_P1> dl_pot(mesh, k0/*, eps, mu, Omega*/);

    SolveResult tm_fk, tm_sk, te_sk, te_fk;

    if (do_tm_fk)
    {
        tm_fk.name = "tm_fk";
        tm_fk.uses_sl_potential = true;
        tm_fk.available = true;

        BIOp<HE_SL_2D_P1xP1> V(mesh, mesh, k0/*, eps, mu, Omega*/);
        tm_fk.A = assemble_biop_matrix(mesh, dof, V, "Assembling V");

        tm_fk.rhs = Eigen::VectorXcd::Zero(nb_dof);
        assemble_rhs_dirichlet_p1(mesh, dof, kx, ky, tm_fk.rhs);

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

        BIOp<HE_DL_2D_P1xP1> K(mesh, mesh, k0/*, eps, mu, Omega*/);
        const Eigen::MatrixXcd Kmat = assemble_biop_matrix(mesh, dof, K, "Assembling K");
        tm_sk.A = 0.5 * M - Kmat;

        tm_sk.rhs = Eigen::VectorXcd::Zero(nb_dof);
        assemble_rhs_dirichlet_p1(mesh, dof, kx, ky, tm_sk.rhs);
        tm_sk.rhs = -tm_sk.rhs;

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

        BIOp<HE_TDL_2D_P1xP1> Kp(mesh, mesh, k0/*, eps, mu, Omega*/);
        const Eigen::MatrixXcd Kpmat = assemble_biop_matrix(mesh, dof, Kp, "Assembling Kp");
        te_sk.A = -0.5 * M + Kpmat;

        te_sk.rhs = Eigen::VectorXcd::Zero(nb_dof);
        assemble_rhs_neumann_p1(mesh, dof, kx, ky, te_sk.rhs);

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

        BIOp<HE_HS_2D_P1xP1> W(mesh, mesh, k0/*, eps, mu, Omega*/);
        te_fk.A = assemble_biop_matrix(mesh, dof, W, "Assembling W");

        te_fk.rhs = Eigen::VectorXcd::Zero(nb_dof);
        assemble_rhs_neumann_p1(mesh, dof, kx, ky, te_fk.rhs);

        te_fk.density = te_fk.A.fullPivLu().solve(te_fk.rhs);
        const Eigen::VectorXcd res = te_fk.A * te_fk.density - te_fk.rhs;
        te_fk.relative_bie_residual = res.norm() / std::max(te_fk.rhs.norm(), Real(1.0e-30));

        std::cout << "relative BIE residual (te_fk) = " << te_fk.relative_bie_residual << "\n";
    }

    if (tm_fk.available)
    {
        const auto samples = sample_total_field_on_circle(mesh, dof, sl_pot, dl_pot, tm_fk, r_obs, n_obs, kx, ky);
        write_field_samples_csv("pec_tm_fk_obs_r2.csv", samples);
        print_total_field_stats("TM first-kind total-field check", samples, r_obs);
        std::cout << "Wrote observations to pec_tm_fk_obs_r2.csv\n";
    }

    if (tm_sk.available)
    {
        const auto samples = sample_total_field_on_circle(mesh, dof, sl_pot, dl_pot, tm_sk, r_obs, n_obs, kx, ky);
        write_field_samples_csv("pec_tm_sk_obs_r2.csv", samples);
        print_total_field_stats("TM second-kind total-field check", samples, r_obs);
        std::cout << "Wrote observations to pec_tm_sk_obs_r2.csv\n";
    }

    if (te_sk.available)
    {
        const auto samples = sample_total_field_on_circle(mesh, dof, sl_pot, dl_pot, te_sk, r_obs, n_obs, kx, ky);
        write_field_samples_csv("pec_te_sk_obs_r2.csv", samples);
        print_total_field_stats("TE second-kind total-field observations", samples, r_obs);
        std::cout << "Wrote observations to pec_te_sk_obs_r2.csv\n";
    }

    if (te_fk.available)
    {
        const auto samples = sample_total_field_on_circle(mesh, dof, sl_pot, dl_pot, te_fk, r_obs, n_obs, kx, ky);
        write_field_samples_csv("pec_te_fk_obs_r2.csv", samples);
        print_total_field_stats("TE first-kind total-field observations", samples, r_obs);
        std::cout << "Wrote observations to pec_te_fk_obs_r2.csv\n";
    }

    if (tm_fk.available && tm_sk.available)
    {
        const auto a = sample_total_field_on_circle(mesh, dof, sl_pot, dl_pot, tm_fk, r_obs, n_obs, kx, ky);
        const auto b = sample_total_field_on_circle(mesh, dof, sl_pot, dl_pot, tm_sk, r_obs, n_obs, kx, ky);
        compare_scattered_fields("TM formulation agreement on observation circle",
                                 a, b, "pec_tm_formulation_comparison_r2.csv");
        std::cout << "Wrote comparison to pec_tm_formulation_comparison_r2.csv\n";
    }

    if (te_sk.available && te_fk.available)
    {
        const auto a = sample_total_field_on_circle(mesh, dof, sl_pot, dl_pot, te_sk, r_obs, n_obs, kx, ky);
        const auto b = sample_total_field_on_circle(mesh, dof, sl_pot, dl_pot, te_fk, r_obs, n_obs, kx, ky);
        compare_scattered_fields("TE formulation agreement on observation circle",
                                 a, b, "pec_te_formulation_comparison_r2.csv");
        std::cout << "Wrote comparison to pec_te_formulation_comparison_r2.csv\n";
    }

    return 0;
}
