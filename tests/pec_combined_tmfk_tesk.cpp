
#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <fstream>
#include <string>

#include "../bemtool/tools.hpp"

using namespace bemtool;

namespace
{
    Cplx incident_plane_wave(const R3& x, Real kx, Real ky)
    {
        return std::exp(iu * (kx * x[0] + ky * x[1]));
    }

    // Problem-specific only: assemble rhs_i = \int_\Gamma g \phi_i ds,
    // where g is either the incident Dirichlet trace or the incident Neumann trace.
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

    template <typename op>
    Cplx eval_layer_field(Potential<op>& sl_pot,
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
            u += sl_pot(x, N2_(e, 0)) * dens(edof[0]);
            u += sl_pot(x, N2_(e, 1)) * dens(edof[1]);
        }
        return u;
    }

    template <class OperatorType>
    Eigen::MatrixXcd assemble_biop_matrix(const Mesh1D& mesh, const Dof<P1_1D>& dof, OperatorType& op,
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
} // namespace

int main(int argc, char* argv[])
{
    constexpr bool do_tm_fk = false;
    constexpr bool do_tm_sk = true;
    constexpr bool do_te_fk = false;
    constexpr bool do_te_sk = false;

    constexpr Real k0 = 2.0;
    constexpr Real eps = 1.0;
    constexpr Real mu = 1.0;
    constexpr Real Omega = 0.0; // stationary PEC benchmark

    constexpr Real theta_inc = 0.0;
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
    if (do_te_fk)
    {
        std::cout << "te_fk, ";
    }
    if (do_te_sk)
    {
        std::cout << "te_sk, ";
    }
    if (do_tm_fk)
    {
        std::cout << "tm_fk, ";
    }
    if (do_tm_sk)
    {
        std::cout << "tm_sk, ";
    }
    std::cout << "\n";

    std::cout << "nb_elt = " << nb_elt << "\n";
    std::cout << "nb_dof = " << nb_dof << "\n";

    // Identity term in weak form via the built-in constant operator.
    BIOp<CST_1D_P1xP1> M_op(mesh, mesh);
    const Eigen::MatrixXcd M = assemble_biop_matrix(mesh, dof, M_op, "Assembling M");

    Eigen::MatrixXcd A = Eigen::MatrixXcd::Zero(nb_dof, nb_dof);
    Eigen::VectorXcd rhs = Eigen::VectorXcd::Zero(nb_dof);
    Eigen::VectorXcd dens(nb_dof);

    Potential<RH_SL_2D_P1> sl_pot(mesh, k0, eps, mu, Omega);
    Potential<RH_DL_2D_P1> dl_pot(mesh, k0, eps, mu, Omega);

    constexpr Real delta = 1.0e-3;
    constexpr int n_bc = 400;

    if (do_tm_fk)
    {
        // TM-side PEC prototype = Dirichlet problem.
        //
        // First-kind formulation:
        //   u_sca = - SL(phi),
        //   gamma_D (u_inc + u_sca) = 0
        // => V phi = gamma_D u_inc.
        BIOp<RH_SL_2D_P1xP1> V(mesh, mesh, k0, eps, mu, Omega);
        A = assemble_biop_matrix(mesh, dof, V, "Assembling V");

        assemble_rhs_dirichlet_p1(mesh, dof, kx, ky, rhs);
        dens = A.fullPivLu().solve(rhs);

        const std::string bc_name = "pec_tm_fk_boundary_check.csv";
        std::ofstream fout_bc(bc_name);
        // Check Dirichlet condition near the boundary:
        //   u_tot ~ 0 on Gamma.
        fout_bc << "theta,x,y,real_utot,imag_utot,abs_utot\n";

        const Real r_bc = 1.0 + delta;
        Real max_abs_utot = 0.0;
        Real sum_sq_abs_utot = 0.0;

        for (int m = 0; m < n_bc; ++m)
        {
            const Real theta = 2.0 * M_PI * static_cast<Real>(m) / static_cast<Real>(n_bc);
            R3 x;
            x[0] = r_bc * std::cos(theta);
            x[1] = r_bc * std::sin(theta);
            x[2] = 0.0;

            const Cplx uinc = incident_plane_wave(x, kx, ky);
            const Cplx usca = -eval_layer_field(sl_pot, mesh, dof, dens, x);
            const Cplx utot = uinc + usca;
            const Real abs_utot = std::abs(utot);

            max_abs_utot = std::max(max_abs_utot, abs_utot);
            sum_sq_abs_utot += abs_utot * abs_utot;

            fout_bc << theta << ',' << x[0] << ',' << x[1] << ','
                << std::real(utot) << ',' << std::imag(utot) << ','
                << abs_utot << '\n';
        }

        const Real rms_abs_utot = std::sqrt(sum_sq_abs_utot / static_cast<Real>(n_bc));
        std::cout << "Wrote boundary check to " << bc_name << "\n";
        std::cout << "Boundary check at r = 1 + delta, delta = " << delta << "\n";
        std::cout << "max |u_tot| = " << max_abs_utot << "\n";
        std::cout << "rms |u_tot| = " << rms_abs_utot << "\n";
        Eigen::VectorXcd bc_res = A * dens - rhs;
        std::cout << "relative boundary residual = "
                  << bc_res.norm() / rhs.norm() << "\n";
        fout_bc.close();
    }

    if (do_tm_sk)
    {
        // TM-side PEC prototype = Dirichlet problem.
        //
        // First-kind formulation:
        //   u_sca = - SL(phi),
        //   gamma_D (u_inc + u_sca) = 0
        // => V phi = gamma_D u_inc.
        BIOp<RH_DL_2D_P1xP1> K(mesh, mesh, k0, eps, mu, Omega);

        const auto Kmat = assemble_biop_matrix(mesh, dof, K, "Assembling K");
        A = 0.5 * M + Kmat;

        assemble_rhs_dirichlet_p1(mesh, dof, kx, ky, rhs);
        rhs = -rhs;

        dens = A.fullPivLu().solve(rhs);
        const std::string bc_name = "pec_tm_sk_boundary_check.csv";
        std::ofstream fout_bc(bc_name);
        // Check Dirichlet condition near the boundary:
        //   u_tot ~ 0 on Gamma.
        fout_bc << "theta,x,y,real_utot,imag_utot,abs_utot\n";

        const Real r_bc = 1.0 + delta;
        Real max_abs_utot = 0.0;
        Real sum_sq_abs_utot = 0.0;

        for (int m = 0; m < n_bc; ++m)
        {
            const Real theta = 2.0 * M_PI * static_cast<Real>(m) / static_cast<Real>(n_bc);
            R3 x;
            x[0] = r_bc * std::cos(theta);
            x[1] = r_bc * std::sin(theta);
            x[2] = 0.0;

            const Cplx uinc = incident_plane_wave(x, kx, ky);
            const Cplx usca = eval_layer_field(dl_pot, mesh, dof, dens, x);
            const Cplx utot = uinc + usca;
            const Real abs_utot = std::abs(utot);

            max_abs_utot = std::max(max_abs_utot, abs_utot);
            sum_sq_abs_utot += abs_utot * abs_utot;

            fout_bc << theta << ',' << x[0] << ',' << x[1] << ','
                << std::real(utot) << ',' << std::imag(utot) << ','
                << abs_utot << '\n';
        }

        const Real rms_abs_utot = std::sqrt(sum_sq_abs_utot / static_cast<Real>(n_bc));
        std::cout << "Wrote boundary check to " << bc_name << "\n";
        std::cout << "Boundary check at r = 1 + delta, delta = " << delta << "\n";
        std::cout << "max |u_tot| = " << max_abs_utot << "\n";
        std::cout << "rms |u_tot| = " << rms_abs_utot << "\n";
        Eigen::VectorXcd bc_res = A * dens - rhs;
        std::cout << "relative boundary residual = "
                  << bc_res.norm() / rhs.norm() << "\n";
        fout_bc.close();
    }

    if (do_te_sk)
    {
        // TE-side PEC prototype = Neumann problem.
        //
        // Second-kind formulation:
        //   (-1/2 I + K') phi = - gamma_N u_inc
        // with u_sca = - SL(phi).
        //
        // This version uses the native RH adjoint-double-layer operator TDL_OP.
        BIOp<RH_TDL_2D_P1xP1> Kp(mesh, mesh, k0, eps, mu, Omega);
        const Eigen::MatrixXcd Kpmat = assemble_biop_matrix(mesh, dof, Kp, "Assembling Kp");

        A = -0.5 * M + Kpmat;
        assemble_rhs_neumann_p1(mesh, dof, kx, ky, rhs);
        // rhs = -rhs;

        dens = A.fullPivLu().solve(rhs);
        const std::string bc_name = "pec_te_sk_boundary_check.csv";
        std::ofstream fout_bc(bc_name);
        // Check Neumann condition near the boundary:
        //   partial_n u_tot ~ 0 on Gamma.
        //
        // Since the current test driver evaluates only the field potential,
        // we estimate the exterior normal derivative by a one-sided radial
        // finite difference.
        fout_bc << "theta,real_dnutot,imag_dnutot,abs_dnutot\n";

        Real max_abs_dnu = 0.0;
        Real sum_sq_abs_dnu = 0.0;

        for (int m = 0; m < n_bc; ++m)
        {
            const Real theta = 2.0 * M_PI * static_cast<Real>(m) / static_cast<Real>(n_bc);

            R3 x1, x2;
            x1[0] = (1.0 + delta) * std::cos(theta);
            x1[1] = (1.0 + delta) * std::sin(theta);
            x1[2] = 0.0;

            x2[0] = (1.0 + 2.0 * delta) * std::cos(theta);
            x2[1] = (1.0 + 2.0 * delta) * std::sin(theta);
            x2[2] = 0.0;

            const Cplx uinc1 = incident_plane_wave(x1, kx, ky);
            const Cplx uinc2 = incident_plane_wave(x2, kx, ky);
            const Cplx usca1 = -eval_layer_field(sl_pot, mesh, dof, dens, x1);
            const Cplx usca2 = -eval_layer_field(sl_pot, mesh, dof, dens, x2);

            const Cplx utot1 = uinc1 + usca1;
            const Cplx utot2 = uinc2 + usca2;

            const Cplx dnu_tot = (utot2 - utot1) / delta;
            const Real abs_dnu_tot = std::abs(dnu_tot);

            max_abs_dnu = std::max(max_abs_dnu, abs_dnu_tot);
            sum_sq_abs_dnu += abs_dnu_tot * abs_dnu_tot;

            fout_bc << theta << ','
                << std::real(dnu_tot) << ',' << std::imag(dnu_tot) << ','
                << abs_dnu_tot << '\n';
        }

        const Real rms_abs_dnu = std::sqrt(sum_sq_abs_dnu / static_cast<Real>(n_bc));
        std::cout << "Wrote boundary check to " << bc_name << "\n";
        std::cout << "Neumann check by one-sided radial finite difference, delta = " << delta << "\n";
        std::cout << "max |dn u_tot| ~= " << max_abs_dnu << "\n";
        std::cout << "rms |dn u_tot| ~= " << rms_abs_dnu << "\n";
        Eigen::VectorXcd bc_res = A * dens - rhs;
        std::cout << "relative boundary residual = "
                  << bc_res.norm() / rhs.norm() << "\n";
        fout_bc.close();
    }

    if (do_te_fk)
    {
        // TE-side PEC prototype = Neumann problem.
        //
        // Second-kind formulation:
        //   (-1/2 I + K') phi = - gamma_N u_inc
        // with u_sca = - SL(phi).
        //
        // This version uses the native RH adjoint-double-layer operator TDL_OP.
        BIOp<RH_HS_2D_P1xP1> W(mesh, mesh, k0, eps, mu, Omega);
        const Eigen::MatrixXcd Wmat = assemble_biop_matrix(mesh, dof, W, "Assembling W");

        A = Wmat;
        assemble_rhs_neumann_p1(mesh, dof, kx, ky, rhs);
        // rhs = -rhs;

        dens = A.fullPivLu().solve(rhs);
        const std::string bc_name = "pec_te_fk_boundary_check.csv";
        std::ofstream fout_bc(bc_name);
        // Check Neumann condition near the boundary:
        //   partial_n u_tot ~ 0 on Gamma.
        //
        // Since the current test driver evaluates only the field potential,
        // we estimate the exterior normal derivative by a one-sided radial
        // finite difference.
        fout_bc << "theta,real_dnutot,imag_dnutot,abs_dnutot\n";

        Real max_abs_dnu = 0.0;
        Real sum_sq_abs_dnu = 0.0;

        for (int m = 0; m < n_bc; ++m)
        {
            const Real theta = 2.0 * M_PI * static_cast<Real>(m) / static_cast<Real>(n_bc);

            R3 x1, x2;
            x1[0] = (1.0 + delta) * std::cos(theta);
            x1[1] = (1.0 + delta) * std::sin(theta);
            x1[2] = 0.0;

            x2[0] = (1.0 + 2.0 * delta) * std::cos(theta);
            x2[1] = (1.0 + 2.0 * delta) * std::sin(theta);
            x2[2] = 0.0;

            const Cplx uinc1 = incident_plane_wave(x1, kx, ky);
            const Cplx uinc2 = incident_plane_wave(x2, kx, ky);
            const Cplx usca1 = eval_layer_field(dl_pot, mesh, dof, dens, x1);
            const Cplx usca2 = eval_layer_field(dl_pot, mesh, dof, dens, x2);

            const Cplx utot1 = uinc1 + usca1;
            const Cplx utot2 = uinc2 + usca2;

            const Cplx dnu_tot = (utot2 - utot1) / delta;
            const Real abs_dnu_tot = std::abs(dnu_tot);

            max_abs_dnu = std::max(max_abs_dnu, abs_dnu_tot);
            sum_sq_abs_dnu += abs_dnu_tot * abs_dnu_tot;

            fout_bc << theta << ','
                << std::real(dnu_tot) << ',' << std::imag(dnu_tot) << ','
                << abs_dnu_tot << '\n';
        }

        const Real rms_abs_dnu = std::sqrt(sum_sq_abs_dnu / static_cast<Real>(n_bc));
        std::cout << "Wrote boundary check to " << bc_name << "\n";
        std::cout << "Neumann check by one-sided radial finite difference, delta = " << delta << "\n";
        std::cout << "max |dn u_tot| ~= " << max_abs_dnu << "\n";
        std::cout << "rms |dn u_tot| ~= " << rms_abs_dnu << "\n";
        Eigen::VectorXcd bc_res = A * dens - rhs;
        std::cout << "relative boundary residual = "
                  << bc_res.norm() / rhs.norm() << "\n";

        fout_bc.close();
    }

    return 0;
}
