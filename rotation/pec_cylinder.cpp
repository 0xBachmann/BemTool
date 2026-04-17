#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <fstream>
#include <string>
#include <algorithm>
#include <functional>

#include "../bemtool/tools.hpp"

#include "helpers.hpp"

using namespace bemtool;
using namespace rotating_helpers;

namespace
{
    Cplx eval_tm_scattered_field(Potential<RH_SL_2D_P0>& sl_pot_p0,
                                 Potential<RH_DL_2D_P1>& dl_pot_p1,
                                 const Mesh1D& mesh,
                                 const Dof<P0_1D>& dof_p0,
                                 const Dof<P1_1D>& dof_p1,
                                 const InterfaceSolution& result,
                                 const Eigen::VectorXcd& beta_inc_p1,
                                 const R3& x)
    {
        // Section 3.6, eq. (3.89) / (3.90):
        //   u_sca = -Psi_SL f_N + Psi_DL g_D
        // with g_D = -beta_inc.
        const Cplx sl = eval_potential(sl_pot_p0, mesh, dof_p0, result.gamma_p0, x);
        const Cplx dl = eval_potential(dl_pot_p1, mesh, dof_p1, beta_inc_p1, x);
        return -sl - dl;
    }

    Cplx eval_te_scattered_field(Potential<RH_SL_2D_P0>& sl_pot_p0,
                                 Potential<RH_DL_2D_P1>& dl_pot_p1,
                                 const Mesh1D& mesh,
                                 const Dof<P0_1D>& dof_p0,
                                 const Dof<P1_1D>& dof_p1,
                                 const InterfaceSolution& result,
                                 const Eigen::VectorXcd& gamma_inc_p0,
                                 const R3& x)
    {
        // Section 3.6, eq. (3.111) / (3.112):
        //   u_sca = -Psi_SL g_N + Psi_DL f_D
        // with g_N = -gamma_inc.
        const Cplx sl = eval_potential(sl_pot_p0, mesh, dof_p0, gamma_inc_p0, x);
        const Cplx dl = eval_potential(dl_pot_p1, mesh, dof_p1, result.beta_p1, x);
        return sl + dl;
    }

    template <typename EvalScattered>
    std::vector<Sample> sample_total_field_on_grid(
        const EvalScattered& eval_scattered,
        Real xmin,
        Real xmax,
        Real ymin,
        Real ymax,
        int nx,
        int ny,
        Real kx,
        Real ky)
    {
        std::vector<Sample> samples;
        samples.reserve(static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny));

        for (int j = 0; j < ny; ++j)
        {
            const Real y = ymin + (ymax - ymin) * static_cast<Real>(j) / static_cast<Real>(ny - 1);

            for (int i = 0; i < nx; ++i)
            {
                const Real x = xmin + (xmax - xmin) * static_cast<Real>(i) / static_cast<Real>(nx - 1);

                R3 p;
                p[0] = x;
                p[1] = y;
                p[2] = 0.0;

                const bool inside = point_inside_circle(p, 1);

                Sample s;
                s.x = p;

                if (inside)
                {
                    s.uinc = 0.0;
                    s.usca = 0.0;
                    s.utot = 0.0;
                }
                else
                {
                    s.uinc = incident_plane_wave(p, kx, ky);
                    s.usca = eval_scattered(p);
                    s.utot = s.uinc + s.usca;
                }

                samples.push_back(s);
            }
        }

        return samples;
    }

    void condition_number_eval(const Eigen::MatrixXcd& A, std::string name)
    {
        Eigen::JacobiSVD<Eigen::MatrixXcd> svd(A);
        const auto& s = svd.singularValues();

        // Largest / smallest singular value
        std::cout << "condition number of " << name << " " << s(0) / s(s.size() - 1) << ", s(0) = " << s(0) << ", s(-1)"
            << s(s.size() - 1) << "\n";
    }
}

int main(int argc, char* argv[])
{
    constexpr Real k0 = 4.0;
    constexpr Real eps = 1.0;
    constexpr Real mu = 1.0;
    constexpr Real Omega = 0.0; // stationary PEC benchmark
    bool print_condition = false;

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

    std::cout << "nb_elt = " << nb_elt << "\n";
    std::cout << "nb_dof(P0) = " << NbDof(dof_p0) << "\n";
    std::cout << "nb_dof(P1) = " << NbDof(dof_p1) << "\n";

    BIOp<CST_1D_P0xP0> M_00_op(mesh, mesh);
    BIOp<CST_1D_P1xP0> M_10_op(mesh, mesh);
    BIOp<CST_1D_P0xP1> M_01_op(mesh, mesh);
    BIOp<CST_1D_P1xP1> M_11_op(mesh, mesh);
    const Eigen::MatrixXcd M_00 = assemble_biop_matrix(dof_p0, dof_p0, M_00_op, "Assembling M(P0xP0)");
    const Eigen::MatrixXcd M_10 = assemble_biop_matrix(dof_p1, dof_p0, M_10_op, "Assembling M(P1xP0)");
    const Eigen::MatrixXcd M_01 = assemble_biop_matrix(dof_p0, dof_p1, M_01_op, "Assembling M(P0xP1)");
    const Eigen::MatrixXcd M_11 = assemble_biop_matrix(dof_p1, dof_p1, M_11_op, "Assembling M(P1xP1)");

    auto beta_inc_fun = [&](const R3& x) -> Cplx
    {
        return dirichlet_trace_incident(x, kx, ky);
    };

    auto gamma_inc_fun = [&](const R3& x) -> Cplx
    {
        return neumann_trace_incident_circle(x, kx, ky);
    };

    const Eigen::VectorXcd beta_inc_p1 = L2ProjectionToP1(beta_inc_fun, mesh, dof_p1, M_11);
    const Eigen::VectorXcd gamma_inc_p0 = interpolate_to_dofs(mesh, dof_p0, gamma_inc_fun);

    Potential<RH_SL_2D_P0> sl_pot_p0(mesh, k0, eps, mu, Omega);
    Potential<RH_DL_2D_P1> dl_pot_p1(mesh, k0, eps, mu, Omega);

    BIOp<RH_SL_2D_P1xP0> V(mesh, mesh, k0, eps, mu, Omega); // P0 -> P1
    BIOp<RH_DL_2D_P1xP1> K(mesh, mesh, k0, eps, mu, Omega); // P1 -> P1
    BIOp<RH_TDL_2D_P0xP0> Kp(mesh, mesh, k0, eps, mu, Omega); // P0 -> P0
    BIOp<RH_HS_2D_P0xP1> W(mesh, mesh, k0, eps, mu, Omega); // P1 -> P0

    const Eigen::MatrixXcd V_mat = assemble_biop_matrix(dof_p1, dof_p0, V, "Assembling V(P1xP0)");
    const Eigen::MatrixXcd K_mat = assemble_biop_matrix(dof_p1, dof_p1, K, "Assembling K(P1xP1)");
    const Eigen::MatrixXcd Kp_mat = assemble_biop_matrix(dof_p0, dof_p0, Kp, "Assembling Kp(P0xP0)");
    const Eigen::MatrixXcd W_mat = assemble_biop_matrix(dof_p0, dof_p1, W, "Assembling W(P0xP1)");

    InterfaceSolution tm_fk, tm_sk, te_fk, te_sk;
    {
        // Eq. (3.86): V f_N = (1/2 I - K) beta_inc
        tm_fk.name = "tm_fk";
        tm_fk.A = V_mat;
        tm_fk.rhs = (0.5 * M_11 - K_mat) * beta_inc_p1;
        tm_fk.gamma_p0 = tm_fk.A.fullPivLu().solve(tm_fk.rhs);

        if (print_condition)
        {
            condition_number_eval(tm_fk.A, tm_fk.name);
        }
        const auto samples = sample_total_field_on_grid(
            [&](const R3& x)
            {
                return eval_tm_scattered_field(sl_pot_p0, dl_pot_p1,
                                               mesh, dof_p0, dof_p1,
                                               tm_fk, beta_inc_p1, x);
            },
            -2.5, 2.5, -2.5, 2.5,
            100, 100,
            kx, ky
        );
        std::string filename = "pec_" + tm_fk.name + "_grid_" + std::to_string(k0) + "_100x100.csv";
        write_samples_csv(filename, samples);
        std::cout << "Wrote grid samples to " << filename << "\n";
    }

    {
        // Eq. (3.88): (1/2 I + K') f_N = W beta_inc
        tm_sk.name = "tm_sk";
        tm_sk.A = 0.5 * M_00 + Kp_mat;
        tm_sk.rhs = W_mat * beta_inc_p1;
        tm_sk.gamma_p0 = tm_sk.A.fullPivLu().solve(tm_sk.rhs);

        if (print_condition)
        {
            condition_number_eval(tm_sk.A, tm_sk.name);
        }
        const auto samples = sample_total_field_on_grid(
            [&](const R3& x)
            {
                return eval_tm_scattered_field(sl_pot_p0, dl_pot_p1,
                                               mesh, dof_p0, dof_p1,
                                               tm_sk, beta_inc_p1, x);
            },
            -2.5, 2.5, -2.5, 2.5,
            100, 100,
            kx, ky
        );
        std::string filename = "pec_" + tm_sk.name + "_grid_" + std::to_string(k0) + "_100x100.csv";
        write_samples_csv(filename, samples);
        std::cout << "Wrote grid samples to " << filename << "\n";
    }

    {
        // Eq. (3.108): W f_D = (1/2 I + K') gamma_inc
        te_fk.name = "te_fk";
        te_fk.A = W_mat;
        te_fk.rhs = (0.5 * M_00 + Kp_mat) * gamma_inc_p0;
        te_fk.beta_p1 = te_fk.A.fullPivLu().solve(te_fk.rhs);

        if (print_condition)
        {
            condition_number_eval(te_fk.A, te_fk.name);
        }
        const auto samples = sample_total_field_on_grid(
            [&](const R3& x)
            {
                return eval_te_scattered_field(sl_pot_p0, dl_pot_p1,
                                               mesh, dof_p0, dof_p1,
                                               te_fk, beta_inc_p1, x);
            },
            -2.5, 2.5, -2.5, 2.5,
            100, 100,
            kx, ky
        );
        std::string filename = "pec_" + te_fk.name + "_grid_" + std::to_string(k0) + "_100x100.csv";
        write_samples_csv(filename, samples);
        std::cout << "Wrote grid samples to " << filename << "\n";
    }

    {
        // Eq. (3.110): (1/2 I - K) f_D = V gamma_inc
        te_sk.name = "te_sk";
        te_sk.A = 0.5 * M_11 - K_mat;
        te_sk.rhs = V_mat * gamma_inc_p0;
        te_sk.beta_p1 = te_sk.A.fullPivLu().solve(te_sk.rhs);

        if (print_condition)
        {
            condition_number_eval(te_sk.A, te_sk.name);
        }

        const auto samples = sample_total_field_on_grid(
            [&](const R3& x)
            {
                return eval_te_scattered_field(sl_pot_p0, dl_pot_p1,
                                               mesh, dof_p0, dof_p1,
                                               te_sk, beta_inc_p1, x);
            },
            -2.5, 2.5, -2.5, 2.5,
            100, 100,
            kx, ky
        );
        std::string filename = "pec_" + te_sk.name + "_grid_" + std::to_string(k0) + "_100x100.csv";
        write_samples_csv(filename, samples);
        std::cout << "Wrote grid samples to " << filename << "\n";
    }


    return 0;
}
