#ifndef DIELECTRIC_CYLINDER_TRANSMISSION_TEST_CPP
#define DIELECTRIC_CYLINDER_TRANSMISSION_TEST_CPP

#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <fstream>
#include <string>
#include <algorithm>
#include <functional>
#include <sstream>
#include <iomanip>

#include "tests/helpers.hpp"

#include "../bemtool/tools.hpp"

using namespace bemtool;
using namespace rotating_helpers;

namespace
{
    // ============================================================
    // Block helpers
    // ============================================================
    Eigen::MatrixXcd make_block_2x2(const Eigen::MatrixXcd& A11,
                                    const Eigen::MatrixXcd& A12,
                                    const Eigen::MatrixXcd& A21,
                                    const Eigen::MatrixXcd& A22)
    {
        const int n1 = static_cast<int>(A11.rows());
        const int n2 = static_cast<int>(A22.rows());

        Eigen::MatrixXcd A = Eigen::MatrixXcd::Zero(n1 + n2, n1 + n2);
        A.topLeftCorner(n1, n1) = A11;
        A.topRightCorner(n1, n2) = A12;
        A.bottomLeftCorner(n2, n1) = A21;
        A.bottomRightCorner(n2, n2) = A22;
        return A;
    }

    Eigen::VectorXcd make_block_rhs(const Eigen::VectorXcd& b1,
                                    const Eigen::VectorXcd& b2)
    {
        Eigen::VectorXcd b(b1.size() + b2.size());
        b << b1, b2;
        return b;
    }

    void split_block_solution(const Eigen::VectorXcd& x,
                              int n_beta,
                              Eigen::VectorXcd& beta,
                              Eigen::VectorXcd& gamma)
    {
        beta = x.head(n_beta);
        gamma = x.tail(x.size() - n_beta);
    }

    std::string format_real_for_filename(Real value)
    {
        std::ostringstream oss;
        oss << std::scientific << std::setprecision(6) << static_cast<double>(value);
        std::string s = oss.str();
        std::replace(s.begin(), s.end(), '+', 'p');
        std::replace(s.begin(), s.end(), '-', 'm');
        std::replace(s.begin(), s.end(), '.', 'p');
        return s;
    }

    std::string make_parameter_postfix(Real k0, Real Omega)
    {
        return std::string("_k0_") + format_real_for_filename(k0)
            + std::string("_Omega_") + format_real_for_filename(Omega);
    }

    template <typename SLPotExt, typename DLPotExt, typename SLPotInt, typename DLPotInt>
    std::vector<Sample> sample_total_field_on_grid(
        const Mesh1D& mesh,
        const Dof<P0_1D>& dof_p0,
        const Dof<P1_1D>& dof_p1,
        Potential<SLPotExt>& sl_pot_ext,
        Potential<DLPotExt>& dl_pot_ext,
        Potential<SLPotInt>& sl_pot_int,
        Potential<DLPotInt>& dl_pot_int,
        const InterfaceSolution& sol,
        const Eigen::VectorXcd& beta_inc_p1,
        const Eigen::VectorXcd& gamma_inc_p0,
        Real xmin,
        Real xmax,
        Real ymin,
        Real ymax,
        int nx,
        int ny,
        Real kx,
        Real ky,
        Real interface_radius,
        Real skip_tolerance)
    {
        std::vector<Sample> samples;
        samples.reserve(nx * ny);

        progress bar("sample field on grid", nx * ny);
        for (int iy = 0; iy < ny; ++iy)
        {
            const Real y = ymin + (ymax - ymin) * static_cast<Real>(iy) / static_cast<Real>(std::max(1, ny - 1));
            for (int ix = 0; ix < nx; ++ix, bar++)
            {
                const Real x0 = xmin + (xmax - xmin) * static_cast<Real>(ix) / static_cast<Real>(std::max(1, nx - 1));

                R3 x;
                x[0] = x0;
                x[1] = y;
                x[2] = 0.0;

                Sample s;
                s.x = x;

                if (point_too_close_to_circle(x, interface_radius, skip_tolerance))
                {
                    continue;
                }

                if (point_inside_circle(x, interface_radius))
                {
                    s.utr = eval_interior_transmitted_field(
                        sl_pot_int, dl_pot_int,
                        mesh, dof_p0, dof_p1,
                        sol.beta_p1, sol.gamma_p0, x);
                    s.utot = s.utr;
                }
                else
                {
                    s.uinc = incident_plane_wave(x, kx, ky);
                    s.usca = eval_exterior_scattered_field(
                        sl_pot_ext, dl_pot_ext,
                        mesh, dof_p0, dof_p1,
                        sol.beta_p1, sol.gamma_p0,
                        beta_inc_p1, gamma_inc_p0, x);
                    s.utot = s.uinc + s.usca;
                }

                samples.push_back(s);
            }
        }
        bar.end();

        return samples;
    }
}

int main(int argc, char* argv[])
{
    // ============================================================
    // Configuration
    // ============================================================
    Real k0 = 3.;
    if (argc > 1)
    {
        k0 = std::strtod(argv[1], nullptr);
    }

    constexpr Real eps_e = 1.0;
    constexpr Real mu_e = 1.0;

    constexpr Real eps_i = 2.25; // n_i = 1.5 if mu_i = 1
    constexpr Real mu_i = 1.0;

    constexpr Real Omega_e = 0.0;
    Real Omega_i = 0;
    if (argc > 2)
    {
        Omega_i = std::strtod(argv[2], nullptr);
    }

    constexpr Real theta_inc = 0.0;
    constexpr Real grid_min = -2.0;
    constexpr Real grid_max = 2.0;
    constexpr int grid_nx = 151;
    constexpr int grid_ny = 151;
    constexpr Real interface_radius = 1.0;
    constexpr Real interface_skip_tol = 0.003;

    const Real n_e = std::sqrt(eps_e * mu_e);
    Real n_i = std::sqrt(eps_i * mu_i);
    std::string out_dir = ".";
    if (argc > 3)
    {
        n_i = std::strtod(argv[3], nullptr);
        out_dir = "n" + std::string(argv[3]);
    }

    std::cout << "Using parameters k0 = " << k0 << ", Omega_i = " << Omega_i << ", n_i = " << n_i << "\n";

    const Real k_e = k0 * n_e;
    const Real k_i = k0 * n_i;

    const Real kx = k_e * std::cos(theta_inc);
    const Real ky = k_e * std::sin(theta_inc);

    // ============================================================
    // Geometry / dofs
    // ============================================================

    Geometry node("../mesh/circle.msh");
    Mesh1D mesh;
    mesh.Load(node, 1);
    Orienting(mesh);

    Dof<P0_1D> dof_p0(mesh); // gamma
    Dof<P1_1D> dof_p1(mesh); // beta

    const int nb_elt = NbElt(mesh);
    const int nb_p0 = NbDof(dof_p0);
    const int nb_p1 = NbDof(dof_p1);

    std::cout << "Dielectric transmission test\n";
    std::cout << "nb_elt = " << nb_elt << "\n";
    std::cout << "nb_dof(P0) = " << nb_p0 << "\n";
    std::cout << "nb_dof(P1) = " << nb_p1 << "\n";

    // ============================================================
    // Mass matrices for trace interpolation
    // ============================================================

    BIOp<CST_1D_P0xP0> M_00_op(mesh, mesh);
    BIOp<CST_1D_P1xP0> M_10_op(mesh, mesh);
    BIOp<CST_1D_P0xP1> M_01_op(mesh, mesh);
    BIOp<CST_1D_P1xP1> M_11_op(mesh, mesh);
    const Eigen::MatrixXcd M_00 = assemble_biop_matrix(dof_p0, dof_p0, M_00_op, "Assembling M(P0xP0)");
    const Eigen::MatrixXcd M_10 = assemble_biop_matrix(dof_p1, dof_p0, M_10_op, "Assembling M(P1xP0)");
    const Eigen::MatrixXcd M_01 = assemble_biop_matrix(dof_p0, dof_p1, M_01_op, "Assembling M(P0xP1)");
    const Eigen::MatrixXcd M_11 = assemble_biop_matrix(dof_p1, dof_p1, M_11_op, "Assembling M(P1xP1)");

    // ============================================================
    // Incident traces (exterior medium)
    // ============================================================

    auto beta_inc_fun = [&](const R3& x) -> Cplx
    {
        return dirichlet_trace_incident(x, kx, ky);
    };

    auto gamma_inc_fun = [&](const R3& x) -> Cplx
    {
        return neumann_trace_incident_circle(x, kx, ky);
    };

    // const Eigen::VectorXcd beta_inc_p1 = L2WithP1Basis(beta_inc_fun, mesh, dof_p1);
    // const Eigen::VectorXcd gamma_inc_p0 = assemble_rhs_from_interpolation(mesh, dof_p0, M_00, gamma_inc_fun);
    const Eigen::VectorXcd beta_inc_p1 = L2ProjectionToP1(beta_inc_fun, mesh, dof_p1, M_11);
    const Eigen::VectorXcd gamma_inc_p0 = interpolate_to_dofs(mesh, dof_p0, gamma_inc_fun);
    // ============================================================
    // Exterior potentials and operators (assemble once)
    // ============================================================

    Potential<RH_SL_2D_P0> sl_pot_ext_p0(mesh, k_e, eps_e, mu_e, Omega_e);
    Potential<RH_DL_2D_P1> dl_pot_ext_p1(mesh, k_e, eps_e, mu_e, Omega_e);

    BIOp<RH_SL_2D_P1xP0> Ve(mesh, mesh, k_e, eps_e, mu_e, Omega_e); // gamma(P0) -> beta(P1)
    BIOp<RH_DL_2D_P1xP1> Ke(mesh, mesh, k_e, eps_e, mu_e, Omega_e); // beta(P1)  -> beta(P1)
    BIOp<RH_TDL_2D_P0xP0> Kpe(mesh, mesh, k_e, eps_e, mu_e, Omega_e); // gamma(P0) -> gamma(P0)
    BIOp<RH_HS_2D_P0xP1> We(mesh, mesh, k_e, eps_e, mu_e, Omega_e); // beta(P1)  -> gamma(P0)

    const Eigen::MatrixXcd Ve_mat = assemble_biop_matrix(dof_p1, dof_p0, Ve, "Assembling Ve");
    const Eigen::MatrixXcd Ke_mat = assemble_biop_matrix(dof_p1, dof_p1, Ke, "Assembling Ke");
    const Eigen::MatrixXcd We_mat = assemble_biop_matrix(dof_p0, dof_p1, We, "Assembling We");
    const Eigen::MatrixXcd Kpe_mat = assemble_biop_matrix(dof_p0, dof_p0, Kpe, "Assembling Kpe");

    const Eigen::VectorXcd f_inc =
        (0.5 * M_11 - Ke_mat) * beta_inc_p1 + Ve_mat * gamma_inc_p0;

    const Eigen::VectorXcd g_inc =
        We_mat * beta_inc_p1 + (0.5 * M_00 + Kpe_mat) * gamma_inc_p0;

    // ============================================================
    // Loop over interior rotation parameters
    // ============================================================


    std::cout << "\n========================================\n";
    std::cout << "Running dielectric case with k0 = " << k0
        << ", Omega_i = " << Omega_i << "\n";
    std::cout << "========================================\n";

    const std::string postfix = make_parameter_postfix(k0, Omega_i);

    InterfaceSolution sol;
    sol.name = std::string("dielectric") + postfix;

    // Interior potentials and operators (reassembled for each Omega_i)
    Potential<RH_SL_2D_P0> sl_pot_int_p0(mesh, k_i, eps_i, mu_i, Omega_i);
    Potential<RH_DL_2D_P1> dl_pot_int_p1(mesh, k_i, eps_i, mu_i, Omega_i);

    BIOp<RH_SL_2D_P1xP0> Vi(mesh, mesh, k_i, eps_i, mu_i, Omega_i);
    BIOp<RH_DL_2D_P1xP1> Ki(mesh, mesh, k_i, eps_i, mu_i, Omega_i);
    BIOp<RH_TDL_2D_P0xP0> Kpi(mesh, mesh, k_i, eps_i, mu_i, Omega_i);
    BIOp<RH_HS_2D_P0xP1> Wi(mesh, mesh, k_i, eps_i, mu_i, Omega_i);

    const Eigen::MatrixXcd Vi_mat = assemble_biop_matrix(dof_p1, dof_p0, Vi, "Assembling Vi");
    const Eigen::MatrixXcd Ki_mat = assemble_biop_matrix(dof_p1, dof_p1, Ki, "Assembling Ki");
    const Eigen::MatrixXcd Wi_mat = assemble_biop_matrix(dof_p0, dof_p1, Wi, "Assembling Wi");
    const Eigen::MatrixXcd Kpi_mat = assemble_biop_matrix(dof_p0, dof_p0, Kpi, "Assembling Kpi");

    // System from thesis Section 3.7:
    // [ -Ke-Ki      -Ve+Vi ] [beta ] = [ f_inc ]
    // [ -We+Wi   Kpe+Kpi   ] [gamma]   [ -g_inc]
    const Eigen::MatrixXcd A11 = -Ke_mat - Ki_mat;
    const Eigen::MatrixXcd A12 = -Ve_mat + Vi_mat;
    const Eigen::MatrixXcd A21 = -We_mat + Wi_mat;
    const Eigen::MatrixXcd A22 = Kpe_mat + Kpi_mat;

    sol.A = make_block_2x2(A11, A12, A21, A22);
    sol.rhs = make_block_rhs(f_inc, -g_inc);

    // Eigen::JacobiSVD<Eigen::MatrixXcd> svd(sol.A);
    // const auto& s = svd.singularValues();

    // // Largest / smallest singular value
    // std::cout << "condition " << s(0) / s(s.size() - 1) << "s(0) = " << s(0) << "s(-1)" << s(s.size() - 1) << "\n";
    // std::exit(0);

    std::cout << "solving system\n";
    const Eigen::VectorXcd x = sol.A.fullPivLu().solve(sol.rhs);
    split_block_solution(x, nb_p1, sol.beta_p1, sol.gamma_p0);
    std::cout << "done\n";

    auto grid_samples = sample_total_field_on_grid(
        mesh,
        dof_p0,
        dof_p1,
        sl_pot_ext_p0,
        dl_pot_ext_p1,
        sl_pot_int_p0,
        dl_pot_int_p1,
        sol,
        beta_inc_p1,
        gamma_inc_p0,
        grid_min,
        grid_max,
        grid_min,
        grid_max,
        grid_nx,
        grid_ny,
        kx,
        ky,
        interface_radius,
        interface_skip_tol);
    std::cout << "returned from sample_total_field_on_grid\n";

    write_samples_csv(out_dir + "/dielectric_grid" + postfix + ".csv", grid_samples);


    return 0;
}

#endif
