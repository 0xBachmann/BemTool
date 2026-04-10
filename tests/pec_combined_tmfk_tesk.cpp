
#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <fstream>
#include <string>

#include "../bemtool/tools.hpp"

using namespace bemtool;

namespace {

Cplx incident_plane_wave(const R3& x, Real kx, Real ky) {
    return std::exp(iu * (kx * x[0] + ky * x[1]));
}

R3 outward_normal_from_segment(const R3& a, const R3& b) {
    const R3 tau = b - a;
    const Real h = std::sqrt(tau[0]*tau[0] + tau[1]*tau[1] + tau[2]*tau[2]);
    R3 n;
    // For positively oriented (counterclockwise) boundary, this is the exterior normal.
    n[0] =  tau[1] / h;
    n[1] = -tau[0] / h;
    n[2] =  0.0;
    return n;
}

// 2-point Gauss rule on [0,1]
static const Real gp[2] = {
    0.5 * (1.0 - 1.0 / std::sqrt(3.0)),
    0.5 * (1.0 + 1.0 / std::sqrt(3.0))
};
static const Real gw[2] = {0.5, 0.5};

void assemble_mass_p1(const Mesh1D& mesh, const Dof<P1_1D>& dof, Eigen::MatrixXcd& M) {
    const int nb_elt = NbElt(mesh);
    for (int e = 0; e < nb_elt; ++e) {
        const auto& elt = mesh[e];
        const R3 a = elt[0];
        const R3 b = elt[1];
        const R3 tau = b - a;
        const Real h = std::sqrt(tau[0]*tau[0] + tau[1]*tau[1] + tau[2]*tau[2]);
        const N2& edof = dof[e];

        for (int q = 0; q < 2; ++q) {
            const Real t = gp[q];
            const Real phi0 = 1.0 - t;
            const Real phi1 = t;
            const Real w = gw[q] * h;

            M(edof[0], edof[0]) += w * phi0 * phi0;
            M(edof[0], edof[1]) += w * phi0 * phi1;
            M(edof[1], edof[0]) += w * phi1 * phi0;
            M(edof[1], edof[1]) += w * phi1 * phi1;
        }
    }
}

// rhs_i = \int_\Gamma u_inc \phi_i ds
void assemble_rhs_dirichlet_p1(const Mesh1D& mesh, const Dof<P1_1D>& dof,
                               Real kx, Real ky, Eigen::VectorXcd& rhs) {
    const int nb_elt = NbElt(mesh);
    for (int e = 0; e < nb_elt; ++e) {
        const auto& elt = mesh[e];
        const R3 a = elt[0];
        const R3 b = elt[1];
        const R3 tau = b - a;
        const Real h = std::sqrt(tau[0]*tau[0] + tau[1]*tau[1] + tau[2]*tau[2]);

        const N2& edof = dof[e];

        for (int q = 0; q < 2; ++q) {
            const Real t = gp[q];
            const R3 x = a + t * tau;
            const Cplx uinc = incident_plane_wave(x, kx, ky);
            const Real phi0 = 1.0 - t;
            const Real phi1 = t;

            rhs(edof[0]) += gw[q] * h * uinc * phi0;
            rhs(edof[1]) += gw[q] * h * uinc * phi1;
        }
    }
}

// rhs_i = \int_\Gamma (\partial_n u_inc) \phi_i ds
void assemble_rhs_neumann_p1(const Mesh1D& mesh, const Dof<P1_1D>& dof,
                             Real kx, Real ky, Eigen::VectorXcd& rhs) {
    const int nb_elt = NbElt(mesh);
    for (int e = 0; e < nb_elt; ++e) {
        const auto& elt = mesh[e];
        const R3 a = elt[0];
        const R3 b = elt[1];
        const R3 tau = b - a;
        const Real h = std::sqrt(tau[0]*tau[0] + tau[1]*tau[1] + tau[2]*tau[2]);
        const R3 n = outward_normal_from_segment(a, b);

        const N2& edof = dof[e];

        for (int q = 0; q < 2; ++q) {
            const Real t = gp[q];
            const R3 x = a + t * tau;
            const Cplx uinc = incident_plane_wave(x, kx, ky);
            const Cplx dn_uinc = iu * (kx * n[0] + ky * n[1]) * uinc;
            const Real phi0 = 1.0 - t;
            const Real phi1 = t;

            rhs(edof[0]) += gw[q] * h * dn_uinc * phi0;
            rhs(edof[1]) += gw[q] * h * dn_uinc * phi1;
        }
    }
}

Cplx eval_single_layer_field(Potential<RH_SL_2D_P1>& sl_pot,
                             const Mesh1D& mesh,
                             const Dof<P1_1D>& dof,
                             const Eigen::VectorXcd& dens,
                             const R3& x) {
    const int nb_elt = NbElt(mesh);
    Cplx u = 0.0;
    for (int e = 0; e < nb_elt; ++e) {
        const N2& edof = dof[e];
        u -= sl_pot(x, N2_(e,0)) * dens(edof[0]);
        u -= sl_pot(x, N2_(e,1)) * dens(edof[1]);
    }
    return u;
}

} // namespace

int main(int argc, char* argv[]) {
    // mode = "tm_fk" or "te_sk"
    std::string mode = "tm_fk";
    if (argc > 1) {
        mode = argv[1];
    }

    const bool do_tm_fk = (mode == "tm_fk");
    const bool do_te_sk = (mode == "te_sk");

    if (!do_tm_fk && !do_te_sk) {
        std::cerr << "Unknown mode '" << mode << "'. Use tm_fk or te_sk.\n";
        return 1;
    }

    const Real k0 = 2.0;
    const Real eps = 1.0;
    const Real mu  = 1.0;
    const Real Omega = 0.0; // stationary PEC benchmark

    const Real theta_inc = 0.0;
    const Real kx = k0 * std::cos(theta_inc);
    const Real ky = k0 * std::sin(theta_inc);

    Geometry node("../mesh/circle.msh");
    Mesh1D mesh;
    mesh.Load(node, 1);
    Orienting(mesh);

    Dof<P1_1D> dof(mesh);
    const int nb_dof = NbDof(dof);
    const int nb_elt = NbElt(mesh);

    std::cout << "mode = " << mode << "\n";
    std::cout << "nb_elt = " << nb_elt << "\n";
    std::cout << "nb_dof = " << nb_dof << "\n";

    Eigen::MatrixXcd A = Eigen::MatrixXcd::Zero(nb_dof, nb_dof);
    Eigen::MatrixXcd M = Eigen::MatrixXcd::Zero(nb_dof, nb_dof);
    Eigen::VectorXcd rhs = Eigen::VectorXcd::Zero(nb_dof);
    Eigen::VectorXcd dens(nb_dof);

    assemble_mass_p1(mesh, dof, M);

    if (do_tm_fk) {
        // TM-side PEC prototype = Dirichlet problem.
        //
        // First-kind formulation:
        //   u_sca = - SL(phi),
        //   gamma_D (u_inc + u_sca) = 0
        // => V phi = gamma_D u_inc.
        BIOp<RH_SL_2D_P1xP1> V(mesh, mesh, k0, eps, mu, Omega);

        progress bar("Assembling V", nb_dof);
        for (int i = 0; i < nb_dof; ++i) {
            bar++;
            for (int j = 0; j < nb_dof; ++j) {
                A(i, j) += V(dof.ToElt(i), dof.ToElt(j));
            }
        }
        bar.end();

        assemble_rhs_dirichlet_p1(mesh, dof, kx, ky, rhs);
        dens = A.fullPivLu().solve(rhs);
    }

    if (do_te_sk) {
        // TE-side PEC prototype = Neumann problem.
        //
        // Second-kind formulation:
        //   (-1/2 I + K') phi = - gamma_N u_inc
        // with u_sca = - SL(phi).
        //
        // No dedicated K' wrapper was found in the current RH header.
        // For the stationary case Omega = 0, we approximate
        //   K' = K^*
        // by assembling K from the double-layer operator and taking
        // its Hermitian adjoint matrix.
        BIOp<RH_DL_2D_P1xP1> K(mesh, mesh, k0, eps, mu, Omega);

        Eigen::MatrixXcd Kmat = Eigen::MatrixXcd::Zero(nb_dof, nb_dof);
        progress bar("Assembling K", nb_dof);
        for (int i = 0; i < nb_dof; ++i) {
            bar++;
            for (int j = 0; j < nb_dof; ++j) {
                Kmat(i, j) += K(dof.ToElt(i), dof.ToElt(j));
            }
        }
        bar.end();

        const Eigen::MatrixXcd Kp = Kmat.adjoint();
        A = -0.5 * M + Kp;

        assemble_rhs_neumann_p1(mesh, dof, kx, ky, rhs);
        rhs = -rhs;
        dens = A.fullPivLu().solve(rhs);
    }

    Eigen::VectorXcd res = A * dens - rhs;
    std::cout << "relative residual = " << res.norm() / rhs.norm() << "\n";

    Potential<RH_SL_2D_P1> sl_pot(mesh, k0, eps, mu, Omega);

    // Field observations
    const Real r_obs = 2.0;
    const int n_obs = 200;
    const std::string obs_name = "pec_" + mode + "_obs.csv";
    std::ofstream fout_obs(obs_name);
    fout_obs << "theta,x,y,real_uinc,imag_uinc,real_usca,imag_usca,real_utot,imag_utot\n";

    for (int m = 0; m < n_obs; ++m) {
        const Real theta = 2.0 * M_PI * static_cast<Real>(m) / static_cast<Real>(n_obs);
        R3 x;
        x[0] = r_obs * std::cos(theta);
        x[1] = r_obs * std::sin(theta);
        x[2] = 0.0;

        const Cplx uinc = incident_plane_wave(x, kx, ky);
        const Cplx usca = eval_single_layer_field(sl_pot, mesh, dof, dens, x);
        const Cplx utot = uinc + usca;

        fout_obs << theta << ',' << x[0] << ',' << x[1] << ','
                 << std::real(uinc) << ',' << std::imag(uinc) << ','
                 << std::real(usca) << ',' << std::imag(usca) << ','
                 << std::real(utot) << ',' << std::imag(utot) << '\n';
    }
    fout_obs.close();

    // Boundary verification
    const Real delta = 1.0e-3;
    const int n_bc = 400;
    const std::string bc_name = "pec_" + mode + "_boundary_check.csv";
    std::ofstream fout_bc(bc_name);

    if (do_tm_fk) {
        // Check Dirichlet condition near the boundary:
        //   u_tot ~ 0 on Gamma.
        fout_bc << "theta,x,y,real_utot,imag_utot,abs_utot\n";

        const Real r_bc = 1.0 + delta;
        Real max_abs_utot = 0.0;
        Real sum_sq_abs_utot = 0.0;

        for (int m = 0; m < n_bc; ++m) {
            const Real theta = 2.0 * M_PI * static_cast<Real>(m) / static_cast<Real>(n_bc);
            R3 x;
            x[0] = r_bc * std::cos(theta);
            x[1] = r_bc * std::sin(theta);
            x[2] = 0.0;

            const Cplx uinc = incident_plane_wave(x, kx, ky);
            const Cplx usca = eval_single_layer_field(sl_pot, mesh, dof, dens, x);
            const Cplx utot = uinc + usca;
            const Real abs_utot = std::abs(utot);

            max_abs_utot = std::max(max_abs_utot, abs_utot);
            sum_sq_abs_utot += abs_utot * abs_utot;

            fout_bc << theta << ',' << x[0] << ',' << x[1] << ','
                    << std::real(utot) << ',' << std::imag(utot) << ','
                    << abs_utot << '\n';
        }

        const Real rms_abs_utot = std::sqrt(sum_sq_abs_utot / static_cast<Real>(n_bc));
        std::cout << "Wrote observations to " << obs_name << "\n";
        std::cout << "Wrote boundary check to " << bc_name << "\n";
        std::cout << "Boundary check at r = 1 + delta, delta = " << delta << "\n";
        std::cout << "max |u_tot| = " << max_abs_utot << "\n";
        std::cout << "rms |u_tot| = " << rms_abs_utot << "\n";
    }

    if (do_te_sk) {
        // Check Neumann condition near the boundary:
        //   partial_n u_tot ~ 0 on Gamma.
        //
        // Since we currently evaluate only the field potential, we estimate
        // the radial derivative outside the unit circle by a one-sided
        // finite difference:
        //   partial_r u(r ~ 1) ~ (u(1+2d)-u(1+d))/d.
        fout_bc << "theta,real_dnutot,imag_dnutot,abs_dnutot\n";

        Real max_abs_dnu = 0.0;
        Real sum_sq_abs_dnu = 0.0;

        for (int m = 0; m < n_bc; ++m) {
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
            const Cplx usca1 = eval_single_layer_field(sl_pot, mesh, dof, dens, x1);
            const Cplx usca2 = eval_single_layer_field(sl_pot, mesh, dof, dens, x2);

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
        std::cout << "Wrote observations to " << obs_name << "\n";
        std::cout << "Wrote boundary check to " << bc_name << "\n";
        std::cout << "Neumann check by one-sided radial finite difference, delta = " << delta << "\n";
        std::cout << "max |dn u_tot| ~= " << max_abs_dnu << "\n";
        std::cout << "rms |dn u_tot| ~= " << rms_abs_dnu << "\n";
        std::cout << "NOTE: te_sk currently uses K' = K^* at Omega = 0.\n";
    }

    fout_bc.close();
    return 0;
}
