
#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <fstream>

#include "../bemtool/tools.hpp"

using namespace bemtool;

namespace {

Cplx incident_plane_wave(const R3& x, Real kx, Real ky) {
    return std::exp(iu * (kx * x[0] + ky * x[1]));
}

// Assemble the Galerkin right-hand side
//     rhs_i = \int_\Gamma u_inc * phi_i ds
// for the Dirichlet equation
//     V phi = gamma_D u_inc .
void assemble_rhs_p1(const Mesh1D& mesh, const Dof<P1_1D>& dof,
                     Real kx, Real ky, Eigen::VectorXcd& rhs) {
    static const Real gp[2] = {
        0.5 * (1.0 - 1.0 / std::sqrt(3.0)),
        0.5 * (1.0 + 1.0 / std::sqrt(3.0))
    };
    static constexpr Real gw[2] = {0.5, 0.5};

    const int nb_elt = NbElt(mesh);
    for (int j = 0; j < nb_elt; ++j) {
        const auto& elt = mesh[j];
        const R3 a = elt[0];
        const R3 b = elt[1];
        const R3 tau = b - a;
        const Real h = std::sqrt(tau[0]*tau[0] + tau[1]*tau[1] + tau[2]*tau[2]);

        const N2& jdof = dof[j];

        for (int q = 0; q < 2; ++q) {
            const Real t = gp[q];
            const R3 x = a + t * tau;
            const Cplx uinc = incident_plane_wave(x, kx, ky);
            const Real phi0 = 1.0 - t;
            const Real phi1 = t;

            rhs(jdof[0]) += gw[q] * h * uinc * phi0;
            rhs(jdof[1]) += gw[q] * h * uinc * phi1;
        }
    }
}

} // namespace

int main(int argc, char* argv[]) {
    // Exterior parameters
    const Real k0 = 2.0;
    const Real eps = 1.0;
    const Real mu  = 1.0;

    // Stationary PEC benchmark
    const Real Omega = 0.0;

    // Incident plane wave
    const Real theta_inc = 0.0;
    const Real kx = k0 * std::cos(theta_inc);
    const Real ky = k0 * std::sin(theta_inc);

    // Geometry
    Geometry node("../mesh/circle.msh");
    Mesh1D mesh;
    mesh.Load(node, 1);
    Orienting(mesh);

    Dof<P1_1D> dof(mesh);
    const int nb_dof = NbDof(dof);
    const int nb_elt = NbElt(mesh);

    std::cout << "nb_elt = " << nb_elt << "\n";
    std::cout << "nb_dof = " << nb_dof << "\n";

    // Dirichlet PEC boundary-integral equation:
    //
    //   u_tot = u_inc + u_sca,
    //   gamma_D u_tot = 0 on Gamma,
    //   u_sca = - SL(phi),
    //
    // hence
    //
    //   V phi = gamma_D u_inc.
    //
    // This is the single-layer Dirichlet formulation corresponding
    // to equation (3.94) in the thesis draft, not the double-layer
    // equation (3.97) and not the Neumann/TE equation (3.115).
    BIOp<RH_SL_2D_P1xP1> V(mesh, mesh, k0, eps, mu, Omega);

    Eigen::MatrixXcd A = Eigen::MatrixXcd::Zero(nb_dof, nb_dof);
    Eigen::VectorXcd rhs = Eigen::VectorXcd::Zero(nb_dof);

    progress bar("Assembling V", nb_dof);
    for (int i = 0; i < nb_dof; ++i) {
        bar++;
        for (int j = 0; j < nb_dof; ++j) {
            A(i, j) += V(dof.ToElt(i), dof.ToElt(j));
        }
    }
    bar.end();

    assemble_rhs_p1(mesh, dof, kx, ky, rhs);

    Eigen::VectorXcd phi = A.fullPivLu().solve(rhs);

    Eigen::VectorXcd res = A * phi - rhs;
    std::cout << "relative residual = " << res.norm() / rhs.norm() << "\n";

    Potential<RH_SL_2D_P1> sl_pot(mesh, k0, eps, mu, Omega);

    // Exterior observation circle
    const Real r_obs = 2.0;
    const int n_obs = 200;
    std::ofstream fout_obs("pec_cylinder_te_sl_obs.csv");
    fout_obs << "theta,x,y,real_uinc,imag_uinc,real_usca,imag_usca,real_utot,imag_utot\n";

    for (int m = 0; m < n_obs; ++m) {
        const Real theta = 2.0 * M_PI * static_cast<Real>(m) / static_cast<Real>(n_obs);
        R3 x;
        x[0] = r_obs * std::cos(theta);
        x[1] = r_obs * std::sin(theta);
        x[2] = 0.0;

        const Cplx uinc = incident_plane_wave(x, kx, ky);

        Cplx usca = 0.0;
        for (int e = 0; e < nb_elt; ++e) {
            const N2& edof = dof[e];
            usca -= sl_pot(x, N2_(e,0)) * phi(edof[0]);
            usca -= sl_pot(x, N2_(e,1)) * phi(edof[1]);
        }

        const Cplx utot = uinc + usca;

        fout_obs << theta << ',' << x[0] << ',' << x[1] << ','
                 << std::real(uinc) << ',' << std::imag(uinc) << ','
                 << std::real(usca) << ',' << std::imag(usca) << ','
                 << std::real(utot) << ',' << std::imag(utot) << '\n';
    }
    fout_obs.close();

    // Boundary-condition check on a slightly offset exterior circle
    const Real delta = 1.0e-3;
    const Real r_bc = 1.0 + delta;
    const int n_bc = 400;
    std::ofstream fout_bc("pec_cylinder_te_sl_boundary_check.csv");
    fout_bc << "theta,x,y,real_utot,imag_utot,abs_utot\n";

    Real max_abs_utot = 0.0;
    Real sum_sq_abs_utot = 0.0;

    for (int m = 0; m < n_bc; ++m) {
        const Real theta = 2.0 * M_PI * static_cast<Real>(m) / static_cast<Real>(n_bc);
        R3 x;
        x[0] = r_bc * std::cos(theta);
        x[1] = r_bc * std::sin(theta);
        x[2] = 0.0;

        const Cplx uinc = incident_plane_wave(x, kx, ky);

        Cplx usca = 0.0;
        for (int e = 0; e < nb_elt; ++e) {
            const N2& edof = dof[e];
            usca -= sl_pot(x, N2_(e,0)) * phi(edof[0]);
            usca -= sl_pot(x, N2_(e,1)) * phi(edof[1]);
        }

        const Cplx utot = uinc + usca;
        const Real abs_utot = std::abs(utot);

        if (abs_utot > max_abs_utot) {
            max_abs_utot = abs_utot;
        }
        sum_sq_abs_utot += abs_utot * abs_utot;

        fout_bc << theta << ',' << x[0] << ',' << x[1] << ','
                << std::real(utot) << ',' << std::imag(utot) << ','
                << abs_utot << '\n';
    }
    fout_bc.close();

    const Real rms_abs_utot = std::sqrt(sum_sq_abs_utot / static_cast<Real>(n_bc));

    std::cout << "Wrote observations to pec_cylinder_te_sl_obs.csv\n";
    std::cout << "Wrote boundary check to pec_cylinder_te_sl_boundary_check.csv\n";
    std::cout << "Boundary check at r = 1 + delta, delta = " << delta << "\n";
    std::cout << "max |u_tot| = " << max_abs_utot << "\n";
    std::cout << "rms |u_tot| = " << rms_abs_utot << "\n";

    return 0;
}
