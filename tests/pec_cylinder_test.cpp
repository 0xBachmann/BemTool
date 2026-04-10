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

// Assemble the right-hand side b_i = \int_\Gamma u_inc \phi_i ds
void assemble_rhs_p1(const Mesh1D& mesh, const Dof<P1_1D>& dof,
                     Real kx, Real ky, Eigen::VectorXcd& rhs) {
    static const Real gp[2] = {
        0.5 * (1.0 - 1.0 / std::sqrt(3.0)),
        0.5 * (1.0 + 1.0 / std::sqrt(3.0))
    };
    static const Real gw[2] = {0.5, 0.5}; // mapped from [-1,1] to [0,1]

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
    const Real k0 = 2.0;         // exterior wavenumber
    const Real eps = 1.0;        // exterior relative permittivity
    const Real mu  = 1.0;        // exterior relative permeability
    const Real Omega = 0.0;      // PEC benchmark: stationary exterior problem

    const Real theta_inc = 0.0;  // incidence angle
    const Real kx = k0 * std::cos(theta_inc);
    const Real ky = k0 * std::sin(theta_inc);

    Geometry node("../mesh/circle.msh");
    Mesh1D mesh;
    mesh.Load(node, 1);
    Orienting(mesh);

    Dof<P1_1D> dof(mesh);
    const int nb_dof = NbDof(dof);
    const int nb_elt = NbElt(mesh);

    std::cout << "nb_elt = " << nb_elt << "\n";
    std::cout << "nb_dof = " << nb_dof << "\n";

    // Single-layer Galerkin matrix for the exterior Dirichlet PEC problem:
    //     V phi = gamma_D u_inc
    // because u_sca = -SL(phi) and gamma_D u_tot = 0.
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

    // Solve V phi = gamma_D u_inc
    Eigen::VectorXcd phi = A.fullPivLu().solve(rhs);

    Eigen::VectorXcd res = A * phi - rhs;
    std::cout << "relative residual = " << res.norm() / rhs.norm() << "\n";

    // Evaluate the total field on a circle r = r_obs > 1.
    Potential<RH_SL_2D_P1> sl_pot(mesh, k0, eps, mu, Omega);

    const Real r_obs = 2.0;
    const int n_obs = 200;
    std::ofstream fout("pec_cylinder_obs.csv");
    fout << "theta,x,y,real_uinc,imag_uinc,real_usca,imag_usca,real_utot,imag_utot\n";

    for (int m = 0; m < n_obs; ++m) {
        const Real theta = 2.0 * M_PI * static_cast<Real>(m) / static_cast<Real>(n_obs);
        R3 x;
        x[0] = r_obs * std::cos(theta);
        x[1] = r_obs * std::sin(theta);
        x[2] = 0.0;

        const Cplx uinc = incident_plane_wave(x, kx, ky);

        // u_sca = - SL(phi)
        Cplx usca = 0.0;
        for (int e = 0; e < nb_elt; ++e) {
            const N2& edof = dof[e];
            usca -= sl_pot(x, N2_(e,0)) * phi(edof[0]);
            usca -= sl_pot(x, N2_(e,1)) * phi(edof[1]);
        }

        const Cplx utot = uinc + usca;

        fout << theta << ',' << x[0] << ',' << x[1] << ','
             << std::real(uinc) << ',' << std::imag(uinc) << ','
             << std::real(usca) << ',' << std::imag(usca) << ','
             << std::real(utot) << ',' << std::imag(utot) << '\n';
    }

    std::cout << "Wrote observations to pec_cylinder_obs.csv\n";
    return 0;
}
