//
// Created by jonas on 4/17/26.
//

#ifndef BEMTOOL_HELPERS_HPP
#define BEMTOOL_HELPERS_HPP

#include <eigen3/Eigen/Dense>
#include "../bemtool/tools.hpp"

namespace rotating_helpers
{
    using namespace bemtool;
    // ============================================================
    // Data containers
    // ============================================================
    struct Sample
    {
        R3 x;

        Cplx uinc = 0.0;
        Cplx usca = 0.0;
        Cplx utr = 0.0;
        Cplx utot = 0.0;
    };

    // ============================================================
    // Incident field
    // ============================================================

    inline Cplx incident_plane_wave(const R3& x, Real kx, Real ky)
    {
        return std::exp(iu * (kx * x[0] + ky * x[1]));
    }

    inline Cplx dirichlet_trace_incident(const R3& x, Real kx, Real ky)
    {
        return incident_plane_wave(x, kx, ky);
    }

    inline Cplx neumann_trace_incident_circle(const R3& x, Real kx, Real ky)
    {
        // outward normal for a circle centered at origin
        const Real r = std::sqrt(x[0] * x[0] + x[1] * x[1]);
        const Real nx = x[0] / r;
        const Real ny = x[1] / r;
        const Cplx uinc = incident_plane_wave(x, kx, ky);
        return iu * (kx * nx + ky * ny) * uinc;
    }

    // ============================================================
    // Interpolation / RHS helpers
    // ============================================================
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
                // should be += and / by number of dofs but only one per element in case of circle for P0 and P1
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
        static constexpr int order = 8;
        static constexpr Real X[8] = {
            -0.9602898564975363, -0.7966664774136267,
            -0.5255324099163290, -0.1834346424956498,
            0.1834346424956498, 0.5255324099163290,
            0.7966664774136267, 0.9602898564975363
        };
        static constexpr Real W[8] = {
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

    template <typename FUNC>
    Eigen::VectorXcd L2ProjectionToP1(FUNC g, const Mesh1D& mesh,
                                      const Dof<P1_1D>& dof_p1, const Eigen::MatrixXcd& M11)
    {
        // L2 projection of Dirichlet data g
        // See Hiptmair ADVNCSE 1.4.2.34

        const Eigen::VectorXcd g_b = L2WithP1Basis(g, mesh, dof_p1);
        const Eigen::VectorXcd g_projected = M11.fullPivLu().solve(g_b);

        return g_projected;
    }

    // ============================================================
    // Assembly helper
    // ============================================================
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
        for (int i = 0; i < nb_row; ++i, bar++)
        {
            for (int j = 0; j < nb_col; ++j)
            {
                A(i, j) += op(test_dof.ToElt(i), trial_dof.ToElt(j));
            }
        }
        bar.end();
        return A;
    }

    // ============================================================
    // Potential evaluation helper
    // ============================================================
    template <typename PotType, typename DofType>
    Cplx eval_potential(Potential<PotType>& pot,
                        const Mesh1D& mesh,
                        const DofType& dof,
                        const Eigen::VectorXcd& dens,
                        const R3& x)
    {
        Cplx u = 0.0;
        const int nb_elt = NbElt(mesh);
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

    // ============================================================
    // Reconstruction formulas
    // ============================================================
    //
    // Thesis formulas:
    //   usca = - Psi_SL^e gamma^c + Psi_DL^e beta^c
    //   utr  =   Psi_SL^i gamma   - Psi_DL^i beta
    //
    // with:
    //   beta^c  = beta - beta_inc
    //   gamma^c = -gamma - gamma_inc
    //
    // See Section 3.7.
    // ============================================================
    template <typename SLPotExt, typename DLPotExt>
    Cplx eval_exterior_scattered_field(Potential<SLPotExt>& sl_pot_ext,
                                       Potential<DLPotExt>& dl_pot_ext,
                                       const Mesh1D& mesh,
                                       const Dof<P0_1D>& dof_p0,
                                       const Dof<P1_1D>& dof_p1,
                                       const Eigen::VectorXcd& beta_p1,
                                       const Eigen::VectorXcd& gamma_p0,
                                       const Eigen::VectorXcd& beta_inc_p1,
                                       const Eigen::VectorXcd& gamma_inc_p0,
                                       const R3& x)
    {
        const Eigen::VectorXcd beta_c = beta_p1 - beta_inc_p1;
        const Eigen::VectorXcd gamma_c = -gamma_p0 - gamma_inc_p0;

        const Cplx sl = eval_potential(sl_pot_ext, mesh, dof_p0, gamma_c, x);
        const Cplx dl = eval_potential(dl_pot_ext, mesh, dof_p1, beta_c, x);

        return -sl + dl;
    }

    template <typename SLPotInt, typename DLPotInt>
    Cplx eval_interior_transmitted_field(Potential<SLPotInt>& sl_pot_int,
                                         Potential<DLPotInt>& dl_pot_int,
                                         const Mesh1D& mesh,
                                         const Dof<P0_1D>& dof_p0,
                                         const Dof<P1_1D>& dof_p1,
                                         const Eigen::VectorXcd& beta_p1,
                                         const Eigen::VectorXcd& gamma_p0,
                                         const R3& x)
    {
        const Cplx sl = eval_potential(sl_pot_int, mesh, dof_p0, gamma_p0, x);
        const Cplx dl = eval_potential(dl_pot_int, mesh, dof_p1, beta_p1, x);

        return sl - dl;
    }

    // ============================================================
    // Sampling
    // ============================================================
    struct InterfaceSolution
    {
        // beta  = Dirichlet trace on Gamma, use P1
        // gamma = Neumann   trace on Gamma, use P0
        Eigen::VectorXcd beta_p1;
        Eigen::VectorXcd gamma_p0;

        Eigen::MatrixXcd A;
        Eigen::VectorXcd rhs;

        std::string name;
    };


    template <typename SLPotExt, typename DLPotExt>
    std::vector<Sample> sample_exterior_total_field_on_circle(
        const Mesh1D& mesh,
        const Dof<P0_1D>& dof_p0,
        const Dof<P1_1D>& dof_p1,
        Potential<SLPotExt>& sl_pot_ext,
        Potential<DLPotExt>& dl_pot_ext,
        const InterfaceSolution& sol,
        const Eigen::VectorXcd& beta_inc_p1,
        const Eigen::VectorXcd& gamma_inc_p0,
        Real radius,
        int n_samples,
        Real kx,
        Real ky)
    {
        std::vector<Sample> samples;
        samples.reserve(n_samples);

        for (int m = 0; m < n_samples; ++m)
        {
            const Real theta = 2.0 * M_PI * static_cast<Real>(m) / static_cast<Real>(n_samples);

            R3 x;
            x[0] = radius * std::cos(theta);
            x[1] = radius * std::sin(theta);
            x[2] = 0.0;

            Sample s;
            s.x = x;
            s.uinc = incident_plane_wave(x, kx, ky);
            s.usca = eval_exterior_scattered_field(sl_pot_ext, dl_pot_ext,
                                                   mesh, dof_p0, dof_p1,
                                                   sol.beta_p1, sol.gamma_p0,
                                                   beta_inc_p1, gamma_inc_p0, x);
            s.utot = s.uinc + s.usca;
            samples.push_back(s);
        }

        return samples;
    }

    template <typename SLPotInt, typename DLPotInt>
    std::vector<Sample> sample_interior_transmitted_field_on_circle(
        const Mesh1D& mesh,
        const Dof<P0_1D>& dof_p0,
        const Dof<P1_1D>& dof_p1,
        Potential<SLPotInt>& sl_pot_int,
        Potential<DLPotInt>& dl_pot_int,
        const InterfaceSolution& sol,
        Real radius,
        int n_samples)
    {
        std::vector<Sample> samples;
        samples.reserve(n_samples);

        for (int m = 0; m < n_samples; ++m)
        {
            const Real theta = 2.0 * M_PI * static_cast<Real>(m) / static_cast<Real>(n_samples);

            R3 x;
            x[0] = radius * std::cos(theta);
            x[1] = radius * std::sin(theta);
            x[2] = 0.0;

            Sample s;
            s.x = x;
            s.utr = eval_interior_transmitted_field(sl_pot_int, dl_pot_int,
                                                    mesh, dof_p0, dof_p1,
                                                    sol.beta_p1, sol.gamma_p0, x);
            s.utot = s.utr;
            samples.push_back(s);
        }

        return samples;
    }

    inline bool point_inside_circle(const R3& x, Real radius = 1.0)
    {
        return (x[0] * x[0] + x[1] * x[1]) < radius * radius;
    }

    inline bool point_too_close_to_circle(const R3& x, Real radius, Real tol)
    {
        const Real r = std::sqrt(x[0] * x[0] + x[1] * x[1]);
        return std::abs(r - radius) < tol;
    }

    // ============================================================
    // CSV helpers
    // ============================================================

    inline void write_samples_csv(const std::string& filename,
                                  const std::vector<Sample>& samples)
    {
        std::ofstream fout(filename);
        fout << "x,y,real_uinc,imag_uinc,real_usca,imag_usca,real_utr,imag_utr,real_utot,imag_utot,abs_utot\n";

        for (const auto& s : samples)
        {
            fout << s.x[0] << ',' << s.x[1] << ','
                << std::real(s.uinc) << ',' << std::imag(s.uinc) << ','
                << std::real(s.usca) << ',' << std::imag(s.usca) << ','
                << std::real(s.utr) << ',' << std::imag(s.utr) << ','
                << std::real(s.utot) << ',' << std::imag(s.utot) << ','
                << std::abs(s.utot) << '\n';
        }
    }
} // namespace

#endif //BEMTOOL_HELPERS_HPP
