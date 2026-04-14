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

#include "../bemtool/tools.hpp"

using namespace bemtool;

namespace
{
    // ============================================================
    // Data containers
    // ============================================================

    struct InterfaceSolution
    {
        // beta  = Dirichlet trace on Gamma, use P1
        // gamma = Neumann   trace on Gamma, use P0
        Eigen::VectorXcd beta_p1;
        Eigen::VectorXcd gamma_p0;

        Eigen::MatrixXcd A;
        Eigen::VectorXcd rhs;

        Real relative_bie_residual = 0.0;
        bool available = false;
        std::string name;
    };

    struct FieldSample
    {
        Real theta = 0.0;
        R3 x;

        Cplx uinc = 0.0;
        Cplx usca = 0.0; // only meaningful outside
        Cplx utr = 0.0; // only meaningful inside
        Cplx utot = 0.0; // outside: uinc + usca, inside: utr
    };

    struct GridSample
    {
        R3 x;

        Cplx uinc = 0.0;
        Cplx usca = 0.0;
        Cplx utr = 0.0;
        Cplx utot = 0.0;
    };

    struct BoundaryCheckSample
    {
        Real theta = 0.0;
        R3 x;

        Cplx uext_tot = 0.0;
        Cplx uint_tr = 0.0;
        Cplx jump_dir = 0.0;

        Cplx dne_ext_tot = 0.0;
        Cplx dni_int_tr = 0.0;
        Cplx jump_neu = 0.0; // should be ext + int (sign convention from thesis)
    };

    // ============================================================
    // Incident field
    // ============================================================

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
        // outward normal for a circle centered at origin
        const Real r = std::sqrt(x[0] * x[0] + x[1] * x[1]);
        const Real nx = x[0] / r;
        const Real ny = x[1] / r;
        const Cplx uinc = incident_plane_wave(x, kx, ky);
        return iu * (kx * nx + ky * ny) * uinc;
    }

    bool point_inside_circle(const R3& x, Real radius = 1.0)
    {
        return (x[0] * x[0] + x[1] * x[1]) < radius * radius;
    }

    bool point_too_close_to_circle(const R3& x, Real radius, Real tol)
    {
        const Real r = std::sqrt(x[0] * x[0] + x[1] * x[1]);
        return std::abs(r - radius) < tol;
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
            0.1834346424956498, 0.5255324099163290,
            0.7966664774136267, 0.9602898564975363
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

    template <typename FUNC>
    Eigen::VectorXcd L2ProjectionToP1(FUNC g, const Mesh1D& mesh,
                                      const Dof<P1_1D>& dof_p1, const Eigen::MatrixXcd& M11)
    {
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
        const Eigen::VectorXcd g_projected = M11.fullPivLu().solve(g_b);

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

    template <typename SLPotExt, typename DLPotExt>
    std::vector<FieldSample> sample_exterior_total_field_on_circle(
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
    std::vector<FieldSample> sample_interior_transmitted_field_on_circle(
        const Mesh1D& mesh,
        const Dof<P0_1D>& dof_p0,
        const Dof<P1_1D>& dof_p1,
        Potential<SLPotInt>& sl_pot_int,
        Potential<DLPotInt>& dl_pot_int,
        const InterfaceSolution& sol,
        Real radius,
        int n_samples)
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
            s.utr = eval_interior_transmitted_field(sl_pot_int, dl_pot_int,
                                                    mesh, dof_p0, dof_p1,
                                                    sol.beta_p1, sol.gamma_p0, x);
            s.utot = s.utr;
            samples.push_back(s);
        }

        return samples;
    }

    template <typename SLPotExt, typename DLPotExt, typename SLPotInt, typename DLPotInt>
    std::vector<GridSample> sample_total_field_on_grid(
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
        std::vector<GridSample> samples;
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

                GridSample s;
                s.x = x;

                bool is_skipped = false;

                if (point_too_close_to_circle(x, interface_radius, skip_tolerance))
                {
                    is_skipped = true;
                    samples.push_back(s);
                    continue;
                }

                bool is_inside = point_inside_circle(x, interface_radius);

                if (is_inside)
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

    // ============================================================
    // CSV helpers
    // ============================================================

    void write_field_samples_csv(const std::string& filename,
                                 const std::vector<FieldSample>& samples)
    {
        std::ofstream fout(filename);
        fout << "theta,x,y,real_uinc,imag_uinc,real_usca,imag_usca,real_utr,imag_utr,real_utot,imag_utot,abs_utot\n";

        for (const auto& s : samples)
        {
            fout << s.theta << ','
                << s.x[0] << ',' << s.x[1] << ','
                << std::real(s.uinc) << ',' << std::imag(s.uinc) << ','
                << std::real(s.usca) << ',' << std::imag(s.usca) << ','
                << std::real(s.utr) << ',' << std::imag(s.utr) << ','
                << std::real(s.utot) << ',' << std::imag(s.utot) << ','
                << std::abs(s.utot) << '\n';
        }
    }

    void write_grid_samples_csv(const std::string& filename,
                                const std::vector<GridSample>& samples)
    {
        std::ofstream fout(filename);
        fout <<
            "x,y,is_inside,is_skipped,real_uinc,imag_uinc,real_usca,imag_usca,real_utr,imag_utr,real_utot,imag_utot,abs_utot\n";

        for (const auto& s : samples)
        {
            fout << s.x[0] << ','
                << s.x[1] << ','
                << std::real(s.uinc) << ',' << std::imag(s.uinc) << ','
                << std::real(s.usca) << ',' << std::imag(s.usca) << ','
                << std::real(s.utr) << ',' << std::imag(s.utr) << ','
                << std::real(s.utot) << ',' << std::imag(s.utot) << ','
                << std::abs(s.utot) << '\n';
        }
        std::cout << "wrote to " << filename << "\n";
    }

    void print_field_stats(const std::string& label,
                           const std::vector<FieldSample>& samples,
                           Real radius)
    {
        Real max_abs = 0.0;
        Real sum_sq = 0.0;

        for (const auto& s : samples)
        {
            const Real a = std::abs(s.utot);
            max_abs = std::max(max_abs, a);
            sum_sq += a * a;
        }

        const Real rms = std::sqrt(sum_sq / std::max<int>(1, samples.size()));

        std::cout << label << " on circle r = " << radius << "\n";
        std::cout << "max |u| = " << max_abs << "\n";
        std::cout << "rms |u| = " << rms << "\n";
    }

    void print_grid_field_stats(const std::string& label,
                                const std::vector<GridSample>& samples)
    {
        Real max_abs = 0.0;
        Real sum_sq = 0.0;
        int count = 0;

        for (const auto& s : samples)
        {
            const Real a = std::abs(s.utot);
            max_abs = std::max(max_abs, a);
            sum_sq += a * a;
            ++count;
        }

        const Real rms = std::sqrt(sum_sq / std::max(1, count));

        std::cout << label << " on Cartesian grid\n";
        std::cout << "evaluated points = " << count << "\n";
        std::cout << "max |u| = " << max_abs << "\n";
        std::cout << "rms |u| = " << rms << "\n";
    }

    // ============================================================
    // Boundary transmission checks
    // ============================================================
    //
    // For a circle benchmark you can directly compare:
    //   uext_tot - uint_tr
    //   dn_ext_tot + dn_int_tr
    //
    // Here we first provide the trace-level check, which is cheaper and
    // should be nearly machine precision if assembly/signs are right.
    // ============================================================

    void print_trace_jump_stats(const Eigen::VectorXcd& beta_p1,
                                const Eigen::VectorXcd& gamma_p0,
                                const Eigen::VectorXcd& beta_inc_p1,
                                const Eigen::VectorXcd& gamma_inc_p0)
    {
        // Exterior traces from transmission conditions:
        // beta^c  = beta - beta_inc
        // gamma^c = -gamma - gamma_inc
        const Eigen::VectorXcd beta_c = beta_p1 - beta_inc_p1;
        const Eigen::VectorXcd gamma_c = -gamma_p0 - gamma_inc_p0;

        // check:
        // beta_c + beta_inc - beta = 0
        // gamma_c + gamma_inc + gamma = 0
        const Eigen::VectorXcd dir_jump = beta_c + beta_inc_p1 - beta_p1;
        const Eigen::VectorXcd neu_jump = gamma_c + gamma_inc_p0 + gamma_p0;

        std::cout << "Trace transmission check\n";
        std::cout << "max |beta_c + beta_inc - beta| = " << dir_jump.cwiseAbs().maxCoeff() << "\n";
        std::cout << "max |gamma_c + gamma_inc + gamma| = " << neu_jump.cwiseAbs().maxCoeff() << "\n";
    }

    // ============================================================
    // Optional comparison helper
    // ============================================================


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

    void compare_scalar_fields(const std::string& label,
                               const std::vector<FieldSample>& a,
                               const std::vector<FieldSample>& b,
                               const std::string& filename)
    {
        std::ofstream fout(filename);
        fout << "theta,real_a,imag_a,real_b,imag_b,abs_diff\n";

        Real max_abs_diff = 0.0;
        Real sum_sq_abs_diff = 0.0;
        Real sum_sq_abs_a = 0.0;

        const int n = static_cast<int>(std::min(a.size(), b.size()));
        for (int i = 0; i < n; ++i)
        {
            const Cplx diff = a[i].utot - b[i].utot;
            const Real ad = std::abs(diff);
            const Real aa = std::abs(a[i].utot);

            max_abs_diff = std::max(max_abs_diff, ad);
            sum_sq_abs_diff += ad * ad;
            sum_sq_abs_a += aa * aa;

            fout << a[i].theta << ','
                << std::real(a[i].utot) << ',' << std::imag(a[i].utot) << ','
                << std::real(b[i].utot) << ',' << std::imag(b[i].utot) << ','
                << ad << '\n';
        }

        const Real rms_abs_diff = std::sqrt(sum_sq_abs_diff / std::max(1, n));
        const Real rel_rms_diff = std::sqrt(sum_sq_abs_diff / std::max(sum_sq_abs_a, Real(1.0e-30)));

        std::cout << label << "\n";
        std::cout << "max abs diff = " << max_abs_diff << "\n";
        std::cout << "rms abs diff = " << rms_abs_diff << "\n";
        std::cout << "relative rms diff = " << rel_rms_diff << "\n";
    }
} // namespace

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
    constexpr Real r_obs_ext = 2.0;
    constexpr Real r_obs_int = 0.5;
    constexpr int n_obs = 720;
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
    sol.available = true;

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

    std::cout << "solving system\n";
    const Eigen::VectorXcd x = sol.A.fullPivLu().solve(sol.rhs);
    split_block_solution(x, nb_p1, sol.beta_p1, sol.gamma_p0);
    std::cout << "done\n";

    const Eigen::VectorXcd res = sol.A * x - sol.rhs;
    sol.relative_bie_residual =
        res.norm() / std::max(sol.rhs.norm(), Real(1.0e-30));

    std::cout << "relative BIE residual = "
        << sol.relative_bie_residual << "\n";
    //
    // print_trace_jump_stats(sol.beta_p1, sol.gamma_p0,
    //                        beta_inc_p1, gamma_inc_p0);

    // auto exterior_circle_samples = sample_exterior_total_field_on_circle(
    //     mesh,
    //     dof_p0,
    //     dof_p1,
    //     sl_pot_ext_p0,
    //     dl_pot_ext_p1,
    //     sol,
    //     beta_inc_p1,
    //     gamma_inc_p0,
    //     r_obs_ext,
    //     n_obs,
    //     kx,
    //     ky);
    //
    // auto interior_circle_samples = sample_interior_transmitted_field_on_circle(
    //     mesh,
    //     dof_p0,
    //     dof_p1,
    //     sl_pot_int_p0,
    //     dl_pot_int_p1,
    //     sol,
    //     r_obs_int,
    //     n_obs);

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

    // print_field_stats("Exterior total field", exterior_circle_samples, r_obs_ext);
    // print_field_stats("Interior transmitted field", interior_circle_samples, r_obs_int);
    // print_grid_field_stats("Total field", grid_samples);

    // write_field_samples_csv(out_dir + "/dielectric_ext_circle" + postfix + ".csv", exterior_circle_samples);
    // write_field_samples_csv(out_dir + "/dielectric_int_circle" + postfix + ".csv", interior_circle_samples);
    write_grid_samples_csv(out_dir + "/dielectric_grid" + postfix + ".csv", grid_samples);


    return 0;
}

#endif
