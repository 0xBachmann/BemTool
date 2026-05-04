#include <gtest/gtest.h>

#include "../bemtool/tools.hpp"
#include "../rotation/helpers.hpp"

#include <Eigen/Dense>
#include <complex>
#include <string>
#include <iostream>

using namespace bemtool;
using namespace rotating_helpers;

namespace
{
    void expect_matrices_close(const Eigen::MatrixXcd& A,
                               const Eigen::MatrixXcd& B,
                               double atol,
                               double rtol)
    {
        ASSERT_EQ(A.rows(), B.rows());
        ASSERT_EQ(A.cols(), B.cols());

        const double scale = std::max(A.norm(), B.norm());
        const double tol = atol + rtol * scale;
        const double err = (A - B).norm();

        EXPECT_LE(err, tol) << "matrix mismatch: ||A-B|| = " << err
                        << ", tol = " << tol;
    }

    std::vector<R3> cartesian_grid_2d(Real xmin, Real xmax,
                                      Real ymin, Real ymax,
                                      int nx, int ny)
    {
        std::vector<R3> pts;
        pts.reserve(nx * ny);

        for (int j = 0; j < ny; ++j) {
            const Real y = ymin + (ymax - ymin) * static_cast<Real>(j) / static_cast<Real>(ny - 1);
            for (int i = 0; i < nx; ++i) {
                const Real x = xmin + (xmax - xmin) * static_cast<Real>(i) / static_cast<Real>(nx - 1);
                R3 p;
                p[0] = x;
                p[1] = y;
                p[2] = 0.0;
                pts.push_back(p);
            }
        }
        return pts;
    }

    void expect_vectors_close(const Eigen::VectorXcd& a,
                              const Eigen::VectorXcd& b,
                              double atol,
                              double rtol)
    {
        ASSERT_EQ(a.size(), b.size());

        const double scale = std::max(a.norm(), b.norm());
        const double tol = atol + rtol * scale;
        const double err = (a - b).norm();

        EXPECT_LE(err, tol) << "vector mismatch: ||a-b|| = " << err
                            << ", tol = " << tol;
    }

    void report_matrix_difference(const Eigen::MatrixXcd& A,
                              const Eigen::MatrixXcd& B)
    {
        Eigen::MatrixXcd D = A - B;
        Eigen::Index imax = 0, jmax = 0;
        double maxabs = 0.0;

        for (Eigen::Index i = 0; i < D.rows(); ++i) {
            for (Eigen::Index j = 0; j < D.cols(); ++j) {
                double a = std::abs(D(i,j));
                if (a > maxabs) {
                    maxabs = a;
                    imax = i;
                    jmax = j;
                }
            }
        }

        std::cout << "||A|| = " << A.norm() << "\n";
        std::cout << "||B|| = " << B.norm() << "\n";
        std::cout << "||A-B|| = " << D.norm() << "\n";
        std::cout << "max |A-B| = " << maxabs
                  << " at (" << imax << ", " << jmax << ")\n";
        std::cout << "A(i,j) = " << A(imax,jmax) << "\n";
        std::cout << "B(i,j) = " << B(imax,jmax) << "\n";
    }
} // namespace

TEST(MeshCircle, LoadMesh)
{
    Geometry node("../mesh/circle.msh");
    Mesh1D mesh;
    mesh.Load(node, 1);
    EXPECT_GT(NbElt(mesh), 0);
}

TEST(RotatingFastVsHelmholtz, SingleLayerMatchesAtZeroOmega)
{
    const Real k0 = 2.0;
    const Real eps = 1.0;
    const Real mu = 1.0;
    const Real Omega = 0.0;

    Geometry node("../mesh/circle.msh");
    Mesh1D mesh;
    mesh.Load(node, 1);
    Orienting(mesh);
    Dof<P0_1D> dof0(mesh);
    Dof<P1_1D> dof1(mesh);

    BIOp<HE_SL_2D_P1xP0> H(mesh, mesh, k0);
    BIOp<RH_SL_2D_P1xP0> R(mesh, mesh, k0, eps, mu, Omega);

    const Eigen::MatrixXcd A_he = assemble_biop_matrix(dof1, dof0, H, "helmholtz");
    const Eigen::MatrixXcd A_rh = assemble_biop_matrix(dof1, dof0, R, "rotating helmholtz");

    expect_matrices_close(A_he, -A_rh, 1e-11, 1e-11); // Due to sign conventions, A_he = -A_rh
}

TEST(RotatingFastVsHelmholtz, DoubleLayerMatchesAtZeroOmega)
{
    const Real k0 = 2.0;
    const Real eps = 1.0;
    const Real mu = 1.0;
    const Real Omega = 0.0;

    Geometry node("../mesh/circle.msh");
    Mesh1D mesh;
    mesh.Load(node, 1);
    Orienting(mesh);
    Dof<P1_1D> dof(mesh);

    BIOp<HE_DL_2D_P1xP1> H(mesh, mesh, k0);
    BIOp<RH_DL_2D_P1xP1> R(mesh, mesh, k0, eps, mu, Omega);

    const Eigen::MatrixXcd A_he = assemble_biop_matrix(dof, dof, H, "helmholtz");
    const Eigen::MatrixXcd A_rh = assemble_biop_matrix(dof, dof, R, "rotating helmholtz");

    report_matrix_difference(A_he, A_rh);

    expect_matrices_close(A_he, A_rh, 1e-11, 1e-11); // Due to sign conventions, A_he = A_rh
}

TEST(RotatingFastVsHelmholtz, AdjointDoubleLayerMatchesAtZeroOmega)
{
    const Real k0 = 2.0;
    const Real eps = 1.0;
    const Real mu = 1.0;
    const Real Omega = 0.0;

    Geometry node("../mesh/circle.msh");
    Mesh1D mesh;
    mesh.Load(node, 1);
    Orienting(mesh);
    Dof<P1_1D> dof(mesh);

    BIOp<HE_TDL_2D_P0xP0> H(mesh, mesh, k0);
    BIOp<RH_TDL_2D_P0xP0> R(mesh, mesh, k0, eps, mu, Omega);

    const Eigen::MatrixXcd A_he = assemble_biop_matrix(dof, dof, H, "helmholtz");
    const Eigen::MatrixXcd A_rh = assemble_biop_matrix(dof, dof, R, "rotating helmholtz");

    report_matrix_difference(A_he, A_rh);

    expect_matrices_close(A_he, -A_rh, 1e-11, 1e-11); // Due to sign conventions, A_he = -A_rh
}

TEST(RotatingFastVsHelmholtz, HypersingularLayerMatchesAtZeroOmega)
{
    const Real k0 = 2.0;
    const Real eps = 1.0;
    const Real mu = 1.0;
    const Real Omega = 0.0;

    Geometry node("../mesh/circle.msh");
    Mesh1D mesh;
    mesh.Load(node, 1);
    Orienting(mesh);
    Dof<P1_1D> dof(mesh);


    BIOp<HE_HS_2D_P1xP1> H(mesh, mesh, k0);
    BIOp<RH_HS_2D_P1xP1> R(mesh, mesh, k0, eps, mu, Omega);

    const Eigen::MatrixXcd A_he = assemble_biop_matrix(dof, dof, H, "helmholtz");
    const Eigen::MatrixXcd A_rh = assemble_biop_matrix(dof, dof, R, "rotating helmholtz");

    report_matrix_difference(A_he, A_rh);

    expect_matrices_close(A_he, -A_rh, 1e-11, 1e-11); // Due to sign conventions, A_he = -A_rh
}

TEST(RotatingFastVsHelmholtz, SLPotentialMatchesAtZeroOmegaOnGrid)
{
    const Real k0    = 2.0;
    const Real eps   = 1.0;
    const Real mu    = 1.0;
    const Real Omega = 0.0;

    Geometry node("../mesh/circle.msh");
    Mesh1D mesh;
    mesh.Load(node, 1);
    Orienting(mesh);
    Dof<P0_1D> dof(mesh);

    const int nb_dof = NbDof(dof);

    // deterministic, nontrivial density
    Eigen::VectorXcd dens(nb_dof);
    for (int i = 0; i < nb_dof; ++i) {
        dens(i) = Cplx(std::cos(0.17 * i), std::sin(0.11 * i));
    }

    Potential<HE_SL_2D_P0> he_pot(mesh, k0);
    Potential<RH_SL_2D_P0> rh_pot(mesh, k0, eps, mu, Omega);

    const auto pts = cartesian_grid_2d(-2.0, 2.0, -2.0, 2.0, 21, 21);

    Eigen::VectorXcd u_he(pts.size());
    Eigen::VectorXcd u_rh(pts.size());

    for (int i = 0; i < static_cast<int>(pts.size()); ++i) {
        u_he(i) = eval_potential(he_pot, mesh, dof, dens, pts[i]);
        u_rh(i) = eval_potential(rh_pot, mesh, dof, dens, pts[i]);
    }

    expect_vectors_close(u_he, -u_rh, 1e-11, 1e-11); // same sign convention as the matrix tests
}

TEST(RotatingFastVsHelmholtz, DLPotentialMatchesAtZeroOmegaOnGrid)
{
    const Real k0    = 2.0;
    const Real eps   = 1.0;
    const Real mu    = 1.0;
    const Real Omega = 0.0;

    Geometry node("../mesh/circle.msh");
    Mesh1D mesh;
    mesh.Load(node, 1);
    Orienting(mesh);
    Dof<P1_1D> dof(mesh);


    const int nb_dof = NbDof(dof);

    Eigen::VectorXcd dens(nb_dof);
    for (int i = 0; i < nb_dof; ++i) {
        dens(i) = Cplx(std::cos(0.13 * i), -std::sin(0.07 * i));
    }

    Potential<HE_DL_2D_P1> he_pot(mesh, k0);
    Potential<RH_DL_2D_P1> rh_pot(mesh, k0, eps, mu, Omega);

    const auto pts = cartesian_grid_2d(-2.0, 2.0, -2.0, 2.0, 21, 21);

    Eigen::VectorXcd u_he(pts.size());
    Eigen::VectorXcd u_rh(pts.size());

    for (int i = 0; i < static_cast<int>(pts.size()); ++i) {
        u_he(i) = eval_potential(he_pot, mesh, dof, dens, pts[i]);
        u_rh(i) = eval_potential(rh_pot, mesh, dof, dens, pts[i]);
    }

    expect_vectors_close(u_he, u_rh, 1e-11, 1e-11); // same sign convention as the matrix tests
}
