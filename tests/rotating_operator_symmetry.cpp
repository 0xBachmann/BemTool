#include <gtest/gtest.h>

#include "../bemtool/tools.hpp"
#include "../bemtool/operator/rotating_helmholtz_fast_op.hpp"
#include "../rotation/helpers.hpp"

#include <Eigen/Dense>

#include <algorithm>
#include <complex>
#include <string>

using namespace bemtool;
using namespace rotating_helpers;

namespace
{
    void expect_matrices_close(
        const Eigen::MatrixXcd& actual,
        const Eigen::MatrixXcd& expected,
        double atol = 1e-11,
        double rtol = 1e-10)
    {
        ASSERT_EQ(actual.rows(), expected.rows());
        ASSERT_EQ(actual.cols(), expected.cols());

        const double scale =
            std::max(actual.norm(), expected.norm());
        const double tolerance = atol + rtol * scale;
        const double error = (actual - expected).norm();

        EXPECT_LE(error, tolerance)
            << "||actual||   = " << actual.norm() << '\n'
            << "||expected|| = " << expected.norm() << '\n'
            << "||difference|| = " << error << '\n'
            << "tolerance = " << tolerance;
    }

    struct Parameters
    {
        Real k0 = 2.0;
        Real eps = 2.25;
        Real mu = 1.0;
        Real Omega = 0.17;
    };
}

// For the single-layer operator:
//
//     V_Omega = transpose(V_{-Omega}).
//
TEST(RotatingOperatorSymmetry, SingleLayerUnderReversedRotation)
{
    const Parameters p;
    Geometry geometry("../mesh/carre.msh");
    Mesh1D mesh;
    mesh.Load(geometry);
    Orienting(mesh);

    Dof<P1_1D> dof(mesh);

    BIOp<RH_SL_2D_P1xP1> positive_operator(
        mesh, mesh, p.k0, p.eps, p.mu, p.Omega
    );

    BIOp<RH_SL_2D_P1xP1> negative_operator(
        mesh, mesh, p.k0, p.eps, p.mu, -p.Omega
    );

    const Eigen::MatrixXcd positive = assemble_biop_matrix(
        dof,
        dof,
        positive_operator,
        "V(+Omega)"
    );

    const Eigen::MatrixXcd negative = assemble_biop_matrix(
        dof,
        dof,
        negative_operator,
        "V(-Omega)"
    );

    expect_matrices_close(
        positive,
        negative.transpose()
    );
}

// Swapping source and observation exchanges the double-layer and
// adjoint double-layer kernels. Therefore:
//
//     K_Omega = transpose(K'_{-Omega}).
//
TEST(
    RotatingOperatorSymmetry,
    DoubleLayerAndAdjointUnderReversedRotation)
{
    const Parameters p;
    Geometry geometry("../mesh/carre.msh");
    Mesh1D mesh;
    mesh.Load(geometry);
    Orienting(mesh);

    Dof<P1_1D> dof(mesh);

    BIOp<RH_DL_2D_P1xP1> double_layer_positive(
        mesh, mesh, p.k0, p.eps, p.mu, p.Omega
    );

    BIOp<RH_TDL_2D_P1xP1> adjoint_negative(
        mesh, mesh, p.k0, p.eps, p.mu, -p.Omega
    );

    const Eigen::MatrixXcd K_positive = assemble_biop_matrix(
        dof,
        dof,
        double_layer_positive,
        "K(+Omega)"
    );

    const Eigen::MatrixXcd Kp_negative = assemble_biop_matrix(
        dof,
        dof,
        adjoint_negative,
        "K'(-Omega)"
    );

    expect_matrices_close(
        K_positive,
        Kp_negative.transpose()
    );
}

// The fully regularized weak hypersingular operator should satisfy
//
//     W_Omega = transpose(W_{-Omega}).
//
// This test is particularly important because the existing Helmholtz
// comparison only checks Omega = 0.
TEST(
    RotatingOperatorSymmetry,
    WeakHypersingularP1xP1UnderReversedRotation)
{
    const Parameters p;
    Geometry geometry("../mesh/carre.msh");
    Mesh1D mesh;
    mesh.Load(geometry);
    Orienting(mesh);

    Dof<P1_1D> dof(mesh);

    BIOp<RH_HS_WEAK_2D_P1xP1> positive_operator(
        mesh, mesh, p.k0, p.eps, p.mu, p.Omega
    );

    BIOp<RH_HS_WEAK_2D_P1xP1> negative_operator(
        mesh, mesh, p.k0, p.eps, p.mu, -p.Omega
    );

    const Eigen::MatrixXcd positive = assemble_biop_matrix(
        dof,
        dof,
        positive_operator,
        "W(+Omega), P1xP1"
    );

    const Eigen::MatrixXcd negative = assemble_biop_matrix(
        dof,
        dof,
        negative_operator,
        "W(-Omega), P1xP1"
    );

    expect_matrices_close(
        positive,
        negative.transpose()
    );
}

// This exercises the two partially regularized implementations.
// Transposition exchanges the test and trial spaces:
//
//     W_Omega^{P1xP0}
//         = transpose(W_{-Omega}^{P0xP1}).
//
TEST(
    RotatingOperatorSymmetry,
    WeakHypersingularMixedSpacesUnderReversedRotation)
{
    const Parameters p;
    Geometry geometry("../mesh/carre.msh");
    Mesh1D mesh;
    mesh.Load(geometry);
    Orienting(mesh);


    Dof<P0_1D> dof_p0(mesh);
    Dof<P1_1D> dof_p1(mesh);

    BIOp<RH_HS_WEAK_2D_P1xP0> positive_operator(
        mesh, mesh, p.k0, p.eps, p.mu, p.Omega
    );

    BIOp<RH_HS_WEAK_2D_P0xP1> negative_operator(
        mesh, mesh, p.k0, p.eps, p.mu, -p.Omega
    );

    const Eigen::MatrixXcd positive = assemble_biop_matrix(
        dof_p1,
        dof_p0,
        positive_operator,
        "W(+Omega), P1xP0"
    );

    const Eigen::MatrixXcd negative = assemble_biop_matrix(
        dof_p0,
        dof_p1,
        negative_operator,
        "W(-Omega), P0xP1"
    );

    expect_matrices_close(
        positive,
        negative.transpose()
    );
}