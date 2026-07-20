#include <gtest/gtest.h>

#include "../bemtool/tools.hpp"
#include "../bemtool/operator/rotating_helmholtz_fast_op.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>

using namespace bemtool;

namespace
{
    void expect_complex_close(
        const Cplx& actual,
        const Cplx& expected,
        double atol = 1e-12,
        double rtol = 1e-11)
    {
        const double scale =
            std::max(std::abs(actual), std::abs(expected));
        const double tolerance = atol + rtol * scale;

        EXPECT_LE(std::abs(actual - expected), tolerance)
            << "actual   = " << actual << '\n'
            << "expected = " << expected << '\n'
            << "error    = " << std::abs(actual - expected) << '\n'
            << "tolerance = " << tolerance;
    }

    R3 rotate_2d(const R3& x, double theta)
    {
        const double c = std::cos(theta);
        const double s = std::sin(theta);

        return R3_(
            c * x[0] - s * x[1],
            s * x[0] + c * x[1],
            x[2]
        );
    }

    struct KernelConfiguration
    {
        R3 x;
        R3 y;
        R3 nx;
        R3 ny;
    };

    const std::array<KernelConfiguration, 4> configurations = {{
        {
            R3_(1.2, 0.3, 0.0),
            R3_(-0.4, 0.8, 0.0),
            R3_(0.6, 0.8, 0.0),
            R3_(-0.8, 0.6, 0.0)
        },
        {
            R3_(-1.1, 0.7, 0.0),
            R3_(0.2, -0.9, 0.0),
            R3_(-0.3, 0.9539392014, 0.0),
            R3_(0.8, 0.6, 0.0)
        },
        {
            R3_(0.6, -1.4, 0.0),
            R3_(-0.7, -0.2, 0.0),
            R3_(1.0, 0.0, 0.0),
            R3_(0.0, -1.0, 0.0)
        },
        {
            R3_(1.8, 1.1, 0.0),
            R3_(0.1, -0.5, 0.0),
            R3_(0.7071067812, 0.7071067812, 0.0),
            R3_(-0.7071067812, 0.7071067812, 0.0)
        }
    }};
}

// See Section 2.9.1 and the reversed-rotation Green's function.
// Since x cross y = -(y cross x), one expects
//
//     G_Omega(x,y) = G_{-Omega}(y,x).
//
TEST(RotatingKernelProperties, ReversedRotationReciprocity)
{
    const Real alpha = 2.3;
    const Cplx sigma(0.0, -0.37);

    for (const auto& config : configurations)
    {
        const Cplx lhs = detail_rh_fast::G(
            config.x,
            config.y,
            alpha,
            sigma
        );

        const Cplx rhs = detail_rh_fast::G(
            config.y,
            config.x,
            alpha,
            -sigma
        );

        expect_complex_close(lhs, rhs);
    }
}

// Differentiating
//
//     G_Omega(x,y) = G_{-Omega}(y,x)
//
// exchanges the observation and source derivatives.
TEST(RotatingKernelProperties, ReversedRotationNormalDerivatives)
{
    const Real alpha = 2.3;
    const Cplx sigma(0.0, -0.37);

    for (const auto& config : configurations)
    {
        const Cplx dnx_lhs = detail_rh_fast::dnx_G(
            config.x,
            config.y,
            config.nx,
            alpha,
            sigma
        );

        const Cplx dnx_rhs = detail_rh_fast::dny_G(
            config.y,
            config.x,
            config.nx,
            alpha,
            -sigma
        );

        expect_complex_close(dnx_lhs, dnx_rhs);

        const Cplx dny_lhs = detail_rh_fast::dny_G(
            config.x,
            config.y,
            config.ny,
            alpha,
            sigma
        );

        const Cplx dny_rhs = detail_rh_fast::dnx_G(
            config.y,
            config.x,
            config.ny,
            alpha,
            -sigma
        );

        expect_complex_close(dny_lhs, dny_rhs);

        const Cplx mixed_lhs = detail_rh_fast::dnx_dny_G(
            config.x,
            config.y,
            config.nx,
            config.ny,
            alpha,
            sigma
        );

        const Cplx mixed_rhs = detail_rh_fast::dnx_dny_G(
            config.y,
            config.x,
            config.ny,
            config.nx,
            alpha,
            -sigma
        );

        expect_complex_close(mixed_lhs, mixed_rhs);
    }
}

// Both |x-y| and x cross y are invariant under simultaneous rotations
// of x and y. The normals must be rotated as well.
TEST(RotatingKernelProperties, SimultaneousRotationInvariance)
{
    const Real alpha = 1.8;
    const Cplx sigma(0.0, -0.21);
    const double theta = 0.731;

    for (const auto& config : configurations)
    {
        const R3 rotated_x = rotate_2d(config.x, theta);
        const R3 rotated_y = rotate_2d(config.y, theta);
        const R3 rotated_nx = rotate_2d(config.nx, theta);
        const R3 rotated_ny = rotate_2d(config.ny, theta);

        expect_complex_close(
            detail_rh_fast::G(
                config.x,
                config.y,
                alpha,
                sigma
            ),
            detail_rh_fast::G(
                rotated_x,
                rotated_y,
                alpha,
                sigma
            )
        );

        expect_complex_close(
            detail_rh_fast::dnx_G(
                config.x,
                config.y,
                config.nx,
                alpha,
                sigma
            ),
            detail_rh_fast::dnx_G(
                rotated_x,
                rotated_y,
                rotated_nx,
                alpha,
                sigma
            )
        );

        expect_complex_close(
            detail_rh_fast::dny_G(
                config.x,
                config.y,
                config.ny,
                alpha,
                sigma
            ),
            detail_rh_fast::dny_G(
                rotated_x,
                rotated_y,
                rotated_ny,
                alpha,
                sigma
            )
        );

        expect_complex_close(
            detail_rh_fast::dnx_dny_G(
                config.x,
                config.y,
                config.nx,
                config.ny,
                alpha,
                sigma
            ),
            detail_rh_fast::dnx_dny_G(
                rotated_x,
                rotated_y,
                rotated_nx,
                rotated_ny,
                alpha,
                sigma
            )
        );
    }
}

// For real Omega, sigma is purely imaginary. Therefore,
// |exp(sigma * (x cross y))| = 1 and the phase correction does not
// change the magnitude of the Green's function.
TEST(RotatingKernelProperties, PhaseFactorHasUnitMagnitude)
{
    const Real alpha = 2.7;
    const Cplx sigma(0.0, -0.42);
    const Cplx zero_sigma(0.0, 0.0);

    for (const auto& config : configurations)
    {
        const Cplx rotating = detail_rh_fast::G(
            config.x,
            config.y,
            alpha,
            sigma
        );

        const Cplx stationary = detail_rh_fast::G(
            config.x,
            config.y,
            alpha,
            zero_sigma
        );

        EXPECT_NEAR(
            std::abs(rotating),
            std::abs(stationary),
            1e-12
        );
    }
}