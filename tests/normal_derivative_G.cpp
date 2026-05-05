#include <gtest/gtest.h>

#include "../bemtool/tools.hpp"
#include "../bemtool/operator/rotating_helmholtz_fast_op.hpp"

#include <complex>
#include <cmath>
#include <vector>

using namespace bemtool;

namespace
{
    R3 add_scaled(const R3& x, const R3& n, double h)
    {
        R3 z;
        z[0] = x[0] + h * n[0];
        z[1] = x[1] + h * n[1];
        z[2] = x[2] + h * n[2];
        return z;
    }

    double euclid_norm(const R3& x)
    {
        return std::sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2]);
    }

    R3 outward_normal_on_circle(const R3& y)
    {
        const double r = euclid_norm(y);
        return R3_(y[0] / r, y[1] / r, 0);
    }

    bool same_point(const R3& a, const R3& b, double tol = 1e-12)
    {
        return std::abs(a[0] - b[0]) <= tol
            && std::abs(a[1] - b[1]) <= tol
            && std::abs(a[2] - b[2]) <= tol;
    }

    std::vector<R3> get_cartesian_points(unsigned nx = 100, unsigned ny = 100)
    {
        std::vector<R3> points;
        const double minx = -2, maxx = 2;
        const double lx = maxx - minx;
        const double miny = -2, maxy = 2;
        const double ly = maxy - miny;
        const double dlx = lx / nx, dly = ly / ny;

        for (int ix = 0; ix <= nx; ++ix)
        {
            for (int iy = 0; iy <= ny; ++iy)
            {
                auto point = R3_(minx + ix * dlx, miny + iy * dly, 0);
                if (same_point(point, R3_(0., 0., 0.))) { continue; }
                points.push_back(point);
            }
        }

        return points;
    }

    std::vector<R3> get_circle_points(unsigned n = 100)
    {
        std::vector<R3> points;
        const double dtheta = M_PI / n;

        for (unsigned ix = 0; ix < n; ++ix)
        {
            points.push_back(R3_(std::cos(dtheta * ix), std::sin(dtheta * ix), 0.));
        }

        return points;
    }

    bool singular_pair(const R3& x, const R3& y, double tol = 1e-12)
    {
        return euclid_norm(R3_(x[0] - y[0], x[1] - y[1], 0)) <= tol;
    }


    void expect_close_complex(Cplx a, Cplx b, double atol, double rtol)
    {
        const double scale = std::max(std::abs(a), std::abs(b));
        const double tol = atol + rtol * scale;
        EXPECT_LE(std::abs(a - b), tol)
        << "a = " << a << ", b = " << b
        << ", |a-b| = " << std::abs(a - b)
        << ", tol = " << tol;
    }

    Real alpha_he(Real k0)
    {
        return k0;
    }

    Cplx sigma_he()
    {
        return {0.0, 0.0};
    }

    Real alpha_rh(Real k0, Real eps, Real mu)
    {
        return k0 * std::sqrt(eps * mu);
    }

    // Adapt this if your class uses a different convention for sigma.
    Cplx sigma_rh(Real k0, Real eps, Real mu, Real Omega)
    {
        constexpr Real c0 = 299792458.0;
        const Real n = std::sqrt(eps * mu);
        return Cplx(0.0, 1.0) * (Omega * alpha_rh(k0, eps, mu) / (n * c0));
    }

    Cplx G_he(const R3& x, const R3& y, Real k0)
    {
        return detail_rh_fast::G(x, y, alpha_he(k0), sigma_he());
    }

    Cplx dnx_G_he(const R3& x, const R3& y, const R3& nx, Real k0)
    {
        return detail_rh_fast::dnx_G(x, y, nx, alpha_he(k0), sigma_he());
    }

    Cplx dny_G_he(const R3& x, const R3& y, const R3& ny, Real k0)
    {
        return detail_rh_fast::dny_G(x, y, ny, alpha_he(k0), sigma_he());
    }

    Cplx dnx_dny_G_he(const R3& x, const R3& y, const R3& nx, const R3& ny, Real k0)
    {
        return detail_rh_fast::dnx_dny_G(x, y, nx, ny, alpha_he(k0), sigma_he());
    }

    Cplx G_rh(const R3& x, const R3& y, Real k0, Real eps, Real mu, Real Omega)
    {
        return detail_rh_fast::G(x, y, alpha_rh(k0, eps, mu), sigma_rh(k0, eps, mu, Omega));
    }

    Cplx dnx_G_rh(const R3& x, const R3& y, const R3& nx,
                  Real k0, Real eps, Real mu, Real Omega)
    {
        return detail_rh_fast::dnx_G(x, y, nx,
                                     alpha_rh(k0, eps, mu),
                                     sigma_rh(k0, eps, mu, Omega));
    }

    Cplx dny_G_rh(const R3& x, const R3& y, const R3& ny,
                  Real k0, Real eps, Real mu, Real Omega)
    {
        return detail_rh_fast::dny_G(x, y, ny,
                                     alpha_rh(k0, eps, mu),
                                     sigma_rh(k0, eps, mu, Omega));
    }

    Cplx dnx_dny_G_rh(const R3& x, const R3& y, const R3& nx, const R3& ny,
                      Real k0, Real eps, Real mu, Real Omega)
    {
        return detail_rh_fast::dnx_dny_G(x, y, nx, ny,
                                         alpha_rh(k0, eps, mu),
                                         sigma_rh(k0, eps, mu, Omega));
    }

    Cplx fd_dnx_from_G_he(const R3& x, const R3& y, const R3& nx, Real k0, double h)
    {
        const R3 xp = add_scaled(x, nx, +h);
        const R3 xm = add_scaled(x, nx, -h);
        return (G_he(xp, y, k0) - G_he(xm, y, k0)) / (2.0 * h);
    }

    Cplx fd_dny_from_G_he(const R3& x, const R3& y, const R3& ny, Real k0, double h)
    {
        const R3 yp = add_scaled(y, ny, +h);
        const R3 ym = add_scaled(y, ny, -h);
        return (G_he(x, yp, k0) - G_he(x, ym, k0)) / (2.0 * h);
    }

    Cplx fd_dnx_from_G_rh(const R3& x, const R3& y, const R3& nx,
                          Real k0, Real eps, Real mu, Real Omega, double h)
    {
        const R3 xp = add_scaled(x, nx, +h);
        const R3 xm = add_scaled(x, nx, -h);
        return (G_rh(xp, y, k0, eps, mu, Omega) - G_rh(xm, y, k0, eps, mu, Omega)) / (2.0 * h);
    }

    Cplx fd_dny_from_G_rh(const R3& x, const R3& y, const R3& ny,
                          Real k0, Real eps, Real mu, Real Omega, double h)
    {
        const R3 yp = add_scaled(y, ny, +h);
        const R3 ym = add_scaled(y, ny, -h);
        return (G_rh(x, yp, k0, eps, mu, Omega) - G_rh(x, ym, k0, eps, mu, Omega)) / (2.0 * h);
    }

    Cplx fd_dnx_dny_from_G_he(const R3& x, const R3& y, const R3& nx, const R3& ny,
                              Real k0, double h)
    {
        const R3 xp = add_scaled(x, nx, +h);
        const R3 xm = add_scaled(x, nx, -h);
        return (dny_G_he(xp, y, ny, k0) - dny_G_he(xm, y, ny, k0)) / (2.0 * h);
    }

    Cplx fd_dnx_dny_from_G_rh(const R3& x, const R3& y, const R3& nx, const R3& ny,
                              Real k0, Real eps, Real mu, Real Omega, double h)
    {
        const R3 xp = add_scaled(x, nx, +h);
        const R3 xm = add_scaled(x, nx, -h);
        return (dny_G_rh(xp, y, ny, k0, eps, mu, Omega)
            - dny_G_rh(xm, y, ny, k0, eps, mu, Omega)) / (2.0 * h);
    }
} // namespace

TEST(KernelFD, HelmholtzObservationNormalDerivativeMatchesFiniteDifference)
{
    const double k0 = 2.0;
    const double h = 1e-6;
    const auto observation_points = get_circle_points();
    const auto source_points = get_cartesian_points();

    for (const auto& x : observation_points)
    {
        const R3 nx = outward_normal_on_circle(x);
        for (const auto& y : source_points)
        {
            if (singular_pair(x, y))
            {
                continue;
            }
            SCOPED_TRACE(::testing::Message()
                << "x = (" << x[0] << ", " << x[1] << ")"
                << ", y = (" << y[0] << ", " << y[1] << ")");

            const Cplx exact = dnx_G_he(x, y, nx, k0);
            const Cplx fd = fd_dnx_from_G_he(x, y, nx, k0, h);

            expect_close_complex(exact, fd, 1e-6, 1e-5);
        }
    }
}

TEST(KernelFD, HelmholtzSourceNormalDerivativeMatchesFiniteDifference)
{
    const double k0 = 2.0;
    const double h = 1e-6;
    const auto observation_points = get_cartesian_points();
    const auto source_points = get_circle_points();

    for (const auto& x : observation_points)
    {
        for (const auto& y : source_points)
        {
            if (singular_pair(x, y))
            {
                continue;
            }
            const R3 ny = outward_normal_on_circle(y);
            SCOPED_TRACE(::testing::Message()
                << "x = (" << x[0] << ", " << x[1] << ")"
                << ", y = (" << y[0] << ", " << y[1] << ")");

            const Cplx exact = dny_G_he(x, y, ny, k0);
            const Cplx fd = fd_dny_from_G_he(x, y, ny, k0, h);

            expect_close_complex(exact, fd, 1e-6, 1e-5);
        }
    }
}

TEST(KernelFD, HelmholtzMixedNormalDerivativeMatchesFiniteDifference)
{
    const double k0 = 2.0;
    const double h = 1e-6;
    const auto observation_points = get_circle_points();
    const auto source_points = get_circle_points();

    for (const auto& x : observation_points)
    {
        const R3 nx = outward_normal_on_circle(x);
        for (const auto& y : source_points)
        {
            if (singular_pair(x, y))
            {
                continue;
            }
            const R3 ny = outward_normal_on_circle(y);
            SCOPED_TRACE(::testing::Message()
                << "x = (" << x[0] << ", " << x[1] << ")"
                << ", y = (" << y[0] << ", " << y[1] << ")");

            const Cplx exact = dnx_dny_G_he(x, y, nx, ny, k0);
            const Cplx fd = fd_dnx_dny_from_G_he(x, y, nx, ny, k0, h);

            expect_close_complex(exact, fd, 1e-5, 1e-4);
        }
    }
}

TEST(KernelFD, RotatingObservationNormalDerivativeMatchesFiniteDifference)
{
    const double k0 = 2.0;
    const double eps = 2.3;
    const double mu = 1.0;
    const double Omega = 1e-3;
    const double h = 1e-6;
    const auto observation_points = get_circle_points();
    const auto source_points = get_cartesian_points();

    for (const auto& x : observation_points)
    {
        const R3 nx = outward_normal_on_circle(x);
        for (const auto& y : source_points)
        {
            if (singular_pair(x, y))
            {
                continue;
            }
            SCOPED_TRACE(::testing::Message()
                << "x = (" << x[0] << ", " << x[1] << ")"
                << ", y = (" << y[0] << ", " << y[1] << ")");

            const Cplx exact = dnx_G_rh(x, y, nx, k0, eps, mu, Omega);
            const Cplx fd = fd_dnx_from_G_rh(x, y, nx, k0, eps, mu, Omega, h);

            expect_close_complex(exact, fd, 1e-6, 1e-5);
        }
    }
}

TEST(KernelFD, RotatingSourceNormalDerivativeMatchesFiniteDifference)
{
    const double k0 = 2.0;
    const double eps = 2.3;
    const double mu = 1.0;
    const double Omega = 1e-3;
    const double h = 1e-6;
    const auto observation_points = get_cartesian_points();
    const auto source_points = get_circle_points();

    for (const auto& x : observation_points)
    {
        for (const auto& y : source_points)
        {
            if (singular_pair(x, y))
            {
                continue;
            }
            const R3 ny = outward_normal_on_circle(y);
            SCOPED_TRACE(::testing::Message()
                << "x = (" << x[0] << ", " << x[1] << ")"
                << ", y = (" << y[0] << ", " << y[1] << ")");

            const Cplx exact = dny_G_rh(x, y, ny, k0, eps, mu, Omega);
            const Cplx fd = fd_dny_from_G_rh(x, y, ny, k0, eps, mu, Omega, h);

            expect_close_complex(exact, fd, 1e-6, 1e-5);
        }
    }
}

TEST(KernelFD, RotatingMixedNormalDerivativeMatchesFiniteDifference)
{
    const double k0 = 2.0;
    const double eps = 2.3;
    const double mu = 1.0;
    const double Omega = 1e-3;
    const double h = 1e-6;
    const auto observation_points = get_circle_points();
    const auto source_points = get_circle_points();

    for (const auto& x : observation_points)
    {
        const R3 nx = outward_normal_on_circle(x);
        for (const auto& y : source_points)
        {
            if (singular_pair(x, y))
            {
                continue;
            }
            const R3 ny = outward_normal_on_circle(y);
            SCOPED_TRACE(::testing::Message()
                << "x = (" << x[0] << ", " << x[1] << ")"
                << ", y = (" << y[0] << ", " << y[1] << ")");

            const Cplx exact = dnx_dny_G_rh(x, y, nx, ny, k0, eps, mu, Omega);
            const Cplx fd = fd_dnx_dny_from_G_rh(x, y, nx, ny, k0, eps, mu, Omega, h);

            expect_close_complex(exact, fd, 1e-5, 1e-4);
        }
    }
}

TEST(KernelFD, RotatingKernelMatchesHelmholtzAtZeroOmegaPointwise)
{
    const double k0 = 2.0;
    const double eps = 1.0;
    const double mu = 1.0;
    const double Omega = 0.0;
    const auto cartesian_points = get_cartesian_points(50, 50);
    const auto circle_points = get_circle_points();

    for (const auto& x : cartesian_points)
    {
        for (const auto& y : cartesian_points)
        {
            if (singular_pair(x, y))
            {
                continue;
            }
            SCOPED_TRACE(::testing::Message()
                << "G: x = (" << x[0] << ", " << x[1] << ")"
                << ", y = (" << y[0] << ", " << y[1] << ")");

            expect_close_complex(
                G_he(x, y, k0),
                G_rh(x, y, k0, eps, mu, Omega),
                1e-12, 1e-12
            );
        }
    }

    for (const auto& x : circle_points)
    {
        const R3 nx = outward_normal_on_circle(x);
        for (const auto& y : cartesian_points)
        {
            if (singular_pair(x, y))
            {
                continue;
            }
            SCOPED_TRACE(::testing::Message()
                << "dnx: x = (" << x[0] << ", " << x[1] << ")"
                << ", y = (" << y[0] << ", " << y[1] << ")");

            expect_close_complex(
                dnx_G_he(x, y, nx, k0),
                dnx_G_rh(x, y, nx, k0, eps, mu, Omega),
                1e-11, 1e-11
            );
        }
    }

    for (const auto& x : cartesian_points)
    {
        for (const auto& y : circle_points)
        {
            if (singular_pair(x, y))
            {
                continue;
            }
            const R3 ny = outward_normal_on_circle(y);
            SCOPED_TRACE(::testing::Message()
                << "dny: x = (" << x[0] << ", " << x[1] << ")"
                << ", y = (" << y[0] << ", " << y[1] << ")");

            expect_close_complex(
                dny_G_he(x, y, ny, k0),
                dny_G_rh(x, y, ny, k0, eps, mu, Omega),
                1e-11, 1e-11
            );
        }
    }

    for (const auto& x : circle_points)
    {
        const R3 nx = outward_normal_on_circle(x);
        for (const auto& y : circle_points)
        {
            if (singular_pair(x, y))
            {
                continue;
            }
            const R3 ny = outward_normal_on_circle(y);
            SCOPED_TRACE(::testing::Message()
                << "dnx_dny: x = (" << x[0] << ", " << x[1] << ")"
                << ", y = (" << y[0] << ", " << y[1] << ")");

            expect_close_complex(
                dnx_dny_G_he(x, y, nx, ny, k0),
                dnx_dny_G_rh(x, y, nx, ny, k0, eps, mu, Omega),
                1e-10, 1e-10
            );
        }
    }
}
