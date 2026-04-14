//===================================================================
//  Rotating Helmholtz potentials (FAST approximation, no spectral sum)
//
//  Uses the same first-order rotating Green's function approximation
//  and the same source-normal derivative convention as the operator
//  implementation.
//
//    G(x,y) = pref * H_0^{(1)}(khat |x-y|) * exp(sigma * cross(x,y))
//
//  where
//    khat  = k0 * sqrt(eps*mu)
//    sigma = -i * (Omega * k0 / c0)
//    cross(x,y) = x_x*y_y - x_y*y_x
//
//  The double-layer potential returns +\partial_{n_y} G(x,y), matching
//  the operator kernel convention used in the assembly code.
//====================================================================
#ifndef BEMTOOL_POTENTIAL_ROTATING_HELMHOLTZ_FAST_HPP
#define BEMTOOL_POTENTIAL_ROTATING_HELMHOLTZ_FAST_HPP

#include "potential.hpp"

namespace bemtool
{
  // -----------------------------
  // Single-layer potential kernel
  // -----------------------------
  template <typename PhiY>
  class PotKernel<RH, SL_POT, 2, PhiY>
  {
  public:
    typedef PotKernelTraits<PhiY> Trait;

  private:
    const typename Trait::MeshY& meshy;
    typename Trait::MatType mat;
    typename Trait::JacY dy;
    PhiY phiy;

    const Real k0;
    const Real eps;
    const Real mu;
    const Real Omega;

    const Real khat;
    const Real c0;
    const Cplx sigma;
    const Cplx pref;

    R3 y0;
    R3 x;
    R3 y;
    Real h{};
    Cplx ker;

  public:
    PotKernel<RH, SL_POT, 2, PhiY>(const typename Trait::MeshY& my,
                                  const Real& k0_,
                                  const Real& eps_,
                                  const Real& mu_,
                                  const Real& Omega_ = 0.)
      : meshy(my)
      , phiy(my)
      , k0(k0_)
      , eps(eps_)
      , mu(mu_)
      , Omega(Omega_)
      , khat(k0_ * std::sqrt(eps_ * mu_))
      , c0(1.0)
      , sigma(-iu * (Omega_ * k0_ / c0))
      , pref(detail_rh_fast::kernel_prefactor())
    {}

    void Assign(const R3& x_, const int& iy)
    {
      x = x_;
      const typename Trait::EltY& ey = meshy[iy];
      y0 = ey[0];
      dy = MatJac(ey);
      h = DetJac(ey);
    }

    const typename Trait::MatType& operator()(const R3& /*x_unused*/, const typename Trait::Rdy& tj)
    {
      y = y0 + dy * tj;

      const Real dx = x[0] - y[0];
      const Real dyv = x[1] - y[1];
      const Real R = std::hypot(dx, dyv);

      const Real z = khat * R;
      const Cplx H0 = Hankel(0, z);
      const Real cross = x[0] * y[1] - x[1] * y[0];
      const Cplx phase = std::exp(sigma * cross);

      const Cplx G = pref * H0 * phase;
      ker = h * G;

      for (int k = 0; k < Trait::nb_dof_y; ++k)
        mat(0, k) = ker * phiy(k, tj);
      return mat;
    }

    const Cplx& operator()(const R3& /*x_unused*/, const typename Trait::Rdy& tj, const int& ky)
    {
      operator()(R3{}, tj);
      static Cplx val;
      val = ker * phiy(ky, tj);
      return val;
    }
  };

  using RH_SL_2D_P0 = PotKernel<RH, SL_POT, 2, P0_1D>;
  using RH_SL_2D_P1 = PotKernel<RH, SL_POT, 2, P1_1D>;


  // -----------------------------
  // Double-layer potential kernel
  // -----------------------------
  template <typename PhiY>
  class PotKernel<RH, DL_POT, 2, PhiY>
  {
  public:
    typedef PotKernelTraits<PhiY> Trait;

  private:
    const typename Trait::MeshY& meshy;
    const std::vector<R3>& normaly;

    typename Trait::MatType mat;
    typename Trait::JacY dy;
    PhiY phiy;

    const Real k0;
    const Real eps;
    const Real mu;
    const Real Omega;

    const Real khat;
    const Real c0;
    const Cplx sigma;
    const Cplx pref;

    R3 y0;
    R3 x;
    R3 y;
    R3 ny;
    Real h{};
    Cplx ker;

  public:
    PotKernel<RH, DL_POT, 2, PhiY>(const typename Trait::MeshY& my,
                                  const Real& k0_,
                                  const Real& eps_,
                                  const Real& mu_,
                                  const Real& Omega_ = 0.)
      : meshy(my)
      , normaly(NormalTo(my))
      , phiy(my)
      , k0(k0_)
      , eps(eps_)
      , mu(mu_)
      , Omega(Omega_)
      , khat(k0_ * std::sqrt(eps_ * mu_))
      , c0(1.0)
      , sigma(-iu * (Omega_ * k0_ / c0))
      , pref(detail_rh_fast::kernel_prefactor())
    {}

    void Assign(const R3& x_, const int& iy)
    {
      x = x_;
      const typename Trait::EltY& ey = meshy[iy];
      y0 = ey[0];
      dy = MatJac(ey);
      h = DetJac(ey);
      ny = normaly[iy];
    }

    const typename Trait::MatType& operator()(const R3& /*x_unused*/, const typename Trait::Rdy& tj)
    {
      y = y0 + dy * tj;

      const Real rx = x[0] - y[0];
      const Real ry = x[1] - y[1];
      const Real R = std::hypot(rx, ry);

      const Real z = khat * R;
      const Cplx H0 = Hankel(0, z);
      const Cplx H1 = Hankel(1, z);

      const Real cross = x[0] * y[1] - x[1] * y[0];
      const Cplx phase = std::exp(sigma * cross);

      const Real s_src = (R > 0.) ? ((rx * ny[0] + ry * ny[1]) / R) : 0.0;
      const Real q_src = x[1] * ny[0] - x[0] * ny[1];

      const Cplx dnG = pref * phase * (khat * H1 * s_src + sigma * H0 * q_src);

      ker = h * dnG;

      for (int k = 0; k < Trait::nb_dof_y; ++k)
        mat(0, k) = ker * phiy(k, tj);
      return mat;
    }

    const Cplx& operator()(const R3& /*x_unused*/, const typename Trait::Rdy& tj, const int& ky)
    {
      operator()(R3{}, tj);
      static Cplx val;
      val = ker * phiy(ky, tj);
      return val;
    }
  };

  using RH_DL_2D_P0 = PotKernel<RH, DL_POT, 2, P0_1D>;
  using RH_DL_2D_P1 = PotKernel<RH, DL_POT, 2, P1_1D>;

}

#endif
