#ifndef BEMTOOL_POTENTIAL_ROTATING_HELMHOLTZ_FAST_HPP
#define BEMTOOL_POTENTIAL_ROTATING_HELMHOLTZ_FAST_HPP

#include "potential.hpp"

namespace bemtool
{
  namespace detail_rh_fast_pot
  {
    inline Real det2_xy(const R3& x, const R3& y)
    {
      return x[1] * y[0] - x[0] * y[1];
    }
  }

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

    const Real k0, eps, mu, Omega, khat, alpha;
    const Cplx pref;

    R3 y0, x, y;
    Real h{};
    Cplx ker;

  public:
    PotKernel(const typename Trait::MeshY& my,
              const Real& k0_,
              const Real& eps_,
              const Real& mu_,
              const Real& Omega_ = 0.)
      : meshy(my), phiy(my),
        k0(k0_), eps(eps_), mu(mu_), Omega(Omega_),
        khat(k0_ * std::sqrt(eps_ * mu_)),
        alpha((k0_ != 0.) ? (Omega_ / (k0_ * k0_)) : 0.),
        pref(iu / static_cast<Real>(4))
    {}

    void Assign(const R3& x_, const int& iy)
    {
      x = x_;
      const typename Trait::EltY& ey = meshy[iy];
      y0 = ey[0];
      dy = MatJac(ey);
      h = DetJac(ey);
    }

    const typename Trait::MatType& operator()(const R3&, const typename Trait::Rdy& tj)
    {
      y = y0 + dy * tj;

      const Real rx = x[0] - y[0];
      const Real ry = x[1] - y[1];
      const Real R = std::hypot(rx, ry);

      const Cplx Gst = pref * Hankel(0, khat * R);

      // reversed kernel \hat G
      const Real det = detail_rh_fast_pot::det2_xy(x, y);
      const Cplx phase = std::exp(-iu * alpha * det);

      ker = h * Gst * phase;

      for (int k = 0; k < Trait::nb_dof_y; ++k)
        mat(0, k) = ker * phiy(k, tj);
      return mat;
    }

    const Cplx& operator()(const R3&, const typename Trait::Rdy& tj, const int& ky)
    {
      operator()(R3{}, tj);
      static Cplx val;
      val = ker * phiy(ky, tj);
      return val;
    }
  };

  using RH_SL_2D_P0 = PotKernel<RH, SL_POT, 2, P0_1D>;
  using RH_SL_2D_P1 = PotKernel<RH, SL_POT, 2, P1_1D>;

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

    const Real k0, eps, mu, Omega, khat, alpha;
    const Cplx pref;

    R3 y0, x, y, ny;
    Real h{};
    Cplx ker;

  public:
    PotKernel(const typename Trait::MeshY& my,
              const Real& k0_,
              const Real& eps_,
              const Real& mu_,
              const Real& Omega_ = 0.)
      : meshy(my), normaly(NormalTo(my)), phiy(my),
        k0(k0_), eps(eps_), mu(mu_), Omega(Omega_),
        khat(k0_ * std::sqrt(eps_ * mu_)),
        alpha((k0_ != 0.) ? (Omega_ / (k0_ * k0_)) : 0.),
        pref(iu / static_cast<Real>(4))
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

    const typename Trait::MatType& operator()(const R3&, const typename Trait::Rdy& tj)
    {
      y = y0 + dy * tj;

      const Real rx = x[0] - y[0];
      const Real ry = x[1] - y[1];
      const Real R = std::hypot(rx, ry);

      constexpr Real epsR = 1e-14;
      Real dRdn_y = 0.0;
      if (R > epsR)
        dRdn_y = -(rx * ny[0] + ry * ny[1]) / R;

      const Real z = khat * R;
      const Cplx Gst = pref * Hankel(0, z);
      const Cplx dGst_dn_y = pref * (khat * DHankel_Dx(0, z)) * dRdn_y;

      const Real det = detail_rh_fast_pot::det2_xy(x, y);
      const Cplx phase = std::exp(-iu * alpha * det);

      const Real ddet_dn_y = x[1] * ny[0] - x[0] * ny[1];
      const Cplx dGhat_dn_y = phase * (dGst_dn_y - iu * alpha * ddet_dn_y * Gst);

      // Thesis convention: Ψ_DL uses +∂_{n_y} \hat G in the source variable.
      ker = h * dGhat_dn_y;

      for (int k = 0; k < Trait::nb_dof_y; ++k)
        mat(0, k) = ker * phiy(k, tj);
      return mat;
    }

    const Cplx& operator()(const R3&, const typename Trait::Rdy& tj, const int& ky)
    {
      operator()(R3{}, tj);
      static Cplx val;
      val = ker * phiy(ky, tj);
      return val;
    }
  };

  using RH_DL_2D_P0 = PotKernel<RH, DL_POT, 2, P0_1D>;
  using RH_DL_2D_P1 = PotKernel<RH, DL_POT, 2, P1_1D>;

} // namespace bemtool

#endif
