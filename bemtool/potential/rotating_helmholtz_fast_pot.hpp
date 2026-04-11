//===================================================================
//  Rotating Helmholtz potentials (FAST approximation, no spectral sum)
//
//  Uses the first-order rotating Green's function approximation
//
//    G(x,y) \approx (i/4) H_0^{(1)}(khat |x-y|) * exp(i * alpha * det(x,y))
//
//  where
//    khat  = k * sqrt(eps*mu)
//    alpha = Omega / k^2
//    det(x,y) = x_y*y_x - x_x*y_y  (i.e. (x \times y)_z)
//
//  DL potential uses BemTool's convention:
//    u(x) = \int_\Gamma (-\partial_{n_y} G(x,y)) \varphi(y) ds_y
//
//  This header is the "fast" analogue of rotating_helmholtz_singrem_real_pot.hpp
//  but with the closed-form approximation (no M-truncation, no c, no keep_D2).
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

    // parameters
    const Real k0;
    const Real eps;
    const Real mu;
    const Real Omega;

    const Real khat;
    const Real alpha;
    const Cplx pref;

    // cached element data
    R3 y0;
    R3 x; // evaluation point
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
      , alpha((k0_ != 0.) ? (Omega_ / (k0_ * k0_)) : 0.)
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

      // stationary Hankel part
      const Real z = khat * R;
      const Cplx Gst = pref * Hankel(0, z);

      // rotating phase
      const Real det = x[1] * y[0] - x[0] * y[1];
      const Cplx phase = std::exp(iu * alpha * det);

      const Cplx G = Gst * phase;
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

    // parameters
    const Real k0;
    const Real eps;
    const Real mu;
    const Real Omega;

    const Real khat;
    const Real alpha;
    const Cplx pref;

    // cached element data
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
      , alpha((k0_ != 0.) ? (Omega_ / (k0_ * k0_)) : 0.)
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

      // stationary Hankel part
      const Real z = khat * R;
      const Cplx H0 = Hankel(0, z);
      const Cplx H1 = Hankel(1, z);
      const Cplx Gst = pref * H0;

      // d/dn_y Gst (derivative w.r.t. source point y)
      // grad_y Gst = (iu/4) * khat * H1(z) * (x-y)/R
      const Real dot = (R > 0.) ? ((rx * ny[0] + ry * ny[1]) / R) : 0.;
      const Cplx dnGst = pref * khat * H1 * dot;

      // rotating phase and its normal derivative factor
      const Real det = x[1] * y[0] - x[0] * y[1];
      const Cplx phase = std::exp(iu * alpha * det);

      // d/dn_y det = (x_y, -x_x) · n_y
      const Real dndet = x[1] * ny[0] - x[0] * ny[1];

      // dnG = phase * (dnGst + i*alpha*dndet*Gst)
      const Cplx dnG = phase * (dnGst + iu * alpha * dndet * Gst);

      // BemTool DL convention: kernel returns -dnG * ds
      ker = -h * dnG;

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
