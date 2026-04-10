#ifndef BEMTOOL_OPERATOR_ROTATING_HELMHOLTZ_HPP
#define BEMTOOL_OPERATOR_ROTATING_HELMHOLTZ_HPP

#include "operator.hpp"
#include <complex>
#include <cmath>

namespace bemtool
{
  namespace detail_rh_fast
  {
    inline Real det2_xy(const R3& x, const R3& y)
    {
      // det(x,y) = x_y y_x - x_x y_y
      return x[1] * y[0] - x[0] * y[1];
    }
  }

  // ============================================================
  // SL operator with reversed kernel \hat G
  // ============================================================
  template <typename PhiX, typename PhiY>
  class BIOpKernel<RH, SL_OP, 2, PhiX, PhiY>
  {
  public:
    typedef BIOpKernelTraits<RH, SL_OP, 2, PhiX, PhiY> Trait;

  private:
    const typename Trait::MeshX& meshx;
    const typename Trait::MeshY& meshy;
    typename Trait::MatType inter;
    typename Trait::JacX dx;
    typename Trait::JacY dy;
    PhiX phix;
    PhiY phiy;

    const Real k0, eps, mu, Omega, khat, alpha;
    const Cplx prefactor;

    R3 x0, y0, x, y;
    Real h{};
    Cplx ker;

  public:
    BIOpKernel(const typename Trait::MeshX& mx,
               const typename Trait::MeshY& my,
               const Real& k0_,
               const Real& eps_,
               const Real& mu_,
               const Real& Omega_ = 0.)
      : meshx(mx), meshy(my), phix(mx), phiy(my),
        k0(k0_), eps(eps_), mu(mu_), Omega(Omega_),
        khat(k0_ * std::sqrt(eps_ * mu_)),
        alpha((k0_ != 0.) ? (Omega_ / (k0_ * k0_)) : 0.),
        prefactor(iu / static_cast<Real>(4))
    {}

    void Assign(const int& ix, const int& iy)
    {
      const typename Trait::EltX& ex = meshx[ix];
      const typename Trait::EltY& ey = meshy[iy];
      x0 = ex[0];
      y0 = ey[0];
      dx = MatJac(ex);
      dy = MatJac(ey);
      h = DetJac(ex) * DetJac(ey);
    }

    const typename Trait::MatType& operator()(const typename Trait::Rdx& tx,
                                              const typename Trait::Rdy& ty)
    {
      x = x0 + dx * tx;
      y = y0 + dy * ty;

      const Real rx = x[0] - y[0];
      const Real ry = x[1] - y[1];
      const Real R = std::hypot(rx, ry);

      const Cplx Gst = prefactor * Hankel(0, khat * R);

      // reversed kernel \hat G = G_st * exp(-i alpha det)
      const Real det = detail_rh_fast::det2_xy(x, y);
      const Cplx phase = std::exp(-iu * alpha * det);

      ker = h * Gst * phase;

      for (int j = 0; j < Trait::nb_dof_x; ++j)
        for (int k = 0; k < Trait::nb_dof_y; ++k)
          inter(j, k) = ker * phix(j, tx) * phiy(k, ty);

      return inter;
    }

    const Cplx& operator()(const typename Trait::Rdx& tx,
                           const typename Trait::Rdy& ty,
                           const int& kx, const int& ky)
    {
      operator()(tx, ty);
      static Cplx val;
      val = ker * phix(kx, tx) * phiy(ky, ty);
      return val;
    }
  };

  using RH_SL_2D_P0xP0 = BIOpKernel<RH, SL_OP, 2, P0_1D, P0_1D>;
  using RH_SL_2D_P1xP1 = BIOpKernel<RH, SL_OP, 2, P1_1D, P1_1D>;
  using RH_SL_2D_P2xP2 = BIOpKernel<RH, SL_OP, 2, P2_1D, P2_1D>;
  using RH_SL_2D_P0xP1 = BIOpKernel<RH, SL_OP, 2, P0_1D, P1_1D>;
  using RH_SL_2D_P1xP0 = BIOpKernel<RH, SL_OP, 2, P1_1D, P0_1D>;

  // ============================================================
  // DL operator with reversed kernel \hat G
  // kernel convention: - \partial_{n_y} \hat G
  // ============================================================
  template <typename PhiX, typename PhiY>
  class BIOpKernel<RH, DL_OP, 2, PhiX, PhiY>
  {
  public:
    typedef BIOpKernelTraits<RH, DL_OP, 2, PhiX, PhiY> Trait;

  private:
    const typename Trait::MeshX& meshx;
    const typename Trait::MeshY& meshy;
    const std::vector<R3>& normaly;
    typename Trait::MatType inter;
    typename Trait::JacX dx;
    typename Trait::JacY dy;
    PhiX phix;
    PhiY phiy;

    const Real k0, eps, mu, Omega, khat, alpha;
    const Cplx prefactor;

    R3 x0, y0, x, y, ny;
    Real h{};
    Cplx ker;

  public:
    BIOpKernel(const typename Trait::MeshX& mx,
               const typename Trait::MeshY& my,
               const Real& k0_,
               const Real& eps_,
               const Real& mu_,
               const Real& Omega_ = 0.)
      : meshx(mx), meshy(my), normaly(NormalTo(my)),
        phix(mx), phiy(my),
        k0(k0_), eps(eps_), mu(mu_), Omega(Omega_),
        khat(k0_ * std::sqrt(eps_ * mu_)),
        alpha((k0_ != 0.) ? (Omega_ / (k0_ * k0_)) : 0.),
        prefactor(iu / static_cast<Real>(4))
    {}

    void Assign(const int& ix, const int& iy)
    {
      const typename Trait::EltX& ex = meshx[ix];
      const typename Trait::EltY& ey = meshy[iy];
      x0 = ex[0];
      y0 = ey[0];
      dx = MatJac(ex);
      dy = MatJac(ey);
      h = DetJac(ex) * DetJac(ey);
      ny = normaly[iy];
    }

    const typename Trait::MatType& operator()(const typename Trait::Rdx& tx,
                                              const typename Trait::Rdy& ty)
    {
      x = x0 + dx * tx;
      y = y0 + dy * ty;

      const Real rx = x[0] - y[0];
      const Real ry = x[1] - y[1];
      const Real R  = std::hypot(rx, ry);

      constexpr Real epsR = 1e-14;
      Real dRdn_y = 0.0;
      if (R > epsR)
        dRdn_y = -(rx * ny[0] + ry * ny[1]) / R;

      const Real z = khat * R;
      const Cplx Gst = prefactor * Hankel(0, z);
      const Cplx dGst_dn_y = prefactor * (khat * DHankel_Dx(0, z)) * dRdn_y;

      const Real det = detail_rh_fast::det2_xy(x, y);
      const Cplx phase = std::exp(-iu * alpha * det);

      // d/dn_y det = (x_y, -x_x) · n_y
      const Real ddet_dn_y = x[1] * ny[0] - x[0] * ny[1];

      const Cplx dGhat_dn_y = phase * (dGst_dn_y - (iu * alpha * ddet_dn_y) * Gst);

      ker = -h * dGhat_dn_y;

      for (int j = 0; j < Trait::nb_dof_x; ++j)
        for (int k = 0; k < Trait::nb_dof_y; ++k)
          inter(j, k) = ker * phix(j, tx) * phiy(k, ty);

      return inter;
    }

    const Cplx& operator()(const typename Trait::Rdx& tx,
                           const typename Trait::Rdy& ty,
                           const int& kx, const int& ky)
    {
      operator()(tx, ty);
      static Cplx val;
      val = ker * phix(kx, tx) * phiy(ky, ty);
      return val;
    }
  };

  using RH_DL_2D_P0xP0 = BIOpKernel<RH, DL_OP, 2, P0_1D, P0_1D>;
  using RH_DL_2D_P1xP1 = BIOpKernel<RH, DL_OP, 2, P1_1D, P1_1D>;
  using RH_DL_2D_P2xP2 = BIOpKernel<RH, DL_OP, 2, P2_1D, P2_1D>;
  using RH_DL_2D_P0xP1 = BIOpKernel<RH, DL_OP, 2, P0_1D, P1_1D>;
  using RH_DL_2D_P1xP0 = BIOpKernel<RH, DL_OP, 2, P1_1D, P0_1D>;

  // ============================================================
  // TDL operator with reversed kernel \hat G
  // kernel convention: + \partial_{n_x} \hat G
  // ============================================================
  template <typename PhiX, typename PhiY>
  class BIOpKernel<RH, TDL_OP, 2, PhiX, PhiY>
  {
  public:
    typedef BIOpKernelTraits<RH, TDL_OP, 2, PhiX, PhiY> Trait;

  private:
    const typename Trait::MeshX& meshx;
    const typename Trait::MeshY& meshy;
    const std::vector<R3>& normalx;
    typename Trait::MatType inter;
    typename Trait::JacX dx;
    typename Trait::JacY dy;
    PhiX phix;
    PhiY phiy;

    const Real k0, eps, mu, Omega, khat, alpha;
    const Cplx prefactor;

    R3 x0, y0, x, y, nx;
    Real h{};
    Cplx ker;

  public:
    BIOpKernel(const typename Trait::MeshX& mx,
               const typename Trait::MeshY& my,
               const Real& k0_,
               const Real& eps_,
               const Real& mu_,
               const Real& Omega_ = 0.)
      : meshx(mx), meshy(my), normalx(NormalTo(mx)),
        phix(mx), phiy(my),
        k0(k0_), eps(eps_), mu(mu_), Omega(Omega_),
        khat(k0_ * std::sqrt(eps_ * mu_)),
        alpha((k0_ != 0.) ? (Omega_ / (k0_ * k0_)) : 0.),
        prefactor(iu / static_cast<Real>(4))
    {}

    void Assign(const int& ix, const int& iy)
    {
      const typename Trait::EltX& ex = meshx[ix];
      const typename Trait::EltY& ey = meshy[iy];
      x0 = ex[0];
      y0 = ey[0];
      dx = MatJac(ex);
      dy = MatJac(ey);
      h = DetJac(ex) * DetJac(ey);
      nx = normalx[ix];
    }

    const typename Trait::MatType& operator()(const typename Trait::Rdx& tx,
                                              const typename Trait::Rdy& ty)
    {
      x = x0 + dx * tx;
      y = y0 + dy * ty;

      const Real rx = x[0] - y[0];
      const Real ry = x[1] - y[1];
      const Real R  = std::hypot(rx, ry);

      constexpr Real epsR = 1e-14;
      Real dRdn_x = 0.0;
      if (R > epsR)
        dRdn_x = (rx * nx[0] + ry * nx[1]) / R;

      const Real z = khat * R;
      const Cplx Gst = prefactor * Hankel(0, z);
      const Cplx dGst_dn_x = prefactor * (khat * DHankel_Dx(0, z)) * dRdn_x;

      const Real det = detail_rh_fast::det2_xy(x, y);
      const Cplx phase = std::exp(-iu * alpha * det);

      // grad_x det = (-y_y, y_x)
      const Real ddet_dn_x = (-y[1]) * nx[0] + y[0] * nx[1];

      const Cplx dGhat_dn_x = phase * (dGst_dn_x - (iu * alpha * ddet_dn_x) * Gst);

      ker = h * dGhat_dn_x;

      for (int j = 0; j < Trait::nb_dof_x; ++j)
        for (int k = 0; k < Trait::nb_dof_y; ++k)
          inter(j, k) = ker * phix(j, tx) * phiy(k, ty);

      return inter;
    }

    const Cplx& operator()(const typename Trait::Rdx& tx,
                           const typename Trait::Rdy& ty,
                           const int& kx, const int& ky)
    {
      operator()(tx, ty);
      static Cplx val;
      val = ker * phix(kx, tx) * phiy(ky, ty);
      return val;
    }
  };

  using RH_TDL_2D_P0xP0 = BIOpKernel<RH, TDL_OP, 2, P0_1D, P0_1D>;
  using RH_TDL_2D_P1xP1 = BIOpKernel<RH, TDL_OP, 2, P1_1D, P1_1D>;
  using RH_TDL_2D_P2xP2 = BIOpKernel<RH, TDL_OP, 2, P2_1D, P2_1D>;
  using RH_TDL_2D_P0xP1 = BIOpKernel<RH, TDL_OP, 2, P0_1D, P1_1D>;
  using RH_TDL_2D_P1xP0 = BIOpKernel<RH, TDL_OP, 2, P1_1D, P0_1D>;

  // ============================================================
  // HS operator in 2D, mirroring Helmholtz structure but with \hat G
  // ============================================================
  template <typename PhiX, typename PhiY>
  class BIOpKernel<RH, HS_OP, 2, PhiX, PhiY>
  {
  public:
    typedef BIOpKernelTraits<RH, HS_OP, 2, PhiX, PhiY> Trait;

  private:
    const typename Trait::MeshX& meshx;
    const typename Trait::MeshY& meshy;
    typename Trait::MatType inter;
    typename Trait::JacX dx;
    typename Trait::JacY dy;
    typename Trait::GradPhiX grad_phix;
    typename Trait::GradPhiY grad_phiy;
    PhiX phix;
    PhiY phiy;
    const std::vector<R3>& normalx;
    const std::vector<R3>& normaly;

    const Real k0, eps, mu, Omega, khat, khat2, alpha;
    const Cplx prefactor;

    R3 x0_y0, x_y, x, y, nx, ny;
    Real h{}, r{};
    Cplx ker, val, val2;

  public:
    BIOpKernel(const typename Trait::MeshX& mx,
               const typename Trait::MeshY& my,
               const Real& k0_,
               const Real& eps_,
               const Real& mu_,
               const Real& Omega_ = 0.)
      : meshx(mx), meshy(my),
        grad_phix(mx), grad_phiy(my),
        phix(mx), phiy(my),
        normalx(NormalTo(mx)), normaly(NormalTo(my)),
        k0(k0_), eps(eps_), mu(mu_), Omega(Omega_),
        khat(k0_ * std::sqrt(eps_ * mu_)),
        khat2(khat * khat),
        alpha((k0_ != 0.) ? (Omega_ / (k0_ * k0_)) : 0.),
        prefactor(iu / static_cast<Real>(4))
    {}

    void Assign(const int& ix, const int& iy)
    {
      const typename Trait::EltX& ex = meshx[ix];
      const typename Trait::EltY& ey = meshy[iy];
      grad_phix.Assign(ix);
      grad_phiy.Assign(iy);
      h = DetJac(ex) * DetJac(ey);
      x0_y0 = ex[0] - ey[0];
      dx = MatJac(ex);
      dy = MatJac(ey);
      nx = normalx[ix];
      ny = normaly[iy];
      x0_y0[2] = 0.0;
    }

    const typename Trait::MatType& operator()(const typename Trait::Rdx& tx,
                                              const typename Trait::Rdy& ty)
    {
      x = dx * tx;
      y = dy * ty;
      x_y = x0_y0 + x - y;
      r = norm2(x_y);

      const Cplx Gst = prefactor * Hankel(0, khat * r);
      // use reversed kernel \hat G
      const Real det = detail_rh_fast::det2_xy(x0_y0 + x, y);
      const Cplx phase = std::exp(-iu * alpha * det);
      ker = h * Gst * phase;

      for (int j = 0; j < Trait::nb_dof_x; ++j)
      {
        for (int k = 0; k < Trait::nb_dof_y; ++k)
        {
          val = (vprod(nx, grad_phix(j, tx)), vprod(ny, grad_phiy(k, ty))) * ker;
          inter(j, k) = val - khat2 * (nx, ny) * phix(j, tx) * phiy(k, ty) * ker;
        }
      }
      return inter;
    }

    const Cplx& operator()(const typename Trait::Rdx& tx,
                           const typename Trait::Rdy& ty,
                           const int& kx, const int& ky)
    {
      x = dx * tx;
      y = dy * ty;
      x_y = x0_y0 + x - y;
      r = norm2(x_y);

      const Cplx Gst = prefactor * Hankel(0, khat * r);
      const Real det = detail_rh_fast::det2_xy(x0_y0 + x, y);
      const Cplx phase = std::exp(-iu * alpha * det);
      ker = h * Gst * phase;

      val = (vprod(nx, grad_phix(kx, tx)), vprod(ny, grad_phiy(ky, ty))) * ker;
      val2 = val - khat2 * (nx, ny) * phix(kx, tx) * phiy(ky, ty) * ker;
      return val2;
    }
  };

  using RH_HS_2D_P1xP1 = BIOpKernel<RH, HS_OP, 2, P1_1D, P1_1D>;
  using RH_HS_2D_P2xP2 = BIOpKernel<RH, HS_OP, 2, P2_1D, P2_1D>;

} // namespace bemtool

#endif
