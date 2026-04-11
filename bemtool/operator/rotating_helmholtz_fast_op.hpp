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
    {
    }

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
  // Thesis convention: K is built from +∂_{n_y} \hat G.
  // The native singular structure is preserved; the rotating
  // phase contributes an additional smooth source-normal term.
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

    R3 x0, y0, x_y, x, y, ny;
    Real h{}, r{}, dotny{}, det{};
    Cplx ker, phase, Gst;

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
        alpha((k0_ != 0.) ? (Omega_ / (k0_ * k0_)) : 0.)
    {
    }

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
      x_y = x - y;
      r = norm2(x_y);

      det = detail_rh_fast::det2_xy(x, y);
      phase = std::exp(-iu * alpha * det);
      dotny = (ny, x_y);

      // Since \hat G = G_st exp(-i alpha det), we get
      // ∂_{n_y} \hat G = phase * ( ∂_{n_y} G_st - i alpha (∂_{n_y} det) G_st ).
      // Gst = (iu / static_cast<Real>(4)) * Hankel0(khat * r);
      const auto dr_dn_y = -dotny / r;
      const Real ddet_dn_y = - x[1] * ny[0] + x[0] * ny[1];

      ker = h * phase * ((iu / static_cast<Real>(4)) * (DHankel0_Dx(khat * r) * khat * dr_dn_y
        + (-iu * alpha * ddet_dn_y) * Hankel0(khat * r)));

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
  // Thesis convention: K' is built from +∂_{n_x} \hat G.
  // The native singular structure is preserved; the rotating
  // phase contributes an additional smooth target-normal term.
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

    R3 x0_y0, x_y, x, y, nx;
    Real h{}, r{}, dotnx{}, det{};
    Cplx ker, phase, Gst;

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
        alpha((k0_ != 0.) ? (Omega_ / (k0_ * k0_)) : 0.)
    {
    }

    void Assign(const int& ix, const int& iy)
    {
      const typename Trait::EltX& ex = meshx[ix];
      const typename Trait::EltY& ey = meshy[iy];
      x0_y0 = ex[0] - ey[0];
      dx = MatJac(ex);
      dy = MatJac(ey);
      h = DetJac(ex) * DetJac(ey);
      nx = normalx[ix];
    }

    const typename Trait::MatType& operator()(const typename Trait::Rdx& tx,
                                              const typename Trait::Rdy& ty)
    {
      x_y = x0_y0 + dx * tx - dy * ty;
      r = norm2(x_y);
      x = x0_y0 + dx * tx;
      y = dy * ty;
      det = detail_rh_fast::det2_xy(x, y);
      phase = std::exp(-iu * alpha * det);

      // Native singular part multiplied by the smooth phase.
      dotnx = (nx, x_y);
      ker = -h * dotnx * (1.0 / r) * static_cast<Real>(0.25) * iu * khat * Hankel1(khat * r) * phase;

      // Smooth correction from differentiating the phase in the target normal.
      Gst = (iu / static_cast<Real>(4)) * Hankel(0, khat * r);
      const Real ddet_dn_x = -y[1] * nx[0] + y[0] * nx[1];
      ker += -h * phase * (iu * alpha * ddet_dn_x) * Gst;

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
  // HS operator in 2D, mirroring the native Helmholtz structure.
  // The rotating correction enters only through the smooth phase.
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

    R3 x0_y0, x_y, x, y, nx, ny;
    Real h{}, r{}, det{};
    Cplx ker, val, val2, phase;

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
        alpha((k0_ != 0.) ? (Omega_ / (k0_ * k0_)) : 0.)
    {
    }

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
    }

    const typename Trait::MatType& operator()(const typename Trait::Rdx& tx,
                                              const typename Trait::Rdy& ty)
    {
      x_y = x0_y0 + dx * tx - dy * ty;
      r = norm2(x_y);
      x = x0_y0 + dx * tx;
      y = dy * ty;
      det = detail_rh_fast::det2_xy(x, y);
      phase = std::exp(-iu * alpha * det);
      ker = h * static_cast<Real>(0.25) * iu * Hankel0(khat * r) * phase;

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
      x_y = x0_y0 + dx * tx - dy * ty;
      r = norm2(x_y);
      x = x0_y0 + dx * tx;
      y = dy * ty;
      det = detail_rh_fast::det2_xy(x, y);
      phase = std::exp(-iu * alpha * det);
      ker = h * static_cast<Real>(0.25) * iu * Hankel0(khat * r) * phase;

      val = (vprod(nx, grad_phix(kx, tx)), vprod(ny, grad_phiy(ky, ty))) * ker;
      val2 = val - khat2 * (nx, ny) * phix(kx, tx) * phiy(ky, ty) * ker;
      return val2;
    }
  };

  using RH_HS_2D_P1xP1 = BIOpKernel<RH, HS_OP, 2, P1_1D, P1_1D>;
  using RH_HS_2D_P2xP2 = BIOpKernel<RH, HS_OP, 2, P2_1D, P2_1D>;
} // namespace bemtool

#endif
