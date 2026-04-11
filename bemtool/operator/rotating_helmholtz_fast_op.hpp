#ifndef BEMTOOL_OPERATOR_ROTATING_HELMHOLTZ_HPP
#define BEMTOOL_OPERATOR_ROTATING_HELMHOLTZ_HPP

#include "operator.hpp"
#include <complex>
#include <cmath>

namespace bemtool
{
  namespace detail_rh_fast
  {
    // BemTool kernel convention from BIOp/operator.hpp:
    //   ker(x,y) with x on meshx (observation / test side)
    //   and y on meshy (source / trial side).
    //
    // This file implements the kernels derived in the thesis for
    //   \hat G(x,y) = (1/(4i)) H_0^{(1)}(kappa |x-y|)
    //                 exp( sigma (x \times y) )
    // with
    //   kappa = k0 * sqrt(eps*mu),
    //   sigma = -i * Omega * k0 / c0.

    inline Real cross2(const R3& a, const R3& b)
    {
      return a[0] * b[1] - a[1] * b[0];
    }

    inline Real cross2_normals(const R3& a, const R3& b)
    {
      return a[0] * b[1] - a[1] * b[0];
    }

    // q_obs = y_2 n_{x,1} - y_1 n_{x,2}
    inline Real q_obs(const R3& y, const R3& nx)
    {
      return y[1] * nx[0] - y[0] * nx[1];
    }

    // q_src = -x_2 n_{y,1} + x_1 n_{y,2}
    inline Real q_src(const R3& x, const R3& ny)
    {
      return -x[1] * ny[0] + x[0] * ny[1];
    }

    inline Cplx kernel_prefactor()
    {
      // 1/(4i) = -i/4
      return -iu / static_cast<Real>(4);
    }
  }

  // ============================================================
  // SL operator: Dirichlet trace of the single-layer potential
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

    const Real k0, eps, mu, Omega, c0, kappa;
    const Cplx sigma;
    const Cplx pref;

    R3 x0, y0, x, y;
    Real h{};
    Cplx ker;

  public:
    BIOpKernel(const typename Trait::MeshX& mx,
               const typename Trait::MeshY& my,
               const Real& k0_,
               const Real& eps_,
               const Real& mu_,
               const Real& Omega_ = 0.,
               const Real& c0_ = 1.)
      : meshx(mx), meshy(my), phix(mx), phiy(my),
        k0(k0_), eps(eps_), mu(mu_), Omega(Omega_), c0(c0_),
        kappa(k0_ * std::sqrt(eps_ * mu_)),
        sigma(-iu * (Omega_ * k0_ / c0_)),
        pref(detail_rh_fast::kernel_prefactor())
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

      const R3 d = x - y;
      const Real R = norm2(d);
      const Cplx H0 = Hankel(0, kappa * R);
      const Cplx phase = std::exp(sigma * detail_rh_fast::cross2(x, y));

      ker = h * pref * H0 * phase;

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
  // DL operator: Dirichlet trace of the double-layer potential
  // = source-normal derivative \partial_{n_y} \hat G(x,y)
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

    const Real k0, eps, mu, Omega, c0, kappa;
    const Cplx sigma;
    const Cplx pref;

    R3 x0, y0, x, y, ny;
    Real h{};
    Cplx ker;

  public:
    BIOpKernel(const typename Trait::MeshX& mx,
               const typename Trait::MeshY& my,
               const Real& k0_,
               const Real& eps_,
               const Real& mu_,
               const Real& Omega_ = 0.,
               const Real& c0_ = 1.)
      : meshx(mx), meshy(my), normaly(NormalTo(my)),
        phix(mx), phiy(my),
        k0(k0_), eps(eps_), mu(mu_), Omega(Omega_), c0(c0_),
        kappa(k0_ * std::sqrt(eps_ * mu_)),
        sigma(-iu * (Omega_ * k0_ / c0_)),
        pref(detail_rh_fast::kernel_prefactor())
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

      const R3 d = x - y;
      const Real R = norm2(d);
      const Real s_src = (ny, d) / R;
      const Real q_src = detail_rh_fast::q_src(x, ny);

      const Cplx H0 = Hankel(0, kappa * R);
      const Cplx H1 = Hankel1(kappa * R);
      const Cplx phase = std::exp(sigma * detail_rh_fast::cross2(x, y));

      ker = h * pref * phase * (kappa * H1 * s_src + sigma * H0 * q_src);

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
  // TDL operator: Neumann trace of the single-layer potential
  // = observation-normal derivative \partial_{n_x} \hat G(x,y)
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

    const Real k0, eps, mu, Omega, c0, kappa;
    const Cplx sigma;
    const Cplx pref;

    R3 x0, y0, x, y, nx;
    Real h{};
    Cplx ker;

  public:
    BIOpKernel(const typename Trait::MeshX& mx,
               const typename Trait::MeshY& my,
               const Real& k0_,
               const Real& eps_,
               const Real& mu_,
               const Real& Omega_ = 0.,
               const Real& c0_ = 1.)
      : meshx(mx), meshy(my), normalx(NormalTo(mx)),
        phix(mx), phiy(my),
        k0(k0_), eps(eps_), mu(mu_), Omega(Omega_), c0(c0_),
        kappa(k0_ * std::sqrt(eps_ * mu_)),
        sigma(-iu * (Omega_ * k0_ / c0_)),
        pref(detail_rh_fast::kernel_prefactor())
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
      nx = normalx[ix];
    }

    const typename Trait::MatType& operator()(const typename Trait::Rdx& tx,
                                              const typename Trait::Rdy& ty)
    {
      x = x0 + dx * tx;
      y = y0 + dy * ty;

      const R3 d = x - y;
      const Real R = norm2(d);
      const Real s_obs = (nx, d) / R;
      const Real q_obs = detail_rh_fast::q_obs(y, nx);

      const Cplx H0 = Hankel(0, kappa * R);
      const Cplx H1 = Hankel1(kappa * R);
      const Cplx phase = std::exp(sigma * detail_rh_fast::cross2(x, y));

      ker = h * pref * phase * (-kappa * H1 * s_obs + sigma * H0 * q_obs);

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
  // HS operator: Neumann trace of the double-layer potential
  // = - \partial_{n_x} \partial_{n_y} \hat G(x,y)
  // ============================================================
  template <typename PhiX, typename PhiY>
  class BIOpKernel<RH, HS_OP, 2, PhiX, PhiY>
  {
  public:
    typedef BIOpKernelTraits<RH, HS_OP, 2, PhiX, PhiY> Trait;

  private:
    const typename Trait::MeshX& meshx;
    const typename Trait::MeshY& meshy;
    const std::vector<R3>& normalx;
    const std::vector<R3>& normaly;
    typename Trait::MatType inter;
    typename Trait::JacX dx;
    typename Trait::JacY dy;
    PhiX phix;
    PhiY phiy;

    const Real k0, eps, mu, Omega, c0, kappa;
    const Cplx sigma;
    const Cplx pref;

    R3 x0, y0, x, y, nx, ny;
    Real h{};
    Cplx ker;

  public:
    BIOpKernel(const typename Trait::MeshX& mx,
               const typename Trait::MeshY& my,
               const Real& k0_,
               const Real& eps_,
               const Real& mu_,
               const Real& Omega_ = 0.,
               const Real& c0_ = 1.)
      : meshx(mx), meshy(my), normalx(NormalTo(mx)), normaly(NormalTo(my)),
        phix(mx), phiy(my),
        k0(k0_), eps(eps_), mu(mu_), Omega(Omega_), c0(c0_),
        kappa(k0_ * std::sqrt(eps_ * mu_)),
        sigma(-iu * (Omega_ * k0_ / c0_)),
        pref(detail_rh_fast::kernel_prefactor())
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
      nx = normalx[ix];
      ny = normaly[iy];
    }

    const typename Trait::MatType& operator()(const typename Trait::Rdx& tx,
                                              const typename Trait::Rdy& ty)
    {
      x = x0 + dx * tx;
      y = y0 + dy * ty;

      const R3 d = x - y;
      const Real R = norm2(d);
      const Real s_obs = (nx, d) / R;
      const Real s_src = (ny, d) / R;
      const Real q_obs = detail_rh_fast::q_obs(y, nx);
      const Real q_src = detail_rh_fast::q_src(x, ny);
      const Real ndot = (nx, ny);
      const Real ncross = detail_rh_fast::cross2_normals(nx, ny);

      const Cplx H0 = Hankel(0, kappa * R);
      const Cplx H1 = Hankel1(kappa * R);
      const Cplx phase = std::exp(sigma * detail_rh_fast::cross2(x, y));

      const Cplx term0 = (sigma * sigma * q_obs * q_src
                          - sigma * ncross
                          - (kappa * kappa) * s_obs * s_src) * H0;

      const Cplx term1 = (kappa * ndot / R
                          - sigma * kappa * (q_obs * s_src - q_src * s_obs)) * H1;

      ker = -h * pref * phase * (term0 + term1);

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

  using RH_HS_2D_P1xP1 = BIOpKernel<RH, HS_OP, 2, P1_1D, P1_1D>;
  using RH_HS_2D_P2xP2 = BIOpKernel<RH, HS_OP, 2, P2_1D, P2_1D>;
} // namespace bemtool

#endif
