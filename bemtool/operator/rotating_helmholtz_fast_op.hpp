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
    // This file implements the kernels
    //   \hat G(x,y) = (1/(4i)) H_0^{(1)}(alpha |x-y|)
    //                 exp( sigma (x \times y) )
    //
    // with alpha and sigma computed by the calling class.

    inline Real cross2(const R3& x, const R3& y)
    {
      return x[0] * y[1] - x[1] * y[0];
    }

    inline Real cross2_normals(const R3& x, const R3& y)
    {
      return x[0] * y[1] - x[1] * y[0];
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

    inline Real dist2d(const R3& x, const R3& y)
    {
      const Real dx = x[0] - y[0];
      const Real dy = x[1] - y[1];
      return std::hypot(dx, dy);
    }

    inline Cplx phase(const R3& x, const R3& y, Cplx sigma)
    {
      return std::exp(sigma * cross2(x, y));
    }

    inline Cplx G(const R3& x, const R3& y, Real alpha, Cplx sigma)
    {
      const Real R = dist2d(x, y);
      const Cplx H0 = Hankel(0, alpha * R);
      const Cplx ph = phase(x, y, sigma);
      return kernel_prefactor() * H0 * ph;
    }

    inline Cplx dnx_G(const R3& x, const R3& y,
                      const R3& nx, Real alpha, Cplx sigma)
    {
      const Real R = dist2d(x, y);
      const Real s_obs = (nx, (x - y)) / R;
      const Real qx = q_obs(y, nx);

      const Cplx H0 = Hankel(0, alpha * R);
      const Cplx H1 = Hankel1(alpha * R);
      const Cplx ph = phase(x, y, sigma);

      return kernel_prefactor() * ph
        * (-alpha * H1 * s_obs + sigma * H0 * qx);
    }

    inline Cplx dny_G(const R3& x, const R3& y,
                      const R3& ny, Real alpha, Cplx sigma)
    {
      const Real R = dist2d(x, y);
      const Real s_src = (ny, (x - y)) / R;
      const Real qy = q_src(x, ny);

      const Cplx H0 = Hankel(0, alpha * R);
      const Cplx H1 = Hankel1(alpha * R);
      const Cplx ph = phase(x, y, sigma);

      return kernel_prefactor() * ph
        * (alpha * H1 * s_src + sigma * H0 * qy);
    }

    inline Cplx dnx_dny_G(const R3& x, const R3& y,
                          const R3& nx, const R3& ny,
                          Real alpha, Cplx sigma)
    {
      const R3 d = x - y;
      const Real R = dist2d(x, y);

      const Real s_obs = (nx, d) / R; // s_X
      const Real s_src = (ny, d) / R; // s_Y
      const Real qx = q_obs(y, nx); // q_X
      const Real qy = q_src(x, ny); // q_Y
      const Real ndot = (nx, ny); // n_X \cdot n_Y
      const Real ncross = cross2_normals(nx, ny); // c_XY = n_X x n_Y

      const Cplx H0 = Hankel(0, alpha * R);
      const Cplx H1 = Hankel1(alpha * R);
      const Cplx ph = phase(x, y, sigma);

      // From the appendix derivation:
      // dnx dny Ghat
      // = (1/(4i)) exp(sigma x×y) [
      //     H0 ( alpha^2 sX sY + sigma cXY + sigma^2 qX qY )
      //   + H1 ( alpha/R (nXY - 2 sX sY)
      //          + alpha sigma (qX sY - qY sX) )
      //   ]
      const Cplx term0 =
      ((alpha * alpha) * s_obs * s_src
        + sigma * ncross
        + sigma * sigma * qx * qy) * H0;

      const Cplx term1 =
      (alpha * (ndot - 2.0 * s_obs * s_src) / R
        + alpha * sigma * (qx * s_src - qy * s_obs)) * H1;

      return kernel_prefactor() * ph * (term0 + term1);
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

    const Real k0, eps, mu, Omega, c0, khat;
    const Cplx sigma;

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
        khat(k0_ * std::sqrt(eps_ * mu_)),
        sigma(-iu * (Omega_ * k0_ / c0_))
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

      ker = h * detail_rh_fast::G(x, y, khat, sigma);

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

    const Real k0, eps, mu, Omega, c0, khat;
    const Cplx sigma;

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
        khat(k0_ * std::sqrt(eps_ * mu_)),
        sigma(-iu * (Omega_ * k0_ / c0_))
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

      ker = h * detail_rh_fast::dny_G(x, y, ny, khat, sigma);

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

    const Real k0, eps, mu, Omega, c0, khat;
    const Cplx sigma;

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
        khat(k0_ * std::sqrt(eps_ * mu_)),
        sigma(-iu * (Omega_ * k0_ / c0_))
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

      ker = h * detail_rh_fast::dnx_G(x, y, nx, khat, sigma);

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

    const Real k0, eps, mu, Omega, c0, khat;
    const Cplx sigma;

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
        khat(k0_ * std::sqrt(eps_ * mu_)),
        sigma(-iu * (Omega_ * k0_ / c0_))
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

      ker = -h * detail_rh_fast::dnx_dny_G(x, y, nx, ny, khat, sigma);

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

  using RH_HS_2D_P0xP1 = BIOpKernel<RH, HS_OP, 2, P0_1D, P1_1D>;
  using RH_HS_2D_P1xP0 = BIOpKernel<RH, HS_OP, 2, P1_1D, P0_1D>;

  using RH_HS_2D_P1xP1 = BIOpKernel<RH, HS_OP, 2, P1_1D, P1_1D>;
  using RH_HS_2D_P2xP2 = BIOpKernel<RH, HS_OP, 2, P2_1D, P2_1D>;


  // ============================================================
  // RH HS operator: weak / partially weak regularized forms
  // ============================================================
  //
  // This implementation dispatches at compile time depending on whether the
  // trial/test basis spaces support tangential derivatives.
  //
  // Cases:
  //   PhiX differentiable, PhiY differentiable:
  //       fully regularized weak form
  //
  //   PhiX differentiable, PhiY not differentiable:
  //       integration by parts only in the X/source variable
  //
  //   PhiX not differentiable, PhiY differentiable:
  //       integration by parts only in the Y/observation variable
  //
  //   PhiX not differentiable, PhiY not differentiable:
  //       disabled
  //

  // ------------------------------------------------------------
  // Compile-time trait: does Phi support GradPhi?
  // ------------------------------------------------------------
  template <typename Phi, typename = void>
  struct basis_has_grad : std::false_type
  {
  };

  // Version if GradBasisFct exists.
  template <typename Phi>
  struct basis_has_grad<Phi, std::void_t<decltype(sizeof(GradBasisFct<Phi>))>>
  : std::true_type
  {
  };

  template <typename T>
  inline constexpr bool basis_has_grad_v = basis_has_grad<T>::value;


  // ------------------------------------------------------------
  // Small geometry helpers
  // ------------------------------------------------------------
  namespace detail_rh_fast
  {
    inline R3 tau_from_normal(const R3& n)
    {
      // Positive tangent tau = e_z x n.
      // For n=(cos(theta),sin(theta),0), this gives
      // tau=(-sin(theta),cos(theta),0).
      R3 tau;
      tau[0] = -n[1];
      tau[1] = n[0];
      tau[2] = 0.;
      return tau;
    }

    inline Real dtau_phi_from_grad(const R3& n, const R3& grad_phi)
    {
      // BemTool's regularized Helmholtz HS form uses
      //
      //   (n_x x grad phi_x) . (n_y x grad phi_y).
      //
      // In 2D, n x grad_phi is parallel to e_z, and its z-component is
      // the oriented tangential derivative.
      return vprod(n, grad_phi)[2];
    }
  }


  // ------------------------------------------------------------
  // Implementation declaration
  // ------------------------------------------------------------
  template <typename PhiX,
            typename PhiY,
            bool HasGradPhiX,
            bool HasGradPhiY>
  class RH_HS_Weak_Impl;


  // ============================================================
  // Case 1: both spaces support tangential derivatives
  // Fully regularized weak form
  // ============================================================
  template <typename PhiX, typename PhiY>
  class RH_HS_Weak_Impl<PhiX, PhiY, true, true>
  {
  public:
    typedef BIOpKernelTraits<RH, HS_OP_WEAK, 2, PhiX, PhiY> Trait;

  private:
    const typename Trait::MeshX& meshx;
    const typename Trait::MeshY& meshy;
    const std::vector<R3>& normalx;
    const std::vector<R3>& normaly;

    typename Trait::MatType inter;
    typename Trait::JacX dx;
    typename Trait::JacY dy;
    typename Trait::GradPhiX grad_phix;
    typename Trait::GradPhiY grad_phiy;

    PhiX phix;
    PhiY phiy;

    const Real k0, eps, mu, Omega, c0, khat, khat2;
    const Cplx sigma;

    R3 x0, y0, x, y, nx, ny;
    Real h{};
    Cplx ker;

  public:
    RH_HS_Weak_Impl(const typename Trait::MeshX& mx,
                    const typename Trait::MeshY& my,
                    const Real& k0_,
                    const Real& eps_,
                    const Real& mu_,
                    const Real& Omega_ = 0.,
                    const Real& c0_ = 1.)
      : meshx(mx), meshy(my),
        normalx(NormalTo(mx)), normaly(NormalTo(my)),
        grad_phix(mx), grad_phiy(my),
        phix(mx), phiy(my),
        k0(k0_), eps(eps_), mu(mu_), Omega(Omega_), c0(c0_),
        khat(k0_ * std::sqrt(eps_ * mu_)),
        khat2(khat * khat),
        sigma(-iu * (Omega_ * k0_ / c0_))
    {
    }

    void Assign(const int& ix, const int& iy)
    {
      const typename Trait::EltX& ex = meshx[ix];
      const typename Trait::EltY& ey = meshy[iy];

      grad_phix.Assign(ix);
      grad_phiy.Assign(iy);

      x0 = ex[0];
      y0 = ey[0];
      dx = MatJac(ex);
      dy = MatJac(ey);
      h = DetJac(ex) * DetJac(ey);

      nx = normalx[ix];
      ny = normaly[iy];
    }

  private:
    Cplx entry(const typename Trait::Rdx& tx,
               const typename Trait::Rdy& ty,
               const int& j,
               const int& k)
    {
      const R3 taux = detail_rh_fast::tau_from_normal(nx);
      const R3 tauy = detail_rh_fast::tau_from_normal(ny);

      const Cplx G = detail_rh_fast::G(x, y, khat, sigma);

      const Real ux = phix(j, tx);
      const Real vy = phiy(k, ty);

      const Real dtx_u = detail_rh_fast::dtau_phi_from_grad(nx, grad_phix(j, tx));
      const Real dty_v = detail_rh_fast::dtau_phi_from_grad(ny, grad_phiy(k, ty));

      const Real qx = detail_rh_fast::q_obs(y, nx);
      const Real qy = detail_rh_fast::q_src(x, ny);
      const Real tx_phase = detail_rh_fast::q_obs(y, taux);
      const Real ty_phase = detail_rh_fast::q_src(x, tauy);
      const Real nn = (nx, ny);
      const Real nx_cross_ny = detail_rh_fast::cross2(nx, ny);

      // Phase-covariant derivatives of basis functions after integration by parts:
      //
      //   adjoint(D_tau) phi = partial_tau phi + sigma * partial_tau(S) * phi.
      const Cplx Du = dtx_u + sigma * tx_phase * ux;
      const Cplx Dv = dty_v + sigma * ty_phase * vy;

      // Covariant normal derivatives of G:
      //
      //   D_n G = partial_n G - sigma * partial_n(S) * G.
      const Cplx DnxG =
        detail_rh_fast::dnx_G(x, y, nx, khat, sigma) - sigma * qx * G;

      const Cplx DnyG =
        detail_rh_fast::dny_G(x, y, ny, khat, sigma) - sigma * qy * G;

      const Cplx lower =
        khat2 * nn * G
        + sigma * qy * DnxG
        + sigma * qx * DnyG
        + (sigma * nx_cross_ny + sigma * sigma * qx * qy) * G;

      return h * (G * Du * Dv - lower * ux * vy);
    }

  public:
    const typename Trait::MatType& operator()(const typename Trait::Rdx& tx,
                                              const typename Trait::Rdy& ty)
    {
      x = x0 + dx * tx;
      y = y0 + dy * ty;

      for (int j = 0; j < Trait::nb_dof_x; ++j)
        for (int k = 0; k < Trait::nb_dof_y; ++k)
          inter(j, k) = entry(tx, ty, j, k);

      return inter;
    }

    const Cplx& operator()(const typename Trait::Rdx& tx,
                           const typename Trait::Rdy& ty,
                           const int& kx,
                           const int& ky)
    {
      x = x0 + dx * tx;
      y = y0 + dy * ty;

      ker = entry(tx, ty, kx, ky);
      return ker;
    }
  };


  // ============================================================
  // Case 2: only PhiX supports tangential derivatives
  // Integrate by parts only in X
  // ============================================================
  template <typename PhiX, typename PhiY>
  class RH_HS_Weak_Impl<PhiX, PhiY, true, false>
  {
  public:
    typedef BIOpKernelTraits<RH, HS_OP_WEAK, 2, PhiX, PhiY> Trait;

  private:
    const typename Trait::MeshX& meshx;
    const typename Trait::MeshY& meshy;
    const std::vector<R3>& normalx;
    const std::vector<R3>& normaly;

    typename Trait::MatType inter;
    typename Trait::JacX dx;
    typename Trait::JacY dy;
    typename Trait::GradPhiX grad_phix;

    PhiX phix;
    PhiY phiy;

    const Real k0, eps, mu, Omega, c0, khat, khat2;
    const Cplx sigma;

    R3 x0, y0, x, y, nx, ny;
    Real h{};
    Cplx val2{};

  public:
    RH_HS_Weak_Impl(const typename Trait::MeshX& mx,
                    const typename Trait::MeshY& my,
                    const Real& k0_,
                    const Real& eps_,
                    const Real& mu_,
                    const Real& Omega_ = 0.,
                    const Real& c0_ = 1.)
      : meshx(mx), meshy(my),
        normalx(NormalTo(mx)), normaly(NormalTo(my)),
        grad_phix(mx),
        phix(mx), phiy(my),
        k0(k0_), eps(eps_), mu(mu_), Omega(Omega_), c0(c0_),
        khat(k0_ * std::sqrt(eps_ * mu_)),
        khat2(khat * khat),
        sigma(-iu * (Omega_ * k0_ / c0_))
    {
    }

    void Assign(const int& ix, const int& iy)
    {
      const typename Trait::EltX& ex = meshx[ix];
      const typename Trait::EltY& ey = meshy[iy];

      grad_phix.Assign(ix);

      x0 = ex[0];
      y0 = ey[0];
      dx = MatJac(ex);
      dy = MatJac(ey);
      h = DetJac(ex) * DetJac(ey);

      nx = normalx[ix];
      ny = normaly[iy];
    }

  private:
    Cplx entry(const typename Trait::Rdx& tx,
               const typename Trait::Rdy& ty,
               const int& j,
               const int& k)
    {
      const R3 taux = detail_rh_fast::tau_from_normal(nx);
      const R3 tauy = detail_rh_fast::tau_from_normal(ny);

      const Cplx G = detail_rh_fast::G(x, y, khat, sigma);

      const Real ux = phix(j, tx);
      const Real vy = phiy(k, ty);

      const Real dtx_u = detail_rh_fast::dtau_phi_from_grad(nx, grad_phix(j, tx));

      const Real qx = detail_rh_fast::q_obs(y, nx);
      const Real qy = detail_rh_fast::q_src(x, ny);
      const Real tx_phase = detail_rh_fast::q_obs(y, taux);
      const Real ty_phase = detail_rh_fast::q_src(x, tauy);
      const Real nn = (nx, ny);
      const Real nx_cross_ny = detail_rh_fast::cross2(nx, ny);

      const Cplx Du = dtx_u + sigma * tx_phase * ux;

      const Cplx DtauYG =
        detail_rh_fast::dny_G(x, y, tauy, khat, sigma)
        - sigma * ty_phase * G;

      const Cplx DnxG =
        detail_rh_fast::dnx_G(x, y, nx, khat, sigma) - sigma * qx * G;

      const Cplx DnyG =
        detail_rh_fast::dny_G(x, y, ny, khat, sigma) - sigma * qy * G;

      const Cplx lower =
        khat2 * nn * G
        + sigma * qy * DnxG
        + sigma * qx * DnyG
        + (sigma * nx_cross_ny + sigma * sigma * qx * qy) * G;

      // One-sided integration by parts in X:
      //
      //   integral u v D_tauX D_tauY G
      //     = - integral (D_tauX)^* u * D_tauY G * v.
      return h * (-Du * DtauYG * vy - lower * ux * vy);
    }

  public:
    const typename Trait::MatType& operator()(const typename Trait::Rdx& tx,
                                              const typename Trait::Rdy& ty)
    {
      x = x0 + dx * tx;
      y = y0 + dy * ty;

      for (int j = 0; j < Trait::nb_dof_x; ++j)
        for (int k = 0; k < Trait::nb_dof_y; ++k)
          inter(j, k) = entry(tx, ty, j, k);

      return inter;
    }

    const Cplx& operator()(const typename Trait::Rdx& tx,
                           const typename Trait::Rdy& ty,
                           const int& kx,
                           const int& ky)
    {
      x = x0 + dx * tx;
      y = y0 + dy * ty;

      val2 = entry(tx, ty, kx, ky);
      return val2;
    }
  };


  // ============================================================
  // Case 3: only PhiY supports tangential derivatives
  // Integrate by parts only in Y
  // ============================================================

  template <typename PhiX, typename PhiY>
  class RH_HS_Weak_Impl<PhiX, PhiY, false, true>
  {
  public:
    typedef BIOpKernelTraits<RH, HS_OP_WEAK, 2, PhiX, PhiY> Trait;

  private:
    const typename Trait::MeshX& meshx;
    const typename Trait::MeshY& meshy;
    const std::vector<R3>& normalx;
    const std::vector<R3>& normaly;

    typename Trait::MatType inter;
    typename Trait::JacX dx;
    typename Trait::JacY dy;
    typename Trait::GradPhiY grad_phiy;

    PhiX phix;
    PhiY phiy;

    const Real k0, eps, mu, Omega, c0, khat, khat2;
    const Cplx sigma;

    R3 x0, y0, x, y, nx, ny;
    Real h{};
    Cplx val2{};

  public:
    RH_HS_Weak_Impl(const typename Trait::MeshX& mx,
                    const typename Trait::MeshY& my,
                    const Real& k0_,
                    const Real& eps_,
                    const Real& mu_,
                    const Real& Omega_ = 0.,
                    const Real& c0_ = 1.)
      : meshx(mx), meshy(my),
        normalx(NormalTo(mx)), normaly(NormalTo(my)),
        grad_phiy(my),
        phix(mx), phiy(my),
        k0(k0_), eps(eps_), mu(mu_), Omega(Omega_), c0(c0_),
        khat(k0_ * std::sqrt(eps_ * mu_)),
        khat2(khat * khat),
        sigma(-iu * (Omega_ * k0_ / c0_))
    {
    }

    void Assign(const int& ix, const int& iy)
    {
      const typename Trait::EltX& ex = meshx[ix];
      const typename Trait::EltY& ey = meshy[iy];

      grad_phiy.Assign(iy);

      x0 = ex[0];
      y0 = ey[0];
      dx = MatJac(ex);
      dy = MatJac(ey);
      h = DetJac(ex) * DetJac(ey);

      nx = normalx[ix];
      ny = normaly[iy];
    }

  private:
    Cplx entry(const typename Trait::Rdx& tx,
               const typename Trait::Rdy& ty,
               const int& j,
               const int& k)
    {
      const R3 taux = detail_rh_fast::tau_from_normal(nx);
      const R3 tauy = detail_rh_fast::tau_from_normal(ny);

      const Cplx G = detail_rh_fast::G(x, y, khat, sigma);

      const Real ux = phix(j, tx);
      const Real vy = phiy(k, ty);

      const Real dty_v = detail_rh_fast::dtau_phi_from_grad(ny, grad_phiy(k, ty));

      const Real qx = detail_rh_fast::q_obs(y, nx);
      const Real qy = detail_rh_fast::q_src(x, ny);
      const Real tx_phase = detail_rh_fast::q_obs(y, taux);
      const Real ty_phase = detail_rh_fast::q_src(x, tauy);
      const Real nn = (nx, ny);
      const Real nx_cross_ny = detail_rh_fast::cross2(nx, ny);

      const Cplx Dv = dty_v + sigma * ty_phase * vy;

      const Cplx DtauXG =
        detail_rh_fast::dnx_G(x, y, taux, khat, sigma)
        - sigma * tx_phase * G;

      const Cplx DnxG =
        detail_rh_fast::dnx_G(x, y, nx, khat, sigma) - sigma * qx * G;

      const Cplx DnyG =
        detail_rh_fast::dny_G(x, y, ny, khat, sigma) - sigma * qy * G;

      const Cplx lower =
        khat2 * nn * G
        + sigma * qy * DnxG
        + sigma * qx * DnyG
        + (sigma * nx_cross_ny + sigma * sigma * qx * qy) * G;

      // One-sided integration by parts in Y:
      //
      //   integral u v D_tauX D_tauY G
      //     = - integral u * D_tauX G * (D_tauY)^* v.
      return h * (-ux * DtauXG * Dv - lower * ux * vy);
    }

  public:
    const typename Trait::MatType& operator()(const typename Trait::Rdx& tx,
                                              const typename Trait::Rdy& ty)
    {
      x = x0 + dx * tx;
      y = y0 + dy * ty;

      for (int j = 0; j < Trait::nb_dof_x; ++j)
        for (int k = 0; k < Trait::nb_dof_y; ++k)
          inter(j, k) = entry(tx, ty, j, k);

      return inter;
    }

    const Cplx& operator()(const typename Trait::Rdx& tx,
                           const typename Trait::Rdy& ty,
                           const int& kx,
                           const int& ky)
    {
      x = x0 + dx * tx;
      y = y0 + dy * ty;

      val2 = entry(tx, ty, kx, ky);
      return val2;
    }
  };


  // ============================================================
  // Case 4: neither space supports tangential derivatives
  // Disabled
  // ============================================================
  template <typename PhiX, typename PhiY>
  class RH_HS_Weak_Impl<PhiX, PhiY, false, false>
  {
  public:
    typedef BIOpKernelTraits<RH, HS_OP_WEAK, 2, PhiX, PhiY> Trait;

    RH_HS_Weak_Impl(const typename Trait::MeshX&,
                    const typename Trait::MeshY&,
                    const Real&,
                    const Real&,
                    const Real&,
                    const Real& = 0.,
                    const Real& = 1.)
    {
      static_assert(basis_has_grad_v<PhiX> || basis_has_grad_v<PhiY>,
                    "RH HS_OP_WEAK requires at least one basis space with tangential derivatives.");
    }
  };


  // ============================================================
  // Public BIOpKernel specialization
  // ============================================================
  template <typename PhiX, typename PhiY>
  class BIOpKernel<RH, HS_OP_WEAK, 2, PhiX, PhiY>
    : public RH_HS_Weak_Impl<PhiX, PhiY, basis_has_grad_v<PhiX>, basis_has_grad_v<PhiY>>
  {
  private:
    using Base =
    RH_HS_Weak_Impl<PhiX, PhiY, basis_has_grad_v<PhiX>, basis_has_grad_v<PhiY>>;

  public:
    using Base::Base;
  };

  using RH_HS_WEAK_2D_P0xP1 = BIOpKernel<RH, HS_OP_WEAK, 2, P0_1D, P1_1D>;
  using RH_HS_WEAK_2D_P1xP0 = BIOpKernel<RH, HS_OP_WEAK, 2, P1_1D, P0_1D>;

  using RH_HS_WEAK_2D_P1xP1 = BIOpKernel<RH, HS_OP_WEAK, 2, P1_1D, P1_1D>;
  using RH_HS_WEAK_2D_P2xP2 = BIOpKernel<RH, HS_OP_WEAK, 2, P2_1D, P2_1D>;
} // namespace bemtool

#endif
