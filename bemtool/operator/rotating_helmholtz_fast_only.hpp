//
// Created by jonas on 3/3/26.
//

// rotating_helmholtz.hpp
#ifndef BEMTOOL_OPERATOR_ROTATING_HELMHOLTZ_HPP
#define BEMTOOL_OPERATOR_ROTATING_HELMHOLTZ_HPP

#include "operator.hpp"   // BIOpKernelTraits, R3, MatJac, DetJac, iu, Real, Cplx, etc.
#include <complex>
#include <cmath>

namespace bemtool
{
  // =====================================================================================
  // NEW equation enum tag: RH  (Rotating Helmholtz)
  // You will add RH to the equation enum used by BIOpKernelTraits.
  // =====================================================================================

  // =====================================================================================
  // RH single-layer boundary operator kernel in 2D: BIOpKernel<RH,SL_OP,2,PhiX,PhiY>
  // Matches the structure/style of BIOpKernel<HE,SL_OP,2,...> but uses mode sum.
  // =====================================================================================
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

    // Physical parameters:
    //   k0   : vacuum wavenumber (omega/c0)
    //   eps, mu : (relative) permittivity and permeability of the medium
    //   Omega: rotation rate
    // Derived:
    //   khat  = k0 * sqrt(eps*mu)
    //   alpha = Omega / k0^2   (Steinberg–Shamir–Boag 2006)
    const Real k0;
    const Real eps;
    const Real mu;
    const Real Omega;
    const Real khat;
    const Real alpha;
    const Cplx prefactor; // i/4

    // Cached element data
    R3 x0, y0; // NOTE: we need absolute x and y, not only x-y
    R3 x, y;
    Real h{}; // DetJac(ex)*DetJac(ey)
    Cplx ker; // h * G_M(x,y)

  public:
    BIOpKernel(const typename Trait::MeshX& mx,
               const typename Trait::MeshY& my,
               const Real& k0_,
               const Real& eps_,
               const Real& mu_,
               const Real& Omega_ = 0.)
      : meshx(mx), meshy(my),
        phix(mx), phiy(my),
        k0(k0_), eps(eps_), mu(mu_), Omega(Omega_),
        khat(k0_ * std::sqrt(eps_ * mu_)),
        alpha((k0_ != 0.) ? (Omega_ / (k0_ * k0_)) : 0.),
        prefactor(iu / static_cast<Real>(4))
    {
      // Fast rotating Green's function (Steinberg–Shamir–Boag 2006, Eq. 3.19):
      //   G_rot ≈ G_st(R; khat) * exp(i * alpha * (x_y y_x - x_x y_y)),
      // with alpha = Omega / k0^2 and khat = k0 * sqrt(eps*mu).
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
      // absolute positions (rotation axis at origin)
      x = x0 + dx * tx;
      y = y0 + dy * ty;

      const Real dx_xy = x[0] - y[0];
      const Real dy_xy = x[1] - y[1];
      const Real R = std::hypot(dx_xy, dy_xy);

      // Stationary Green's function
      const Real zR = khat * R;
      const Cplx Gst = prefactor * Hankel(0, zR);

      // Rotation phase: exp(i * (Omega/k0^2) * (x_y y_x - x_x y_y))
      const Real det = x[1] * y[0] - x[0] * y[1];
      const Cplx phase = std::exp(iu * alpha * det);

      ker = h * (phase * Gst);

      for (int j = 0; j < Trait::nb_dof_x; ++j)
      {
        for (int k = 0; k < Trait::nb_dof_y; ++k)
        {
          inter(j, k) = ker * phix(j, tx) * phiy(k, ty);
        }
      }
      return inter;
    }

    const Cplx& operator()(const typename Trait::Rdx& tx,
                           const typename Trait::Rdy& ty,
                           const int& kx, const int& ky)
    {
      x = x0 + dx * tx;
      y = y0 + dy * ty;

      const Real dx_xy = x[0] - y[0];
      const Real dy_xy = x[1] - y[1];
      const Real R = std::hypot(dx_xy, dy_xy);

      const Real zR = khat * R;
      const Cplx Gst = prefactor * Hankel(0, zR);

      const Real det = x[1] * y[0] - x[0] * y[1];
      const Cplx phase = std::exp(iu * alpha * det);

      ker = h * (phase * Gst);

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

    // Physical parameters:
    //   k0   : vacuum wavenumber (omega/c0)
    //   eps, mu : (relative) permittivity and permeability of the medium
    //   Omega: rotation rate
    // Derived:
    //   khat  = k0 * sqrt(eps*mu)
    //   alpha = Omega / k0^2   (Steinberg–Shamir–Boag 2006)
    const Real k0;
    const Real eps;
    const Real mu;
    const Real Omega;
    const Real khat;
    const Real alpha;
    const Cplx prefactor; // i/4

    // Cached element data
    R3 x0, y0;
    R3 x, y;
    R3 ny;
    Real h{};
    Cplx ker;

  public:
    BIOpKernel(const typename Trait::MeshX& mx,
               const typename Trait::MeshY& my,
               const Real& k0_,
               const Real& eps_,
               const Real& mu_,
               const Real& Omega_ = 0.)
      : meshx(mx), meshy(my),
        phix(mx), phiy(my),
        k0(k0_), eps(eps_), mu(mu_), Omega(Omega_),
        normaly(NormalTo(my)),
        khat(k0_ * std::sqrt(eps_ * mu_)),
        alpha((k0_ != 0.) ? (Omega_ / (k0_ * k0_)) : 0.),
        prefactor(iu / static_cast<Real>(4))
    {
      // Fast rotating Green's function normal derivative:
      //   ∂_{n_y}G_rot ≈ exp(i*alpha*det) * ( ∂_{n_y}G_st + i*alpha*(∂_{n_y}det)*G_st )
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

      const Real dx_xy = x[0] - y[0];
      const Real dy_xy = x[1] - y[1];
      const Real R = std::hypot(dx_xy, dy_xy);

      // d/dn_y R = - ( (x-y)·n_y ) / R
      constexpr Real epsR = 1e-14;
      Real dRdn = 0;
      if (R > epsR)
      {
        dRdn = -(dx_xy * ny[0] + dy_xy * ny[1]) / R;
      }

      const Real zR = khat * R;
      const Cplx Gst = prefactor * Hankel(0, zR);

      // ∂_{n_y}G_st
      const Cplx H0p = DHankel_Dx(0, zR); // d/dz H0
      const Cplx dGst_dn = prefactor * (khat * H0p) * dRdn;

      // Rotation phase and its normal-derivative correction
      const Real det = x[1] * y[0] - x[0] * y[1];
      const Cplx phase = std::exp(iu * alpha * det);

      // ∂_{n_y} det = grad_y(det)·n_y, with grad_y(det) = (x_y, -x_x)
      const Real ddet_dn = x[1] * ny[0] - x[0] * ny[1];

      const Cplx dGdn = phase * (dGst_dn + (iu * alpha * ddet_dn) * Gst);

      // Match BEMTool DL convention used in your previous implementation
      ker = -h * dGdn;

      for (int j = 0; j < Trait::nb_dof_x; ++j)
      {
        for (int k = 0; k < Trait::nb_dof_y; ++k)
        {
          inter(j, k) = ker * phix(j, tx) * phiy(k, ty);
        }
      }
      return inter;
    }

    const Cplx& operator()(const typename Trait::Rdx& tx,
                           const typename Trait::Rdy& ty,
                           const int& kx, const int& ky)
    {
      // Keep consistency with the block evaluator:
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
} // namespace bemtool

#endif
