// rotating_helmholtz.hpp
#ifndef BEMTOOL_OPERATOR_ROTATING_HELMHOLTZ_HPP
#define BEMTOOL_OPERATOR_ROTATING_HELMHOLTZ_HPP

#include "operator.hpp"   // BIOpKernelTraits, R3, MatJac, DetJac, iu, Real, Cplx, etc.
#include <complex>
#include <cmath>
#include <vector>

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

    // Parameters (you’ll decide later how you pass these through BIOp)
    const Real khat; // your \hat{k}
    const Real Omega; // rotation rate
    const Real c; // wave speed (or scaling)
    const int M; // truncation
    const bool keep_D2; // whether to include -Omega^2 m^2 / c^2 term

    // Precomputed per-mode values (m = -M..M).
    std::vector<Real> kappa_m_cache;
    const Cplx prefactor; // i/4 (consistent with HE convention)

    // Cached element data
    R3 x0, y0; // NOTE: we need absolute x and y, not only x-y
    R3 x, y;
    Real h{}; // DetJac(ex)*DetJac(ey)
    Cplx ker; // h * G_M(x,y)

  public:
    BIOpKernel(const typename Trait::MeshX& mx,
               const typename Trait::MeshY& my,
               const Real& khat_,
               const Real& Omega_ = 0.,
               const Real& c_ = 1.,
               const int& M_ = 5,
               const bool& keep_D2_ = false)
      : meshx(mx), meshy(my),
        phix(mx), phiy(my),
        khat(khat_), Omega(Omega_), c(c_), M(M_), keep_D2(keep_D2_)
      , kappa_m_cache(2 * M_ + 1)
      , prefactor(iu / static_cast<Real>(4))
    {
      // Precompute kappa_m and Hankel kind selection once per kernel instance.
      for (int m = -M; m <= M; ++m)
      {
        // NOTE: This kernel is meant to match the "sign-flipped" test operator
        //   (k^2 - 2 i k D) + D^2  (i.e. multiply (-k^2 + 2 i k D) by -1, keep D^2 as-is).
        // After Fourier in phi: D -> (Omega/c) * (i m), hence -2 i k D -> +2 k m (Omega/c).
        // D^2 contributes -(Omega^2/c^2) m^2.
        Real kappa2 = +(khat * khat) + static_cast<Real>(2) * khat * static_cast<Real>(m) * (Omega / c);
        if (keep_D2) kappa2 -= (Omega * Omega) * static_cast<Real>(m * m) / (c * c);

        const Real kappa_m = std::sqrt(kappa2);
        const std::size_t idx = (m + M);
        kappa_m_cache[idx] = kappa_m;
      }
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
      // absolute positions (crucial for r,phi and r',phi')
      x = x0 + dx * tx;
      y = y0 + dy * ty;

      const Real rx = std::hypot(x[0], x[1]);
      const Real ry = std::hypot(y[0], y[1]);
      const Real phix_ang = std::atan2(x[1], x[0]);
      const Real phiy_ang = std::atan2(y[1], y[0]);

      const Real rmin = (rx < ry) ? rx : ry;
      const Real rmax = (rx < ry) ? ry : rx;
      const Real dphi = phix_ang - phiy_ang;

      // Singular + rotating remainder evaluation (stable when r≈r')
      // We write: G_rot = G_HE_direct + Σ_m (term_rot(m) - term_HE(m)).
      // For Ω=0, term_rot==term_HE so the sum vanishes, and we recover HE to machine precision.
      const Real dx_xy = x[0] - y[0];
      const Real dy_xy = x[1] - y[1];
      const Real R = std::hypot(dx_xy, dy_xy);

      // Direct Helmholtz singular part (same local singularity): (i/4) H_0^{(1)}(k R)
      const Real zR(khat * R);
      Cplx G = prefactor * Hankel(0, zR);

      // Remainder mode-sum (converges much faster at r=r')
      // Build phases by recurrence to avoid exp() per mode.
      const Cplx e_idphi = std::exp(iu * dphi);
      Cplx phase = std::exp(iu * static_cast<Real>(-M) * dphi);

      constexpr Real eps = 1e-14;
      const bool rmin_is_zero = (rmin < eps);

      for (int m = -M; m <= M; ++m)
      {
        const std::size_t idx = (m + M);

        // --- rotating term ---
        const Real kappa_m = kappa_m_cache[idx];
        const Real zmin_rot = kappa_m * rmin;
        const Real zmax_rot = kappa_m * rmax;

        Cplx Jm_rot = 0;
        if (rmin_is_zero)
        {
          if (m == 0) Jm_rot = 1;
        }
        else
        {
          Jm_rot = BesselJ(m, zmin_rot);
        }

        const Cplx Hm_rot = Hankel(m, zmax_rot);

        // --- reference Helmholtz (non-rotating) Graf term ---
        const Real zmin_he = khat * rmin;
        const Real zmax_he = khat * rmax;

        Cplx Jm_he = 0;
        if (rmin_is_zero)
        {
          if (m == 0) Jm_he = 1;
        }
        else
        {
          Jm_he = BesselJ(m, zmin_he);
        }

        const Cplx Hm_he = Hankel(m, zmax_he);

        // Add remainder contribution
        G += prefactor * phase * (Jm_rot * Hm_rot - Jm_he * Hm_he);

        phase *= e_idphi;
      }

      ker = h * G;

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
      // compute ker at (tx,ty)
      x = x0 + dx * tx;
      y = y0 + dy * ty;

      const Real rx = std::hypot(x[0], x[1]);
      const Real ry = std::hypot(y[0], y[1]);
      const Real phix_ang = std::atan2(x[1], x[0]);
      const Real phiy_ang = std::atan2(y[1], y[0]);

      const Real rmin = (rx < ry) ? rx : ry;
      const Real rmax = (rx < ry) ? ry : rx;
      const Real dphi = phix_ang - phiy_ang;

      // Singular + rotating remainder evaluation (stable when r≈r')
      const Real dx_xy = x[0] - y[0];
      const Real dy_xy = x[1] - y[1];
      const Real R = std::hypot(dx_xy, dy_xy);

      const Real zR = khat * R;
      Cplx G = prefactor * Hankel(0, zR);

      const Cplx e_idphi = std::exp(iu * dphi);
      Cplx phase = std::exp(iu * static_cast<Real>(-M) * dphi);

      constexpr Real eps = 1e-14;
      const bool rmin_is_zero = (rmin < eps);

      for (int m = -M; m <= M; ++m)
      {
        const std::size_t idx = (m + M);

        const Real kappa_m = kappa_m_cache[idx];
        const Real zmin_rot = kappa_m * rmin;
        const Real zmax_rot = kappa_m * rmax;

        Cplx Jm_rot = 0;
        if (rmin_is_zero)
        {
          if (m == 0) Jm_rot = 1;
        }
        else
        {
          Jm_rot = BesselJ(m, zmin_rot);
        }

        const Cplx Hm_rot = Hankel(m, zmax_rot);

        const Real zmin_he = khat * rmin;
        const Real zmax_he = khat * rmax;

        Cplx Jm_he = 0;
        if (rmin_is_zero)
        {
          if (m == 0) Jm_he = 1;
        }
        else
        {
          Jm_he = BesselJ(m, zmin_he);
        }

        const Cplx Hm_he = Hankel(m, zmax_he);

        G += prefactor * phase * (Jm_rot * Hm_rot - Jm_he * Hm_he);
        phase *= e_idphi;
      }

      ker = h * G;

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

  // =====================================================================================
  // RH double-layer boundary operator kernel in 2D: BIOpKernel<RH,DL_OP,2,PhiX,PhiY>
  // Convention matches BEMTool's HE,DL_OP,2: this is ∂_{n_y} G(x,y) with ny = NormalTo(y-element).
  // =====================================================================================
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

    // Parameters
    const Real khat;
    const Real Omega;
    const Real c;
    const int M;
    const bool keep_D2;

    // Precomputed per-mode values (m = -M..M).
    std::vector<Real> kappa_m_cache;
    const Cplx prefactor; // i/4 (consistent with HE convention)

    // Cached element data
    R3 x0, y0;
    R3 x, y;
    R3 ny;
    Real h{};
    Cplx ker;

  public:
    BIOpKernel(const typename Trait::MeshX& mx,
               const typename Trait::MeshY& my,
               const Real& khat_,
               const Real& Omega_ = 0.,
               const Real& c_ = 1.,
               const int& M_ = 5,
               const bool& keep_D2_ = false)
      : meshx(mx), meshy(my),
        normaly(NormalTo(my)),
        phix(mx), phiy(my),
        khat(khat_), Omega(Omega_), c(c_), M(M_), keep_D2(keep_D2_)
      , kappa_m_cache((2 * M_ + 1))
      , prefactor(iu / static_cast<Real>(4))
    {
      for (int m = -M; m <= M; ++m)
      {
        // See SL kernel for sign convention.
        Real kappa2 = +(khat * khat) + static_cast<Real>(2) * khat * static_cast<Real>(m) * (Omega / c);
        if (keep_D2) kappa2 -= (Omega * Omega) * static_cast<Real>(m * m) / (c * c);

        const Real kappa_m = std::sqrt(kappa2);
        const std::size_t idx = (m + M);
        kappa_m_cache[idx] = kappa_m;
      }
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

      // polar coords about origin (rotation axis)
      const Real rx = std::hypot(x[0], x[1]);
      const Real ry = std::hypot(y[0], y[1]);

      const Real phix_ang = std::atan2(x[1], x[0]);
      const Real phiy_ang = std::atan2(y[1], y[0]);
      const Real dphi = phix_ang - phiy_ang;

      // geometry derivatives at y:
      // ∂_{n_y} r_y = (n_y · y)/r_y
      // ∂_{n_y} φ_y = (-y2 n1 + y1 n2)/r_y^2
      Real dn_r = 0;
      Real dn_phi = 0;

      constexpr Real eps = 1e-14;
      if (ry > eps)
      {
        dn_r = (ny[0] * y[0] + ny[1] * y[1]) / ry;
        dn_phi = (-y[1] * ny[0] + y[0] * ny[1]) / (ry * ry);
      }

      // piecewise: which branch depends on y?
      const bool y_is_rmin = (ry < rx);
      const Real rmin = y_is_rmin ? ry : rx;
      const Real rmax = y_is_rmin ? rx : ry;

      // Singular + rotating remainder evaluation for ∂_{n_y}G (stable when r≈r')
      // dGdn_rot = dGdn_HE_direct + Σ_m (dterm_rot(m) - dterm_HE(m)).
      // Direct Helmholtz normal-derivative part (matches BemTool's HE singularity):
      const Real dx_xy = x[0] - y[0];
      const Real dy_xy = x[1] - y[1];
      const Real R = std::hypot(dx_xy, dy_xy);

      // d/dn_y R = - ( (x-y)·n_y ) / R
      Real dRdn = 0;
      if (R > eps)
      {
        dRdn = - (dx_xy * ny[0] + dy_xy * ny[1]) / R;
      }

      const Real zR = khat * R;
      const Cplx H0p = DHankel_Dx(0, zR); // d/dz H0
      Cplx dGdn = prefactor * (khat * H0p) * dRdn;

      // Phase recurrence (avoid exp() per mode)
      const Cplx e_idphi = std::exp(iu * dphi);
      Cplx phase = std::exp(iu * static_cast<Real>(-M) * dphi);

      const bool rmin_is_zero = (rmin < eps);

      for (int m = -M; m <= M; ++m)
      {
        const std::size_t idx = (m + M);

        // ---------- rotating term pieces ----------
        const Real kappa_m = kappa_m_cache[idx];
        const Real zmin_rot = kappa_m * rmin;
        const Real zmax_rot = kappa_m * rmax;

        const Cplx dphase = phase * (-static_cast<Real>(m) * dn_phi) * iu; // ∂n phase

        Cplx Jm_min_rot = 0;
        if (rmin_is_zero)
        {
          if (m == 0) Jm_min_rot = 1;
        }
        else
        {
          Jm_min_rot = BesselJ(m, zmin_rot);
        }

        const Cplx Hm_max_rot = Hankel(m, zmax_rot);

        Cplx dJdn_rot = 0;
        Cplx dHdn_rot = 0;
        if (y_is_rmin)
        {
          dJdn_rot = kappa_m * DBesselJ_Dx(m, zmin_rot) * dn_r;
        }
        else
        {
          const Cplx Hp_rot = DHankel_Dx(m, zmax_rot);
          dHdn_rot = kappa_m * Hp_rot * dn_r;
        }

        const Cplx dterm_rot = dphase * Jm_min_rot * Hm_max_rot
          + phase * dJdn_rot * Hm_max_rot
          + phase * Jm_min_rot * dHdn_rot;

        // ---------- reference Helmholtz term pieces ----------
        const Real kappa_he = khat;
        const Real zmin_he = kappa_he * rmin;
        const Real zmax_he = kappa_he * rmax;

        Cplx Jm_min_he = 0;
        if (rmin_is_zero)
        {
          if (m == 0) Jm_min_he = 1;
        }
        else
        {
          Jm_min_he = BesselJ(m, zmin_he);
        }

        const Cplx Hm_max_he = Hankel(m, zmax_he);

        Cplx dJdn_he = 0;
        Cplx dHdn_he = 0;
        if (y_is_rmin)
        {
          dJdn_he = kappa_he * DBesselJ_Dx(m, zmin_he) * dn_r;
        }
        else
        {
          const Cplx Hp_he = DHankel_Dx(m, zmax_he);
          dHdn_he = kappa_he * Hp_he * dn_r;
        }

        const Cplx dterm_he = dphase * Jm_min_he * Hm_max_he
          + phase * dJdn_he * Hm_max_he
          + phase * Jm_min_he * dHdn_he;

        // Add remainder contribution
        dGdn += prefactor * (dterm_rot - dterm_he);

        phase *= e_idphi;
      }

      // BEMTool's 2D Helmholtz DL kernel is defined with an extra minus sign:
      //   HE,DL_OP,2: ker = -h * (n_y · (x-y)) * (1/r) * (i/4) * k * H_1^{(1)}(k r)
      // which corresponds to  -∂_{n_y} G  when  G = (i/4) H_0^{(1)}(k r)
      // (since d/dr H_0^{(1)}(z) = -H_1^{(1)}(z)). See helmholtz_op.hpp.
      // To stay consistent with BEMTool, we therefore return ker = -h * ∂_{n_y}G.
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
