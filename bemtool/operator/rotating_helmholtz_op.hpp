// rotating_helmholtz.hpp
#ifndef BEMTOOL_OPERATOR_ROTATING_HELMHOLTZ_HPP
#define BEMTOOL_OPERATOR_ROTATING_HELMHOLTZ_HPP

#include "operator.hpp"   // BIOpKernelTraits, R3, MatJac, DetJac, iu, Real, Cplx, etc.
#include <complex>
#include <cmath>
#include <vector>

// Complex-valued Bessel/Hankel functions (AMOS wrapper).
// API: sp_bessel::besselJ / besselJp / hankelH1 / hankelH1p / hankelH2 / hankelH2p
// See: https://blog.joey-dumont.ca/complex_bessel/docs.html
#include <complex_bessel.h>

namespace bemtool
{
  // --- Helpers: integer-order Bessel/Hankel for complex argument ---
  // complex_bessel already implements negative orders via reflection formulae.
  inline Cplx besselJ_int(const int m, const Cplx& z) { return sp_bessel::besselJ(m, z); }
  inline Cplx besselY_int(const int m, const Cplx& z) { return sp_bessel::besselY(m, z); }
  inline Cplx hankel1_int(const int m, const Cplx& z) { return sp_bessel::hankelH1(m, z); }
  inline Cplx hankel2_int(const int m, const Cplx& z) { return sp_bessel::hankelH2(m, z); }

  // Pick H^(1) vs H^(2) based on sign(Im(kappa_m))
  inline Cplx hankel_rad_int(const int m, const Cplx& z, const Cplx& kappa_m)
  {
    const Real imk = std::imag(kappa_m);
    if (imk > static_cast<Real>(0)) return hankel1_int(m, z);
    if (imk < static_cast<Real>(0)) return hankel2_int(m, z);
    return hankel1_int(m, z); // Im=0: outgoing convention
  }

  // Square-root branch choice used to define kappa_m = sqrt(kappa_m^2).
  // We enforce the "radiation"/decay-friendly branch
  //   Im(kappa_m) >= 0, and if Im(kappa_m)=0 then Re(kappa_m) >= 0.
  // With the far-field asymptotics H_m^{(1)}(kappa r) ~ exp(i kappa r)/sqrt(r),
  // this ensures outgoing oscillations for real kappa>0 and exponential decay for Im(kappa)>0.
  inline Cplx kappa_from_kappa2(const Cplx& kappa2)
  {
    Cplx k = std::sqrt(kappa2);
    const Real im = std::imag(k);
    const Real re = std::real(k);
    if (im < static_cast<Real>(0) || (im == static_cast<Real>(0) && re < static_cast<Real>(0))) k = -k;
    return k;
  }

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
    std::vector<Cplx> kappa_m_cache;
    std::vector<int> hankel_kind_cache; // 1 => H^(1), 2 => H^(2)
    const Cplx prefactor; // 1/(4 i)

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
               const int& M_ = 20,
               const bool& keep_D2_ = false)
      : meshx(mx), meshy(my),
        phix(mx), phiy(my),
        khat(khat_), Omega(Omega_), c(c_), M(M_), keep_D2(keep_D2_)
      , kappa_m_cache(static_cast<std::size_t>(2 * M_ + 1))
      , hankel_kind_cache(static_cast<std::size_t>(2 * M_ + 1), 1)
      , prefactor(static_cast<Real>(1) / (static_cast<Real>(4) * iu))
    {
      // Precompute kappa_m and Hankel kind selection once per kernel instance.
      for (int m = -M; m <= M; ++m)
      {
        // NOTE: This kernel is meant to match the "sign-flipped" test operator
        //   (k^2 - 2 i k D) + D^2  (i.e. multiply (-k^2 + 2 i k D) by -1, keep D^2 as-is).
        // After Fourier in phi: D -> (Omega/c) * (i m), hence -2 i k D -> +2 k m (Omega/c).
        // D^2 contributes -(Omega^2/c^2) m^2.
        Cplx kappa2 = +(khat * khat) + static_cast<Real>(2) * khat * static_cast<Real>(m) * (Omega / c);
        if (keep_D2) kappa2 -= (Omega * Omega) * static_cast<Real>(m * m) / (c * c);

        const Cplx kappa_m = kappa_from_kappa2(kappa2);
        const auto idx = static_cast<std::size_t>(m + M);
        kappa_m_cache[idx] = kappa_m;

        const Real imk = std::imag(kappa_m);
        hankel_kind_cache[idx] = (imk < static_cast<Real>(0)) ? 2 : 1; // Im=0 => outgoing => H^(1)
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

      // truncated mode-sum Green function
      Cplx G = 0;

      // Build phases by recurrence to avoid exp() per mode.
      const Cplx e_idphi = std::exp(iu * dphi);
      Cplx phase = std::exp(iu * static_cast<Real>(-M) * dphi);

      constexpr Real eps = (Real)1e-14;
      const bool rmin_is_zero = (rmin < eps);

      for (int m = -M; m <= M; ++m)
      {
        const auto idx = static_cast<std::size_t>(m + M);
        const Cplx kappa_m = kappa_m_cache[idx];
        const Cplx zmin = kappa_m * rmin;
        const Cplx zmax = kappa_m * rmax;

        // Axis handling: J_m(0) = 0 for m!=0, J_0(0)=1.
        Cplx Jm = (Real)0;
        if (rmin_is_zero)
        {
          if (m == 0) Jm = (Real)1;
        }
        else
        {
          Jm = besselJ_int(m, zmin);
        }

        const Cplx Hm = (hankel_kind_cache[idx] == 1) ? hankel1_int(m, zmax) : hankel2_int(m, zmax);
        G += prefactor * phase * (Jm * Hm);

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

      Cplx G = (Real)0;
      const Cplx e_idphi = std::exp(iu * dphi);
      Cplx phase = std::exp(iu * static_cast<Real>(-M) * dphi);

      constexpr Real eps = (Real)1e-14;
      const bool rmin_is_zero = (rmin < eps);

      for (int m = -M; m <= M; ++m)
      {
        const auto idx = static_cast<std::size_t>(m + M);
        const Cplx kappa_m = kappa_m_cache[idx];
        const Cplx zmin = kappa_m * rmin;
        const Cplx zmax = kappa_m * rmax;

        Cplx Jm = (Real)0;
        if (rmin_is_zero)
        {
          if (m == 0) Jm = (Real)1;
        }
        else
        {
          Jm = besselJ_int(m, zmin);
        }

        const Cplx Hm = (hankel_kind_cache[idx] == 1) ? hankel1_int(m, zmax) : hankel2_int(m, zmax);
        G += prefactor * phase * (Jm * Hm);
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
    std::vector<Cplx> kappa_m_cache;
    std::vector<int> hankel_kind_cache; // 1 => H^(1), 2 => H^(2)
    const Cplx prefactor; // 1/(4 i)

    // Cached element data
    R3 x0, y0;
    R3 x, y;
    R3 ny;
    Real h{};
    Cplx ker;

    // ---- argument-derivative via complex_bessel derivatives (more stable than m±1 recurrences) ----
    static inline Cplx besselJ_prime_int(const int m, const Cplx& z) { return sp_bessel::besselJp(m, z, 1); }
    static inline Cplx hankel1_prime_int(const int m, const Cplx& z) { return sp_bessel::hankelH1p(m, z, 1); }
    static inline Cplx hankel2_prime_int(const int m, const Cplx& z) { return sp_bessel::hankelH2p(m, z, 1); }

  public:
    BIOpKernel(const typename Trait::MeshX& mx,
               const typename Trait::MeshY& my,
               const Real& khat_,
               const Real& Omega_ = 0.,
               const Real& c_ = 1.,
               const int& M_ = 20,
               const bool& keep_D2_ = false)
      : meshx(mx), meshy(my),
        normaly(NormalTo(my)),
        phix(mx), phiy(my),
        khat(khat_), Omega(Omega_), c(c_), M(M_), keep_D2(keep_D2_)
      , kappa_m_cache(static_cast<std::size_t>(2 * M_ + 1))
      , hankel_kind_cache(static_cast<std::size_t>(2 * M_ + 1), 1)
      , prefactor(static_cast<Real>(1) / (static_cast<Real>(4) * iu))
    {
      for (int m = -M; m <= M; ++m)
      {
        // See SL kernel for sign convention.
        Cplx kappa2 = +(khat * khat) + static_cast<Real>(2) * khat * static_cast<Real>(m) * (Omega / c);
        if (keep_D2) kappa2 -= (Omega * Omega) * static_cast<Real>(m * m) / (c * c);

        const Cplx kappa_m = kappa_from_kappa2(kappa2);
        const auto idx = static_cast<std::size_t>(m + M);
        kappa_m_cache[idx] = kappa_m;
        const Real imk = std::imag(kappa_m);
        hankel_kind_cache[idx] = (imk < static_cast<Real>(0)) ? 2 : 1;
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
      Real dn_r = (Real)0;
      Real dn_phi = (Real)0;

      constexpr Real eps = (Real)1e-14;
      if (ry > eps)
      {
        dn_r = (ny[0] * y[0] + ny[1] * y[1]) / ry;
        dn_phi = (-y[1] * ny[0] + y[0] * ny[1]) / (ry * ry);
      }

      // piecewise: which branch depends on y?
      const bool y_is_rmin = (ry < rx);
      const Real rmin = y_is_rmin ? ry : rx;
      const Real rmax = y_is_rmin ? rx : ry;

      Cplx dGdn = (Real)0;

      // Phase recurrence (avoid exp() per mode)
      const Cplx e_idphi = std::exp(iu * dphi);
      Cplx phase = std::exp(iu * static_cast<Real>(-M) * dphi);

      const bool rmin_is_zero = (rmin < eps);

      for (int m = -M; m <= M; ++m)
      {
        const auto idx = static_cast<std::size_t>(m + M);
        const Cplx kappa_m = kappa_m_cache[idx];

        const Cplx zmin = kappa_m * rmin;
        const Cplx zmax = kappa_m * rmax;

        // ∂_{n_y} phase = phase * ( - i m ∂_{n_y} phi_y )
        const Cplx dphase = phase * (-static_cast<Real>(m) * dn_phi) * iu;

        // Axis handling for the J-factor at rmin.
        Cplx Jm_min = (Real)0;
        if (rmin_is_zero)
        {
          if (m == 0) Jm_min = (Real)1;
        }
        else
        {
          Jm_min = besselJ_int(m, zmin);
        }

        const bool use_h1 = (hankel_kind_cache[idx] == 1);
        const Cplx Hm_max = use_h1 ? hankel1_int(m, zmax) : hankel2_int(m, zmax);

        // radial part derivative wrt y (only hits the factor that depends on r_y)
        Cplx dJdn = 0;
        Cplx dHdn = 0;

        if (y_is_rmin)
        {
          // rmin = r_y, so differentiate J_m(kappa_m r_y)
          dJdn = kappa_m * besselJ_prime_int(m, zmin) * dn_r;
        }
        else
        {
          // rmax = r_y, so differentiate H_m(kappa_m r_y)
          const Cplx Hp = use_h1 ? hankel1_prime_int(m, zmax) : hankel2_prime_int(m, zmax);
          dHdn = kappa_m * Hp * dn_r;
        }

        // product rule: ∂n (phase * J * H)
        const Cplx dterm = dphase * Jm_min * Hm_max
          + phase * dJdn * Hm_max
          + phase * Jm_min * dHdn;

        dGdn += prefactor * dterm;
        phase *= e_idphi;
      }

      // Match BEMTool DL structure: ker = h * ∂_{n_y}G
      ker = h * dGdn;

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
