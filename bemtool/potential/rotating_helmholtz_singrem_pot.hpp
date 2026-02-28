//===================================================================
//
//  Rotating Helmholtz potentials (singular + remainder representation)
//  Analogous to helmholtz_pot.hpp and rotating_helmholtz_singrem_op.hpp
//
//  Provides:
//    PotKernel<RH,SL_POT,2,PhiY>  : single-layer potential kernel in 2D
//    PotKernel<RH,DL_POT,2,PhiY>  : double-layer potential kernel in 2D
//
//  The Green's function is implemented as
//    G_RH = (i/4) H0^{(1)}(khat |x-y|) + sum_{m=-M}^M (i/4) e^{im(Δφ)}
//            [ J_m(kappa_m r_<) H_m^{(kind)}(kappa_m r_>) - J_m(khat r_<) H_m^{(1)}(khat r_>) ]
//
//  and the DL uses the same normal derivative + BemTool sign convention
//  as rotating_helmholtz_singrem_op.hpp (i.e. returns -∂_{n_y}G times ds).
//
//====================================================================
#ifndef BEMTOOL_POTENTIAL_ROTATING_HELMHOLTZ_HPP
#define BEMTOOL_POTENTIAL_ROTATING_HELMHOLTZ_HPP

#include "potential.hpp"

namespace bemtool
{
  // ---- helper copied from rotating_helmholtz_singrem_op.hpp ----
  static inline Cplx kappa_from_kappa2_pot(const Cplx& kappa2)
  {
    // principal sqrt then enforce outgoing branch consistent with operator header
    Cplx k = std::sqrt(kappa2);
    const Real im = std::imag(k);
    const Real re = std::real(k);
    if (im < static_cast<Real>(0) || (im == static_cast<Real>(0) && re < static_cast<Real>(0))) k = -k;
    return k;
  }

  /*=========================================
    SINGLE LAYER ROTATING HELMHOLTZ POTENTIAL
    =========================================
    u(x) = \int_\Gamma G_RH(x,y) \varphi(y) ds_y
  */

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
    const Real khat;
    const Real Omega;
    const Real c;
    const int M;
    const bool keep_D2;

    std::vector<Cplx> kappa_m_cache;
    std::vector<int> hankel_kind_cache; // 1 => H^(1), 2 => H^(2)
    const Cplx prefactor;

    // cached element data
    R3 y0;
    R3 x; // evaluation point (fixed during Assign)
    R3 y;
    Real h{};
    Cplx ker;

    // x polar cache
    Real rx{};
    Real phix_ang{};

  public:
    PotKernel<RH, SL_POT, 2, PhiY>(const typename Trait::MeshY& my,
                                   const Real& khat_,
                                   const Real& Omega_ = 0.,
                                   const Real& c_ = 1.,
                                   const int& M_ = 20,
                                   const bool& keep_D2_ = false)
      : meshy(my)
        , phiy(my)
        , khat(khat_)
        , Omega(Omega_)
        , c(c_)
        , M(M_)
        , keep_D2(keep_D2_)
        , kappa_m_cache(2 * M_ + 1)
        , hankel_kind_cache(2 * M_ + 1, 1)
        , prefactor(iu / static_cast<Real>(4))
    {
      for (int m = -M; m <= M; ++m)
      {
        Cplx kappa2 = +(khat * khat) + static_cast<Real>(2) * khat * static_cast<Real>(m) * (Omega / c);
        if (keep_D2) kappa2 -= (Omega * Omega) * static_cast<Real>(m * m) / (c * c);

        const Cplx kappa_m = kappa_from_kappa2_pot(kappa2);
        const std::size_t idx = static_cast<std::size_t>(m + M);
        kappa_m_cache[idx] = kappa_m;

        const Real imk = std::imag(kappa_m);
        hankel_kind_cache[idx] = (imk < static_cast<Real>(0)) ? 2 : 1;
      }
    }

    void Assign(const R3& x_, const int& iy)
    {
      x = x_;
      rx = std::hypot(x[0], x[1]);
      phix_ang = std::atan2(x[1], x[0]);

      const typename Trait::EltY& ey = meshy[iy];
      y0 = ey[0];
      dy = MatJac(ey);
      h = DetJac(ey);
    }

    const typename Trait::MatType& operator()(const R3& /*x_unused*/, const typename Trait::Rdy& tj)
    {
      // physical y(t)
      y = y0 + dy * tj;

      const Real ry = std::hypot(y[0], y[1]);
      const Real phiy_ang = std::atan2(y[1], y[0]);

      const Real rmin = (rx < ry) ? rx : ry;
      const Real rmax = (rx < ry) ? ry : rx;
      const Real dphi = phix_ang - phiy_ang;

      // direct singular part
      const Real dx_xy = x[0] - y[0];
      const Real dy_xy = x[1] - y[1];
      const Real R = std::hypot(dx_xy, dy_xy);

      const Cplx zR(khat * R, 0);
      Cplx G = prefactor * hankel1_int(0, zR);

      // remainder sum
      const Cplx e_idphi = std::exp(iu * dphi);
      Cplx phase = std::exp(iu * static_cast<Real>(-M) * dphi);

      constexpr Real eps = 1e-14;
      const bool rmin_is_zero = (rmin < eps);

      const Cplx zmin_he(khat * rmin, 0);
      const Cplx zmax_he(khat * rmax, 0);

      for (int m = -M; m <= M; ++m)
      {
        const std::size_t idx = static_cast<std::size_t>(m + M);

        const Cplx kappa_m = kappa_m_cache[idx];
        const Cplx zmin_rot = kappa_m * rmin;
        const Cplx zmax_rot = kappa_m * rmax;

        Cplx Jm_rot = 0;
        if (rmin_is_zero) { if (m == 0) Jm_rot = 1; }
        else { Jm_rot = besselJ_int(m, zmin_rot); }

        const Cplx Hm_rot = (hankel_kind_cache[idx] == 1) ? hankel1_int(m, zmax_rot) : hankel2_int(m, zmax_rot);

        Cplx Jm_he = 0;
        if (rmin_is_zero) { if (m == 0) Jm_he = 1; }
        else { Jm_he = besselJ_int(m, Cplx(khat, 0) * rmin); }

        const Cplx Hm_he = hankel1_int(m, zmax_he);

        G += prefactor * phase * (Jm_rot * Hm_rot - Jm_he * Hm_he);
        phase *= e_idphi;
      }

      ker = h * G;

      for (int k = 0; k < Trait::nb_dof_y; ++k)
        mat(0, k) = ker * phiy(k, tj);
      return mat;
    }

    const Cplx& operator()(const R3& /*x_unused*/, const typename Trait::Rdy& tj, const int& ky)
    {
      // reuse block eval to keep consistent
      operator()(R3{}, tj);
      static Cplx val;
      val = ker * phiy(ky, tj);
      return val;
    }
  };

  using RH_SL_2D_P0 = PotKernel<RH, SL_POT, 2, P0_1D>;
  using RH_SL_2D_P1 = PotKernel<RH, SL_POT, 2, P1_1D>;


  /*=========================================
    DOUBLE LAYER ROTATING HELMHOLTZ POTENTIAL
    =========================================
    u(x) = \int_\Gamma (-\partial_{n_y} G_RH(x,y)) \varphi(y) ds_y
    (sign matches BemTool's DL convention in helmholtz_pot.hpp)
  */

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
    const Real khat;
    const Real Omega;
    const Real c;
    const int M;
    const bool keep_D2;

    std::vector<Cplx> kappa_m_cache;
    std::vector<int> hankel_kind_cache;
    const Cplx prefactor;

    // cached element data
    R3 y0;
    R3 x;
    R3 y;
    R3 ny;
    Real h{};
    Cplx ker;

    // x polar cache
    Real rx{};
    Real phix_ang{};

    // derivative helpers (match singrem_op)
    static inline Cplx besselJ_prime_int(const int m, const Cplx& z) { return sp_bessel::besselJp(m, z, 1); }
    static inline Cplx hankel1_prime_int(const int m, const Cplx& z) { return sp_bessel::hankelH1p(m, z, 1); }
    static inline Cplx hankel2_prime_int(const int m, const Cplx& z) { return sp_bessel::hankelH2p(m, z, 1); }

  public:
    PotKernel<RH, DL_POT, 2, PhiY>(const typename Trait::MeshY& my,
                                   const Real& khat_,
                                   const Real& Omega_ = 0.,
                                   const Real& c_ = 1.,
                                   const int& M_ = 20,
                                   const bool& keep_D2_ = false)
      : meshy(my)
        , normaly(NormalTo(my))
        , phiy(my)
        , khat(khat_)
        , Omega(Omega_)
        , c(c_)
        , M(M_)
        , keep_D2(keep_D2_)
        , kappa_m_cache(2 * M_ + 1)
        , hankel_kind_cache(2 * M_ + 1, 1)
        , prefactor(iu / static_cast<Real>(4))
    {
      for (int m = -M; m <= M; ++m)
      {
        Cplx kappa2 = +(khat * khat) + static_cast<Real>(2) * khat * static_cast<Real>(m) * (Omega / c);
        if (keep_D2) kappa2 -= (Omega * Omega) * static_cast<Real>(m * m) / (c * c);

        const Cplx kappa_m = kappa_from_kappa2_pot(kappa2);
        const std::size_t idx = static_cast<std::size_t>(m + M);
        kappa_m_cache[idx] = kappa_m;

        const Real imk = std::imag(kappa_m);
        hankel_kind_cache[idx] = (imk < static_cast<Real>(0)) ? 2 : 1;
      }
    }

    void Assign(const R3& x_, const int& iy)
    {
      x = x_;
      rx = std::hypot(x[0], x[1]);
      phix_ang = std::atan2(x[1], x[0]);

      const typename Trait::EltY& ey = meshy[iy];
      y0 = ey[0];
      dy = MatJac(ey);
      h = DetJac(ey);
      ny = normaly[iy];
    }

    const typename Trait::MatType& operator()(const R3& /*x_unused*/, const typename Trait::Rdy& tj)
    {
      // y(t)
      y = y0 + dy * tj;

      constexpr Real eps = 1e-14;

      const Real ry = std::hypot(y[0], y[1]);
      const Real phiy_ang = std::atan2(y[1], y[0]);

      const Real dphi = phix_ang - phiy_ang;

      // derivatives of r and phi with respect to normal at y
      Real dn_r = 0;
      if (ry > eps)
        dn_r = (y[0] * ny[0] + y[1] * ny[1]) / ry;

      Real dn_phi = 0;
      if (ry > eps)
        dn_phi = (-y[1] * ny[0] + y[0] * ny[1]) / (ry * ry);

      const bool y_is_rmin = (ry < rx);
      const Real rmin = y_is_rmin ? ry : rx;
      const Real rmax = y_is_rmin ? rx : ry;

      // direct part: d/dn_y of (i/4) H0(khat R)
      const Real dx_xy = x[0] - y[0];
      const Real dy_xy = x[1] - y[1];
      const Real R = std::hypot(dx_xy, dy_xy);

      Real dRdn = 0;
      if (R > eps)
        dRdn = -(dx_xy * ny[0] + dy_xy * ny[1]) / R;

      const Cplx zR(khat * R, 0);
      const Cplx H0p = sp_bessel::hankelH1p(0, zR, 1);
      Cplx dGdn = prefactor * (Cplx(khat, 0) * H0p) * dRdn;

      // remainder sum
      const Cplx e_idphi = std::exp(iu * dphi);
      Cplx phase = std::exp(iu * static_cast<Real>(-M) * dphi);

      const bool rmin_is_zero = (rmin < eps);

      for (int m = -M; m <= M; ++m)
      {
        const std::size_t idx = static_cast<std::size_t>(m + M);
        const Cplx kappa_m = kappa_m_cache[idx];

        const Cplx zmin_rot = kappa_m * rmin;
        const Cplx zmax_rot = kappa_m * rmax;

        const Cplx dphase = phase * (-static_cast<Real>(m) * dn_phi) * iu;

        Cplx Jm_min_rot = 0;
        if (rmin_is_zero) { if (m == 0) Jm_min_rot = 1; }
        else { Jm_min_rot = besselJ_int(m, zmin_rot); }

        const bool use_h1_rot = (hankel_kind_cache[idx] == 1);
        const Cplx Hm_max_rot = use_h1_rot ? hankel1_int(m, zmax_rot) : hankel2_int(m, zmax_rot);

        Cplx dJdn_rot = 0;
        Cplx dHdn_rot = 0;
        if (y_is_rmin)
        {
          dJdn_rot = kappa_m * besselJ_prime_int(m, zmin_rot) * dn_r;
        }
        else
        {
          const Cplx Hp_rot = use_h1_rot ? hankel1_prime_int(m, zmax_rot) : hankel2_prime_int(m, zmax_rot);
          dHdn_rot = kappa_m * Hp_rot * dn_r;
        }

        const Cplx dterm_rot = dphase * Jm_min_rot * Hm_max_rot
          + phase * dJdn_rot * Hm_max_rot
          + phase * Jm_min_rot * dHdn_rot;

        // reference Helmholtz term
        const Cplx kappa_he(khat, 0);
        const Cplx zmin_he = kappa_he * rmin;
        const Cplx zmax_he = kappa_he * rmax;

        Cplx Jm_min_he = 0;
        if (rmin_is_zero) { if (m == 0) Jm_min_he = 1; }
        else { Jm_min_he = besselJ_int(m, zmin_he); }

        const Cplx Hm_max_he = hankel1_int(m, zmax_he);

        Cplx dJdn_he = 0;
        Cplx dHdn_he = 0;
        if (y_is_rmin)
        {
          dJdn_he = kappa_he * besselJ_prime_int(m, zmin_he) * dn_r;
        }
        else
        {
          const Cplx Hp_he = hankel1_prime_int(m, zmax_he);
          dHdn_he = kappa_he * Hp_he * dn_r;
        }

        const Cplx dterm_he = dphase * Jm_min_he * Hm_max_he
          + phase * dJdn_he * Hm_max_he
          + phase * Jm_min_he * dHdn_he;

        dGdn += prefactor * (dterm_rot - dterm_he);
        phase *= e_idphi;
      }

      // BemTool DL convention: kernel is -∂_{n_y}G times ds Jacobian
      ker = -h * dGdn;

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
} // namespace bemtool

#endif
