// rotating_helmholtz.hpp
#ifndef BEMTOOL_OPERATOR_ROTATING_HELMHOLTZ_HPP
#define BEMTOOL_OPERATOR_ROTATING_HELMHOLTZ_HPP

#include "operator.hpp"   // BIOpKernelTraits, R3, MatJac, DetJac, iu, Real, Cplx, etc.
#include <complex>
#include <cmath>

// You need *complex* Bessel J_m and Y_m.
// Option A (quick): boost::math (works for complex arguments)
// Option B (robust): AMOS zbesj/zbesh wrappers (recommended long-term)
#include <boost/math/special_functions/bessel.hpp>
// --- replace Boost includes + helpers with AMOS-based helpers ---

#include <complex>
#include <cmath>

// AMOS Fortran entry points (names may need trailing underscore depending on your toolchain)
extern "C" {

// ZBESJ: J_{FNU+k-1}(Z), k=1..N
void zbesj_(const double* zr, const double* zi,
            const double* fnu, const int* kode, const int* n,
            double* cyr, double* cyi, int* nz, int* ierr);

// ZBESY: Y_{FNU+k-1}(Z), k=1..N  (needs work arrays)
void zbesy_(const double* zr, const double* zi,
            const double* fnu, const int* kode, const int* n,
            double* cyr, double* cyi, int* nz,
            double* cwrkr, double* cwrki, int* ierr);

// ZBESH: H^{(m)}_{FNU+k-1}(Z), m=1 or 2, k=1..N (needs work arrays)
void zbesh_(const double* zr, const double* zi,
            const double* fnu, const int* kode, const int* m, const int* n,
            double* cyr, double* cyi, int* nz,
            int* ierr);
}

namespace bemtool
{
  inline Cplx amos_J_int(int m, const Cplx& z)
  {
    const int norder = (m < 0) ? -m : m;
    const double fnu = static_cast<double>(norder);
    const int kode = 1;
    const int n = 1;

    double cyr = 0.0, cyi = 0.0;
    int nz = 0, ierr = 0;
    const double zr = std::real(z), zi = std::imag(z);

    zbesj_(&zr, &zi, &fnu, &kode, &n, &cyr, &cyi, &nz, &ierr);
    Cplx val(cyr, cyi);

    // J_{-n} = (-1)^n J_n
    if (m < 0 && (norder & 1)) val = -val;

    // Optional: handle ierr != 0 (throw, assert, or fallback)
    return val;
  }

  inline Cplx amos_Y_int(int m, const Cplx& z)
  {
    const int norder = (m < 0) ? -m : m;
    const double fnu = static_cast<double>(norder);
    const int kode = 1;
    const int n = 1;

    double cyr = 0.0, cyi = 0.0;
    int nz = 0, ierr = 0;
    double cwrkr = 0.0, cwrki = 0.0;
    const double zr = std::real(z), zi = std::imag(z);

    zbesy_(&zr, &zi, &fnu, &kode, &n, &cyr, &cyi, &nz, &cwrkr, &cwrki, &ierr);
    Cplx val(cyr, cyi);

    // Y_{-n} = (-1)^n Y_n   (integer n)
    if (m < 0 && (norder & 1)) val = -val;

    return val;
  }

  inline Cplx amos_H1_int(int m, const Cplx& z)
  {
    const int norder = (m < 0) ? -m : m;
    const double fnu = static_cast<double>(norder);
    const int kode = 1;
    const int hankel_kind = 1; // 1 => H^(1)
    const int n = 1;

    double cyr = 0.0, cyi = 0.0;
    int nz = 0, ierr = 0;
    const double zr = std::real(z), zi = std::imag(z);

    zbesh_(&zr, &zi, &fnu, &kode, &hankel_kind, &n, &cyr, &cyi, &nz, &ierr);
    Cplx val(cyr, cyi);

    // H_{-n}^{(1)} = (-1)^n H_n^{(1)}  (integer n)
    if (m < 0 && (norder & 1)) val = -val;

    return val;
  }

  inline Cplx amos_H2_int(int m, const Cplx& z)
  {
    const int norder = (m < 0) ? -m : m;
    const double fnu = static_cast<double>(norder);
    const int kode = 1;
    const int hankel_kind = 2; // 2 => H^(2)
    const int n = 1;

    double cyr = 0.0, cyi = 0.0;
    int nz = 0, ierr = 0;
    const double zr = std::real(z), zi = std::imag(z);

    zbesh_(&zr, &zi, &fnu, &kode, &hankel_kind, &n, &cyr, &cyi, &nz, &ierr);
    Cplx val(cyr, cyi);

    if (m < 0 && (norder & 1)) val = -val;
    return val;
  }

  // Now bind your existing helper names to AMOS:
  inline Cplx besselJ_int(const int m, const Cplx& z) { return amos_J_int(m, z); }
  inline Cplx besselY_int(const int m, const Cplx& z) { return amos_Y_int(m, z); }
  inline Cplx hankel1_int(const int m, const Cplx& z) { return amos_H1_int(m, z); }
  inline Cplx hankel2_int(const int m, const Cplx& z) { return amos_H2_int(m, z); }

  inline Cplx hankel_rad_int(const int m, const Cplx& z, const Cplx& kappa_m)
  {
    // matches your thesis discussion around choosing H^(1)/H^(2) by Im{kappa_m} :contentReference[oaicite:6]{index=6}
    const Real imk = std::imag(kappa_m);
    if (imk > (Real)0) return hankel1_int(m, z);
    if (imk < (Real)0) return hankel2_int(m, z);
    return hankel1_int(m, z); // Im=0: outgoing convention
  }

  // A minimal sqrt helper: only “fix” the purely-real negative case (avoid Re<0 when Im~0).
  inline Cplx kappa_from_kappa2(const Cplx& kappa2)
  {
    Cplx k = std::sqrt(kappa2);
    constexpr Real eps = (Real)1e-14;
    if (std::abs(std::imag(k)) < eps && std::real(k) < (Real)0) k = -k;
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
      Cplx G = (Real)0;

      for (int m = -M; m <= M; ++m)
      {
        // kappa_m^2 formula (toggle keep_D2 as needed)
        // kappa_m^2 = -khat^2 - 2*khat*m*(Omega/c)  [and optionally -Omega^2 m^2 / c^2]
        Cplx kappa2 = -(khat * khat) - (Real)2 * khat * (Real)m * (Omega / c);
        if (keep_D2)
        {
          kappa2 -= (Omega * Omega) * (Real)(m * m) / (c * c);
        }

        const Cplx kappa_m = kappa_from_kappa2(kappa2);

        const Cplx zmin = kappa_m * rmin;
        const Cplx zmax = kappa_m * rmax;

        const Cplx Jm = besselJ_int(m, zmin);
        const Cplx Hm = hankel_rad_int(m, zmax, kappa_m);

        const Cplx phase = std::exp(iu * (Real)m * dphi);

        // prefactor 1/(4i)
        G += ((Real)1 / ((Real)4 * iu)) * phase * (Jm * Hm);
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
      for (int m = -M; m <= M; ++m)
      {
        Cplx kappa2 = -(khat * khat) - (Real)2 * khat * (Real)m * (Omega / c);
        if (keep_D2) kappa2 -= (Omega * Omega) * (Real)(m * m) / (c * c);

        const Cplx kappa_m = kappa_from_kappa2(kappa2);
        const Cplx zmin = kappa_m * rmin;
        const Cplx zmax = kappa_m * rmax;

        const Cplx Jm = besselJ_int(m, zmin);
        const Cplx Hm = hankel_rad_int(m, zmax, kappa_m);
        const Cplx phase = std::exp(iu * (Real)m * dphi);

        G += phase * (Jm * Hm) * ((Real)1 / ((Real)4 * iu));
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

    // Cached element data
    R3 x0, y0;
    R3 x, y;
    R3 ny;
    Real h;
    Cplx ker;

    // ---- argument-derivative recurrences ----
    inline Cplx besselJ_prime_int(const int m, const Cplx& z) const
    {
      return (Real)0.5 * (besselJ_int(m - 1, z) - besselJ_int(m + 1, z));
    }

    inline Cplx hankel1_prime_int(const int m, const Cplx& z) const
    {
      return (Real)0.5 * (hankel1_int(m - 1, z) - hankel1_int(m + 1, z));
    }

    inline Cplx hankel2_prime_int(const int m, const Cplx& z) const
    {
      return (Real)0.5 * (hankel2_int(m - 1, z) - hankel2_int(m + 1, z));
    }

    inline Cplx hankel_rad_prime_int(const int m, const Cplx& z, const Cplx& kappa_m) const
    {
      const Real imk = std::imag(kappa_m);
      if (imk > (Real)0) return hankel1_prime_int(m, z);
      if (imk < (Real)0) return hankel2_prime_int(m, z);
      return hankel1_prime_int(m, z);
    }

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

      for (int m = -M; m <= M; ++m)
      {
        // kappa_m^2 (your thesis formula; toggle keep_D2)
        Cplx kappa2 = -(khat * khat) - (Real)2 * khat * (Real)m * (Omega / c);
        if (keep_D2) kappa2 -= (Omega * Omega) * (Real)(m * m) / (c * c);

        const Cplx kappa_m = kappa_from_kappa2(kappa2);

        const Cplx zmin = kappa_m * rmin;
        const Cplx zmax = kappa_m * rmax;

        const Cplx phase = std::exp(iu * (Real)m * dphi);

        // ∂_{n_y} phase = phase * ∂_{n_y}( i m (phi_x - phi_y) )
        //               = phase * ( - i m ∂_{n_y} phi_y )
        const Cplx dphase = phase * (-(Real)m * dn_phi) * iu;

        const Cplx Jm_min = besselJ_int(m, zmin);
        const Cplx Hm_max = hankel_rad_int(m, zmax, kappa_m);

        // radial part derivative wrt y (only hits the factor that depends on r_y)
        Cplx dJdn = (Real)0;
        Cplx dHdn = (Real)0;

        if (y_is_rmin)
        {
          // rmin = r_y, so differentiate J_m(kappa_m r_y)
          dJdn = kappa_m * besselJ_prime_int(m, zmin) * dn_r;
        }
        else
        {
          // rmax = r_y, so differentiate H_m(kappa_m r_y)
          dHdn = kappa_m * hankel_rad_prime_int(m, zmax, kappa_m) * dn_r;
        }

        // product rule: ∂n (phase * J * H)
        const Cplx dterm = dphase * Jm_min * Hm_max
          + phase * dJdn * Hm_max
          + phase * Jm_min * dHdn;

        // prefactor 1/(4i)
        dGdn += dterm * ((Real)1 / ((Real)4 * iu));
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
