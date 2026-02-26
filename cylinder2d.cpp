#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>

#include "bemtool/tools.hpp"

using namespace bemtool;

// plane wave u_inc(x) = exp(i k d·x), d = (cos th, sin th)
static inline Cplx u_inc(const Real& kappa, const Real& th, const R3& x)
{
  Real phase = kappa * (std::cos(th) * x[0] + std::sin(th) * x[1]);
  return exp(iu * phase);
}

static void write_legacy_vtk_polyline_with_sigma(
    const std::string& filename,
    const Dof<P1_1D>& dof,
    const EigenDense::VectorType& sigma)
{
  const int nb_elt = NbElt(dof);
  const int nb_dof = NbDof(dof);

  // gather point coordinates for each global dof
  std::vector<R3> X(nb_dof);
  std::vector<char> seen(nb_dof, 0);

  for(int e=0; e<nb_elt; ++e){
    const N2& edof = dof[e];
    const array<2,R3> xloc = dof(e);
    for(int a=0; a<2; ++a){
      int id = edof[a];
      if(!seen[id]){
        X[id] = xloc[a];
        seen[id] = 1;
      }
    }
  }

  std::ofstream f(filename);
  f << "# vtk DataFile Version 2.0\n";
  f << "BemTool boundary density sigma\n";
  f << "ASCII\n";
  f << "DATASET POLYDATA\n";

  // points
  f << "POINTS " << nb_dof << " double\n";
  for(int i=0;i<nb_dof;++i){
    f << X[i][0] << " " << X[i][1] << " " << X[i][2] << "\n";
  }

  // lines: each element is a segment with 2 point indices
  f << "LINES " << nb_elt << " " << 3*nb_elt << "\n";
  for(int e=0; e<nb_elt; ++e){
    const N2& edof = dof[e];
    f << "2 " << edof[0] << " " << edof[1] << "\n";
  }

  // point data
  f << "POINT_DATA " << nb_dof << "\n";

  f << "SCALARS sigma_re double 1\n";
  f << "LOOKUP_TABLE default\n";
  for(int i=0;i<nb_dof;++i) f << sigma[i].real() << "\n";

  f << "SCALARS sigma_im double 1\n";
  f << "LOOKUP_TABLE default\n";
  for(int i=0;i<nb_dof;++i) f << sigma[i].imag() << "\n";

  f << "SCALARS sigma_abs double 1\n";
  f << "LOOKUP_TABLE default\n";
  for(int i=0;i<nb_dof;++i) f << std::abs(sigma[i]) << "\n";
}

// Hankel H_n^(1)(z) = J_n(z) + i Y_n(z)
static inline Cplx hankel1(int n, Real z){
  using std::cyl_bessel_j;
  using std::cyl_neumann;
  return Cplx(cyl_bessel_j(n, z), cyl_neumann(n, z));
}

// 2D Helmholtz Green's function: G(r) = i/4 H0^(1)(k r)
static inline Cplx G2D(Real k, Real r){
  return (iu * 0.25) * hankel1(0, k * r);
}

// dG/dr = -(i k/4) H1^(1)(k r)
static inline Cplx dGdr2D(Real k, Real r){
  return -(iu * k * 0.25) * hankel1(1, k * r);
}

// A very simple panel quadrature for CFIE potentials at a target x:
// u_scat(x) ≈ Σ_e ∫_e [ ∂_{n_y}G(x,y) - i eta G(x,y) ] φ(y) ds_y
// Here we use midpoint rule and P1 average on each segment.
static Cplx eval_uscat_cfie_midpoint(
    const Mesh1D& mesh,
    const Dof<P1_1D>& dof,
    const EigenDense::VectorType& phi,
    Real kappa,
    Real eta,
    const R3& x)
{
  const int nb_elt = NbElt(mesh);
  Cplx uscat = Cplx(0.,0.);

  for(int e=0; e<nb_elt; ++e){
    const N2& edof = dof[e];
    const array<2,R3> yloc = dof(e);

    // endpoints
    R3 y0 = yloc[0];
    R3 y1 = yloc[1];

    // midpoint and segment length
    R3 ym = 0.5*(y0 + y1);
    R3 t  = (y1 - y0);
    Real ds = std::sqrt(t[0]*t[0] + t[1]*t[1]);

    // density at midpoint (P1 average)
    Cplx phim = 0.5*(phi[edof[0]] + phi[edof[1]]);

    // outward normal (auto-correct using radial test; fine for star-shaped obstacles like a circle)
    // Start with a 90° rotation of tangent
    R3 n; n[2]=0.;
    n[0] =  t[1];
    n[1] = -t[0];
    Real nn = std::sqrt(n[0]*n[0] + n[1]*n[1]);
    if(nn > 0){ n[0]/=nn; n[1]/=nn; }

    // ensure it's outward: for a circle centered at origin, outward normal has positive dot with midpoint
    Real dot = n[0]*ym[0] + n[1]*ym[1];
    if(dot < 0){ n[0] = -n[0]; n[1] = -n[1]; }

    // distance
    R3 rvec = x - ym;
    Real r = std::sqrt(rvec[0]*rvec[0] + rvec[1]*rvec[1]);

    // skip if target exactly on source (shouldn't happen for exterior sampling)
    if(r < 1e-12) continue;

    // kernels at midpoint
    Cplx G  = G2D(kappa, r);

    // ∂_{n_y}G = (i k/4) H1(k r) * ((x-y)·n)/r
    // Using relation from dG/dr and chain rule:
    // ∇_y G · n = -dG/dr * ((x-y)·n)/r
    // (with dG/dr above)
    Real proj = (rvec[0]*n[0] + rvec[1]*n[1]) / r;
    Cplx dGdn = -dGdr2D(kappa, r) * proj;

    // CFIE potential combination
    uscat += (dGdn - iu * eta * G) * phim * ds;
  }

  return uscat;
}

// Legacy VTK structured grid writer (STRUCTURED_POINTS)
static void write_vtk_structured_points(
    const std::string& filename,
    int nx, int ny,
    Real x0, Real y0,
    Real dx, Real dy,
    const std::vector<Real>& data,
    const std::string& fieldname)
{
  std::ofstream f(filename);
  f << "# vtk DataFile Version 2.0\n";
  f << "Scattered field on grid\n";
  f << "ASCII\n";
  f << "DATASET STRUCTURED_POINTS\n";
  f << "DIMENSIONS " << nx << " " << ny << " 1\n";
  f << "ORIGIN " << x0 << " " << y0 << " 0\n";
  f << "SPACING " << dx << " " << dy << " 1\n";
  f << "POINT_DATA " << (nx*ny) << "\n";
  f << "SCALARS " << fieldname << " double 1\n";
  f << "LOOKUP_TABLE default\n";
  for(int i=0;i<nx*ny;++i) f << data[i] << "\n";
}

int main(int argc, char** argv)
{
  // ---------- parameters ----------
  Real kappa = 2.0; // wavenumber
  Real theta_inc = 0.0; // incidence direction angle
  Real eta = kappa;
  std::string mshfile = "mesh/circle.msh"; // same as demo layout

  // ---------- mesh ----------
  Geometry geo(mshfile);
  Mesh1D mesh;
  mesh.Load(geo, 1);
  Orienting(mesh);

  int nb_elt = NbElt(mesh);
  std::cout << "nb_elt:\t" << nb_elt << "\n";

  // ---------- dofs ----------
  Dof<P1_1D> dof(mesh);
  int nb_dof = NbDof(dof);
  std::cout << "nb_dof:\t" << nb_dof << "\n";

  // ---------- operators ----------
  // Single-layer and double-layer boundary operators
  typedef LA_SL_2D_P1xP1 OpSL;
  typedef LA_DL_2D_P1xP1 OpDL;

  BIOp<OpSL> S(mesh, mesh, kappa);
  BIOp<OpDL> D(mesh, mesh, kappa);

  // ---------- assemble CFIE matrix A = 0.5 I + D - i eta S ----------
  EigenDense A(nb_dof, nb_dof);
  Clear(A);

  progress bar("Assembly (CFIE)", nb_dof);
  for (int j = 0; j < nb_dof; ++j)
  {
    bar++;
    for (int k = 0; k < nb_dof; ++k)
    {
      // same global-dof assembly strategy as test2D.cpp :contentReference[oaicite:1]{index=1}
      A(j, k) += D(dof.ToElt(j), dof.ToElt(k));
      A(j, k) -= iu * eta * S(dof.ToElt(j), dof.ToElt(k));
    }
  }
  bar.end();

  // add + 0.5 I
  for (int j = 0; j < nb_dof; ++j)
    A(j, j) += 0.5;

  // ---------- RHS: -u_inc on Gamma ----------
  EigenDense::VectorType g(nb_dof, Cplx(0.0, 0.0));
  for (int e = 0; e < nb_elt; ++e)
  {
    const N2& edof = dof[e];
    const array<2, R3> xloc = dof(e);
    for (int a = 0; a < 2; ++a)
      g[edof[a]] = -u_inc(kappa, theta_inc, xloc[a]);
  }

  // ---------- solve A * phi = g ----------
  EigenDense::VectorType sigma(nb_dof);
  lu_solve(A, g, sigma);

  // ---------- output density ----------
  std::ofstream out("sigma.csv");
  out << "j,Re_sigma,Im_sigma\n";
  for (int j = 0; j < nb_dof; ++j)
  {
    out << j << "," << sigma[j].real() << "," << sigma[j].imag() << "\n";
  }
  out.close();

  write_legacy_vtk_polyline_with_sigma("sigma_boundary.vtk", dof, sigma);


  // --------- estimate cylinder radius a from mesh nodes ----------
  Real a = 0.0;
  for(int e=0; e<nb_elt; ++e){
    const array<2,R3> yloc = dof(e);
    for(int k=0;k<2;++k){
      Real rr = std::sqrt(yloc[k][0]*yloc[k][0] + yloc[k][1]*yloc[k][1]);
      if(rr > a) a = rr;
    }
  }
  std::cout << "estimated radius a = " << a << "\n";

  // --------- sample on a grid ----------
  int nx = 301, ny = 301;
  Real xmin = -2.5, xmax = 2.5;
  Real ymin = -2.5, ymax = 2.5;
  Real dx = (xmax - xmin) / (nx - 1);
  Real dy = (ymax - ymin) / (ny - 1);

  std::vector<Real> utot_abs(nx*ny, 0.0);
  std::vector<Real> uscat_abs(nx*ny, 0.0);

  progress bar2("Assembly (CFIE)", nx*ny);
  for(int j=0; j<ny; ++j){
    Real y = ymin + j*dy;
    for(int i=0; i<nx; ++i){
      bar2++;
      Real x = xmin + i*dx;

      int id = i + nx*j;

      // exclude cylinder interior
      Real r = std::sqrt(x*x + y*y);
      if(r < a){
        utot_abs[id]  = 0.0;   // or use NaN if you prefer
        uscat_abs[id] = 0.0;
        continue;
      }

      R3 X; X[0]=x; X[1]=y; X[2]=0.0;

      // scattered field from CFIE potential
      Cplx uscat = eval_uscat_cfie_midpoint(mesh, dof, sigma, kappa, eta, X);

      // incident + total
      Cplx uinc  = u_inc(kappa, theta_inc, X);
      Cplx utot  = uinc + uscat;

      uscat_abs[id] = std::abs(uscat);
      utot_abs[id]  = std::abs(utot);
    }
  }
  bar2.end();

  // --------- write VTK files ----------
  write_vtk_structured_points("uscat_abs.vtk", nx, ny, xmin, ymin, dx, dy, uscat_abs, "abs_uscat");
  write_vtk_structured_points("utot_abs.vtk",  nx, ny, xmin, ymin, dx, dy, utot_abs,  "abs_utot");

  std::cout << "Wrote uscat_abs.vtk and utot_abs.vtk (open in ParaView)\n";

  return 0;
}
