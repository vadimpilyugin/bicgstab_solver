#include "stdio.h"
#include "math.h"
#include "assert.h"
#include "stdlib.h"

#include "algorithm.h"

void SpMV(CSRMatrix A, Vector X, Vector Y) {
  assert(A.N == X.size);
  assert(A.N == Y.size);

  int i;
  for (i = 0; i < A.N; i++) {
    double s = 0;
    int j;

    for (j = A.IA[i]; j < A.IA[i+1]; j++) {
      s += X.data[A.JA[j]] * A.A[j];
    }
    Y.data[i] = s;
  }
}

static double max(double a1, double a2) {
  return a1 > a2 ? a1 : a2;
}

void axpby(Vector X, Vector Y, double a, double b) {
  assert(X.size == Y.size);
  int i;
  for (i = 0; i < X.size; i++) {
    X.data[i] = a*X.data[i] + b*Y.data[i];
  }
}

double dot(Vector X, Vector Y) {
  assert(X.size == Y.size);

  double s = 0;
  int i;
  for (i = 0; i < X.size; i++) {
    s += X.data[i] * Y.data[i];
  }
  return s;
}

void fill_with_constant(Vector X, const int c) {
  int i;
  for (i = 0; i < X.size; i++) {
    X.data[i] = c;
  }
}

// X = Y
void assign_vector(Vector X, Vector Y) {
  assert(X.size == Y.size);

  int i;
  for (i = 0; i < X.size; i++)
    X.data[i] = Y.data[i];
}

int bicgstab_solve(CSRMatrix A, Vector BB, Vector XX, double tol, int maxit, int info) {
  double Rhoi_1 = 1.0;
  double alphai = 1.0;
  double wi = 1.0;
  double betai_1 = 1.0;
  double Rhoi_2 = 1.0;
  double alphai_1 = 1.0;
  double wi_1 = 1.0;
  double RhoMin = 1E-60;
  double mineps = 1E-15;

  CSRMatrix DD = csr_diag_inverse(A);
  Vector RR   = make_vector(A.N);
  Vector RR2  = make_vector(A.N);
  Vector PP   = make_vector(A.N);
  Vector PP2  = make_vector(A.N);
  Vector VV   = make_vector(A.N);
  Vector SS   = make_vector(A.N);
  Vector SS2  = make_vector(A.N);
  Vector TT   = make_vector(A.N);

  fill_with_constant(XX, 0);
  assign_vector(RR, BB);
  assign_vector(RR2, BB);
  assign_vector(PP, RR);

  double initres = sqrt(dot(RR,RR));
  double eps = max(mineps, tol*initres);
  double res = initres;
  int i;

  for (i = 0; i < maxit; i++) {
    if (info) {
      printf("It %d: res = %e tol=%e\n", i, res, res / initres); 
    }
    if (res < eps)
      break; 
    if (res > initres / mineps)
      return -1; 
    if (i == 0)
      Rhoi_1 = initres * initres;
    else
      Rhoi_1 = dot(RR2, RR);
    if (fabs(Rhoi_1) < RhoMin)
      return -1;
    if (i > 0) {
      betai_1 = (Rhoi_1 * alphai_1) / (Rhoi_2 * wi_1); 
      axpby(PP, RR, betai_1, 1.0);
      axpby(PP, VV, 1.0, -wi_1 * betai_1); 
    }
    SpMV(DD, PP, PP2); 
    SpMV(A, PP2, VV);  
    alphai = dot(RR2, VV); 
    if (fabs(alphai) < RhoMin)
      return -3;
    alphai = Rhoi_1 / alphai;
    assign_vector(SS, RR);
    axpby(SS, VV, 1.0, -alphai);
    SpMV(DD, SS, SS2); 
    SpMV(A, SS2, TT);
    wi = dot(TT, TT);
    if (fabs(wi) < RhoMin)
      return -4; 
    wi = dot(TT, SS) / wi;  
    if (fabs(wi) < RhoMin)
      return -5;
    axpby(XX, PP2, 1.0, alphai); 
    axpby(XX, SS2, 1.0, wi);
    assign_vector(RR, SS);
    axpby(RR, TT, 1.0, -wi); 
    alphai_1 = alphai;  
    Rhoi_2 = Rhoi_1;  
    wi_1 = wi; 
    res = sqrt(dot(RR,RR)); 
  }
  if (info)
    printf("Solver_BiCGSTAB: outres: %g\n", res);
  free_vector(&RR);
  free_vector(&RR2);
  free_vector(&PP);
  free_vector(&PP2);
  free_vector(&VV);
  free_vector(&SS);
  free_vector(&SS2);
  free_vector(&TT);
  csr_free(&DD);
  return i; 
}