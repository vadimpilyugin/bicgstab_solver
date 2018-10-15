#include "stdio.h"
#include "stdlib.h"
#include "math.h"

#include "create_csr.h"
#include "algorithm.h"
#include "algorithm_par.h"

void my_assert (int expr, const char *s) {
  if (!expr) {
    fprintf(stderr, "%s\n", s);
    exit(1);
  }
}

void test_create_csr() {

  int max_values[N_DIMENSIONS] = {2,2,2};
  CSRMatrix m = csr_matrix(max_values);

  int IA[] = {0, 4, 8, 12, 16, 20, 24, 28, 32};
  int JA[] = {0, 1, 2, 4, 0, 1, 3, 5, 0, 2, 3, 6, 1, 2, 3, 7, 0, 4, 5, 6, 1, 4, 5, 7, 2, 4, 6, 7, 3, 5, 6, 7};
  double A[] = {2.2102758805035565, 0.9092974268256817, 0.1411200080598672, -0.9589242746631385, 0.9092974268256817, 2.77772913022837, -0.9589242746631385, 0.6569865987187891, 0.1411200080598672, 0.9159193906506048, -0.27941549819892586, 0.4121184852417566, -0.9589242746631385, -0.27941549819892586, 2.462162977354045, -0.9999902065507035, -0.9589242746631385, 2.753229151313533, -0.5440211108893698, -0.9999902065507035, 0.6569865987187891, -0.5440211108893698, 1.7832922210782798, 0.4201670368266409, 0.4121184852417566, -0.9999902065507035, 2.6429876522360636, 0.9906073556948704, -0.9999902065507035, 0.4201670368266409, 0.9906073556948704, 2.6518410589794366};
  double eps = 1e-7;
  int i;

  print_csr(m);

  for (i = 0; i < sizeof(IA)/sizeof(int); i++) {
    my_assert(IA[i] == m.IA[i], "IA are not equal!");
  }

  for (i = 0; i < sizeof(JA)/sizeof(int); i++) {
    my_assert(JA[i] == m.JA[i], "JA are not equal!");
  }

  for (i = 0; i < sizeof(A)/sizeof(double); i++) {
    my_assert(fabs(m.A[i] - A[i]) < eps, "A's are not equal!");
  }

  my_assert(m.N == 8, "Dimension is incorrect!");
  csr_free(&m);

  printf("%s\n", "----------------\n");
}

void test_csr_inverse() {

  int IA[] = {0, 3, 6, 9};
  int JA[] = {0, 1, 2, 0, 1, 2, 0, 1, 2};
  double AA[] = {1, 2, 3, 15, 5, 6, 7, 13, 9};
  int N = 3;

  CSRMatrix A;
  A.IA = IA;
  A.JA = JA;
  A.A = AA;
  A.N = N;
  CSRMatrix D = csr_diag_inverse(A);

  int ID[] = {0, 1, 2, 3};
  int JD[] = {0, 1, 2};
  double DD[] = {1.0, 0.2, 0.11111111};
  int ND = 3;
  double eps = 1e-7;
  int i;

  print_csr(D);

  for (i = 0; i < sizeof(ID)/sizeof(int); i++) {
    my_assert(ID[i] == D.IA[i], "ID are not equal!");
  }

  for (i = 0; i < sizeof(JD)/sizeof(int); i++) {
    my_assert(JD[i] == D.JA[i], "JD are not equal!");
  }

  for (i = 0; i < sizeof(DD)/sizeof(double); i++) {
    printf("%lf vs %lf\n", D.A[i], DD[i]);
    my_assert(fabs(D.A[i] - DD[i]) < eps, "DD's are not equal!");
  }

  my_assert(D.N == ND, "Dimension is incorrect!");

  csr_free(&D);

  printf("%s\n", "----------------\n");
}

void test_solve() {

  int IA[] = {0, 3, 6, 9};
  int JA[] = {0, 1, 2, 0, 1, 2, 0, 1, 2};
  double AA[] = {1, 2, 3, 15, 5, 6, 7, 13, 9};
  int N = 3;

  CSRMatrix A;
  A.IA = IA;
  A.JA = JA;
  A.A = AA;
  A.N = N;

  double b[] = {11.0,62.0,52.0};
  Vector BB;
  BB.size = sizeof(b) / sizeof(double);
  BB.data = b;

  double x[] = {1,1,1};
  Vector XX;
  XX.size = sizeof(x) / sizeof(double);
  XX.data = x;

  double eps = 1e-2;
  double tol = eps;
  int maxit = 50;
  int info = 1;

  bicgstab_solve(A, BB, XX, tol, maxit, info);

  double solution[] = {3,1,2};
  int i;

  for (i = 0; i < sizeof(solution) / sizeof(double); i++) {
    printf("%lf vs %lf\n", solution[i], XX.data[i]);
    my_assert(fabs(solution[i] - XX.data[i]) < eps, "Not equal to solution!");
  }

  printf("%s\n", "----------------\n");

}

void test_big_slau() {
  int max_values[] = {100, 100, 100};
  int N = 1;
  int i;
  for (i = 0; i < sizeof(max_values) / sizeof(int); i++) {
    N *= max_values[i];
  }
  printf("N = %d\n", N);

  CSRMatrix A = csr_matrix(max_values);
  Vector BB = make_vector(N);
  Vector XX = make_vector(N);

  for (i = 0; i < N; i++) {
    BB.data[i] = sin(i + 1);
  }

  double eps = 1e-4;
  double tol = eps*eps;
  int maxit = 50;
  int info = 1;

  bicgstab_solve(A, BB, XX, tol, maxit, info);
  Vector Y = make_vector(N);
  SpMV(A, XX, Y);

  for (i = 0; i < N; i++) {
    // printf("%lf vs %lf\n", Y.data[i], BB.data[i]);
    my_assert(fabs(Y.data[i] - BB.data[i]) < eps, "Solution is incorrect!");
  }

  csr_free(&A);
  free_vector(&BB);
  free_vector(&XX);
  free_vector(&Y);

  printf("%s\n", "----------------\n");
}

void test_dot() {
  int N = 1000;
  int i;

  Vector XX = make_vector(N);
  Vector YY = make_vector(N);

  for (i = 0; i < N; i++) {
    XX.data[i] = sin(i + 1);
    YY.data[i] = cos(i + 1);
  }

  double d = dot(XX, YY);
  printf("dot(XX, YY) for X = sin(i), Y = cos(i), N = %d: %lf\n", N, d);
  printf("True ans: 0.452019\n");

  free_vector(&XX);
  free_vector(&YY);

  printf("%s\n", "----------------\n");
}

void test_axpby() {
  int N = 100;
  int i;

  Vector XX = make_vector(N);
  Vector YY = make_vector(N);
  for (i = 0; i < N; i++) {
    XX.data[i] = sin(i + 1);
    YY.data[i] = cos(i + 1);
  }

  axpby(XX, YY, 1.0, -1.0);
  double s = 0;
  for (i = 0; i < N; i++) {
    s += XX.data[i];
  }

  printf("axpby(XX, YY, a, b) = %lf\n", s);
  printf("True ans: 0.405118\n");

  free_vector(&XX);
  free_vector(&YY);

  printf("%s\n", "----------------\n");
}

void test_spmv() {
  int max_values[] = {10, 10, 10};
  int N = 1;
  int i;
  for (i = 0; i < sizeof(max_values) / sizeof(int); i++) {
    N *= max_values[i];
  }

  CSRMatrix A = csr_matrix(max_values);

  Vector XX = make_vector(N);
  Vector YY = make_vector(N);
  for (i = 0; i < N; i++) {
    XX.data[i] = sin(i + 1);
  }
  SpMV(A, XX, YY);

  double s = 0;
  for (i = 0; i < N; i++) {
    s += YY.data[i];
  }
  printf("spMV(A, XX, YY) = %lf\n", s);

  free_vector(&XX);
  free_vector(&YY);
  csr_free(&A);

  printf("%s\n", "----------------\n");
}

void test_big_slau_par() {
  int max_values[] = {100, 100, 100};
  int N = 1;
  int i;
  for (i = 0; i < sizeof(max_values) / sizeof(int); i++) {
    N *= max_values[i];
  }
  printf("N = %d\n", N);

  CSRMatrix A = csr_matrix(max_values);
  Vector BB = make_vector(N);
  Vector XX = make_vector(N);

  for (i = 0; i < N; i++) {
    BB.data[i] = sin(i + 1);
  }

  double eps = 1e-4;
  double tol = eps*eps;
  int maxit = 50;
  int info = 1;

  bicgstab_solve_par(A, BB, XX, tol, maxit, info);
  Vector Y = make_vector(N);
  SpMV_par(A, XX, Y);

  for (i = 0; i < N; i++) {
    // printf("%lf vs %lf\n", Y.data[i], BB.data[i]);
    my_assert(fabs(Y.data[i] - BB.data[i]) < eps, "Solution is incorrect!");
  }

  csr_free(&A);
  free_vector(&BB);
  free_vector(&XX);
  free_vector(&Y);

  printf("%s\n", "----------------\n");
}

void test_dot_par() {
  int N = 1000;
  int i;

  Vector XX = make_vector(N);
  Vector YY = make_vector(N);

  for (i = 0; i < N; i++) {
    XX.data[i] = sin(i + 1);
    YY.data[i] = cos(i + 1);
  }

  double d = dot_par(XX, YY);
  printf("dot_par(XX, YY) for X = sin(i), Y = cos(i), N = %d: %lf\n", N, d);
  printf("True ans: 0.452019\n");

  free_vector(&XX);
  free_vector(&YY);

  printf("%s\n", "----------------\n");
}

void test_axpby_par() {
  int N = 100;
  int i;

  Vector XX = make_vector(N);
  Vector YY = make_vector(N);
  for (i = 0; i < N; i++) {
    XX.data[i] = sin(i + 1);
    YY.data[i] = cos(i + 1);
  }

  axpby_par(XX, YY, 1.0, -1.0);
  double s = 0;
  for (i = 0; i < N; i++) {
    s += XX.data[i];
  }

  printf("axpby_par(XX, YY, a, b) = %lf\n", s);
  printf("True ans: 0.405118\n");

  free_vector(&XX);
  free_vector(&YY);

  printf("%s\n", "----------------\n");
}

void test_spmv_par() {
  int max_values[] = {10, 10, 10};
  int N = 1;
  int i;
  for (i = 0; i < sizeof(max_values) / sizeof(int); i++) {
    N *= max_values[i];
  }

  CSRMatrix A = csr_matrix(max_values);

  Vector XX = make_vector(N);
  Vector YY = make_vector(N);
  for (i = 0; i < N; i++) {
    XX.data[i] = sin(i + 1);
  }
  SpMV_par(A, XX, YY);

  double s = 0;
  for (i = 0; i < N; i++) {
    s += YY.data[i];
  }
  printf("spMV_par(A, XX, YY) = %lf\n", s);

  free_vector(&XX);
  free_vector(&YY);
  csr_free(&A);

  printf("%s\n", "----------------\n");
}

int main() {
  test_create_csr();
  test_csr_inverse();
  test_solve();
  test_dot();
  test_dot_par();
  test_axpby();
  test_axpby_par();
  test_spmv();
  test_spmv_par();
  test_big_slau();
  test_big_slau_par();
  return 0;
}