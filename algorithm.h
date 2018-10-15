#include "create_csr.h"

int bicgstab_solve(CSRMatrix A, Vector BB, Vector XX, double tol, int maxit, int info);
Vector make_vector(int size);
void SpMV(CSRMatrix A, Vector X, Vector Y);
void free_vector(Vector *X);
double dot(Vector X, Vector Y);
void axpby(Vector X, Vector Y, double a, double b);