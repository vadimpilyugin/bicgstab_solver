#include "stdlib.h"
#include "assert.h"
#include "math.h"
#include "stdio.h"

#include "create_csr.h"

Vector make_vector(int size) {
  Vector V;
  V.size = size;
  V.data = (double *)malloc(size * sizeof(double));
  return V;
}

void free_vector(Vector *X) {
  free(X -> data);
  X -> size = 0;
  X -> data = NULL;
}

static void fill_neighbours(int nx, int ny, int nz, int i, 
  int neighbours[MAX_NEIGHBOURS], int max_values[N_DIMENSIONS]) {

  neighbours[0] = nz == 0 ? -1 : i - max_values[X_DIM] * max_values[Y_DIM];
  neighbours[1] = ny == 0 ? -1 : i - max_values[X_DIM];
  neighbours[2] = nx == 0 ? -1 : i - 1;
  neighbours[3] = i;
  neighbours[4] = nx == max_values[X_DIM] - 1 ? -1 : i + 1;
  neighbours[5] = ny == max_values[Y_DIM] - 1 ? -1 : i + max_values[X_DIM];
  neighbours[6] = nz == max_values[Z_DIM] - 1 ? -1 : i + max_values[X_DIM] * max_values[Y_DIM];
}

static void csr_init_malloc(CSRMatrix *A, int N, int *curr_cap) {
  A -> IA = (int *)malloc((N + 1) * sizeof(int));
  A -> JA = (int *)malloc(DEFAULT_SIZE * sizeof(int));
  A -> A = (double *)malloc(DEFAULT_SIZE * sizeof(double));
  *curr_cap = DEFAULT_SIZE;
}

static void csr_realloc(CSRMatrix *A, int *curr_cap) {
  *curr_cap *= *curr_cap;

  void *tmp = (int *)realloc(A -> JA, (*curr_cap) * sizeof(int));
  assert(tmp != NULL);
  A -> JA = tmp;

  tmp = (double *)realloc(A -> A, (*curr_cap) * sizeof(double));
  assert(tmp != NULL);
  A -> A = tmp;
}

void csr_free(CSRMatrix *A) {
  free(A -> IA);
  free(A -> JA);
  free(A -> A);

  A -> N = 0;
  A -> IA = NULL;
  A -> JA = NULL;
  A -> A = NULL;
}

CSRMatrix csr_matrix(int max_values[N_DIMENSIONS]) {
  int N = 1; // matrix size
  int i;
  for (i = 0; i < N_DIMENSIONS; i++) {
    N *= max_values[i];
  }
  CSRMatrix A;
  A.N = N;
  int curr_cap;
  csr_init_malloc(&A, N, &curr_cap);
  int M = 0; // number of non-zero elements in the matrix

  // coordinates inside a grid
  int nx = 0;
  int ny = 0;
  int nz = 0;

  for (i = 0; i < N; i++) {
    A.IA[i] = M;
    // printf("After A.IA[i] = M\n");

    int neighbours[MAX_NEIGHBOURS];
    fill_neighbours(nx,ny,nz,i,neighbours,max_values);
    // printf("After fill neighbours\n");

    int j;
    // for (j = 0; j < MAX_NEIGHBOURS; j++) {
    //   printf("neighbours[%d] = %d\n", j, neighbours[j]);
    // }

    int middle = -1;
    double s = 0;
    for (j = 0; j < MAX_NEIGHBOURS; j++) {
      // printf("Next j: %d\n", j);
      // printf("neighbours[%d] = %d\n", j, neighbours[j]);
      if (neighbours[j] >= 0) {
        // printf("j >= 0: %d\n", j);
        if (M == curr_cap) {
          csr_realloc(&A, &curr_cap);
        }
        A.JA[M] = neighbours[j];
        if (i == neighbours[j]) {
          A.A[M] = 0;
          middle = M;
        } else {
          double sin_ij = sin(i+neighbours[j]+1);
          A.A[M] = sin_ij;
          s += fabs(sin_ij);
        }
        M += 1;
      }
    }
    assert(middle != -1);
    A.A[middle] = 1.1*s;

    nx += 1;
    if (nx == max_values[X_DIM]) {
      nx = 0;
      ny += 1;
    }
    if (ny == max_values[Y_DIM]) {
      ny = 0;
      nz += 1;
    }
    if (i < N-1) {
      assert(nz < max_values[Z_DIM]);
    }
  }
  A.IA[N] = M;
  return A;
}

static void csr_fixed_malloc(CSRMatrix *A, int N, int M) {
  A -> IA = (int *)malloc((N + 1) * sizeof(int));
  A -> JA = (int *)malloc(M * sizeof(int));
  A -> A = (double *)malloc(M * sizeof(double));
  A -> N = N;
  A -> IA[N] = M; // M non-zero elements in A
}

void print_csr(CSRMatrix A) {
  int i;
  printf("\nIA: [");
  for (i = 0; i < A.N; i++) {
    printf("%d, ", A.IA[i]);
  }
  printf("]\n");
  printf("JA: [");
  for (i = 0; i < A.IA[A.N]; i++) {
    printf("%d, ", A.JA[i]);
  }
  printf("]\n");
  printf("A: [");
  for (i = 0; i < A.IA[A.N]; i++) {
    printf("%lf, ", A.A[i]);
  }
  printf("]\n\n");
}

CSRMatrix csr_diag_inverse(CSRMatrix A) {
  CSRMatrix D;
  csr_fixed_malloc(&D, A.N, A.N); // Matrix of size N with N non-zero elements
  int i;
  for (i = 0; i < A.N; i++) {
    D.IA[i] = i;
    D.JA[i] = i;
    int j;
    for (j = A.IA[i]; j < A.IA[i+1]; j++) {
      if (A.JA[j] == i) {
        D.A[i] = 1/A.A[j];
      }
    }
  }
  return D;
}
