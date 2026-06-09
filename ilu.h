
#ifndef ILU_H
#define ILU_H

struct ILUFact;

struct ILUFact* ILU_factorize(int N, int nnz, const int* row, const int* col, const double* val);

void ILU_solve(struct ILUFact* ilu, const double* b, double* res);

void ILU_multiply(struct ILUFact* ilu, const double* b, double* res);

// Independent L*U apply (uses global row index for the L/U split).
void ILU_multiply_reference(struct ILUFact* ilu, const double* b, double* res);

// One column of L*U via dense factors assembled from CSR (rank 0, p=1 only).
void ILU_dense_naive_lu_column(
    struct ILUFact* ilu, int N, int col_j, double* col_out
);

void ILU_free(struct ILUFact* ilu);

#endif //ILU_H
