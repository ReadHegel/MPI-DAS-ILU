
#ifndef ILU_H
#define ILU_H

struct ILUFact;

struct ILUFact* ILU_factorize(int N, int nnz, int* row, int* col, double* val);

void ILU_solve(struct ILUFact* ilu, double* b, double* res);

void ILU_multiply(struct ILUFact* ilu, double* b, double* res);

void ILU_free(struct ILUFact* ilu);

#endif //ILU_H
