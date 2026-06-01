#include <cstdio>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <mpi.h>
#include <iostream>
#include <unistd.h>
#include "ilu.h"

using namespace std;

#define N_TESTS 10

#define EPS 10e-4

int test_rank_first_row(int rank, int N, int world_size)
{
    return rank * N / world_size;
}

// Reads matrix in a MatrixMarket format
void read_matrix(char* in_file, int* N, int* nnz, int** row, int** col, double** val)
{
    FILE *fp = fopen(in_file, "r");
    char * line = NULL;
    size_t len = 0;
    ssize_t read;
    int l = -1;
    if (fp == NULL)
        exit(EXIT_FAILURE);

    while ((read = getline(&line, &len, fp)) != -1)
    {
        if (line[0] == '%') continue;
        if (l == -1)
        {
            int M;
            sscanf(line, "%d %d %d", N, &M, nnz);
            assert(M == *N);
            *row = (int*) malloc(*nnz * sizeof(int));
            *col = (int*) malloc(*nnz * sizeof(int));
            *val = (double*) malloc(*nnz * sizeof(double));
        }
        else
        {
            sscanf(line, "%d %d %lf", *row + l, *col + l, *val + l);
            // Fix numbering from 1
            (*row)[l]--;
            (*col)[l]--;
        }
        l++;
    }

    fclose(fp);
    if (line)
        free(line);
}

bool test_vector(struct ILUFact* ilu, int N, double* v)
{
    int rank;
    int world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    int first_row = test_rank_first_row(rank, N, world_size);
    int last_row = test_rank_first_row(rank + 1, N, world_size);
    int n_local_rows = last_row - first_row;

    int success = 1;

    double* v_part = (double*) malloc(n_local_rows * sizeof(double));
    memcpy(v_part, v + first_row, n_local_rows * sizeof(double));
    double* x = (double*) malloc(n_local_rows * sizeof(double));
    double* res = (double*) malloc(n_local_rows * sizeof(double));
    ILU_solve(ilu, v_part, x);
    // print vector x
    for (int r = 0; r < world_size; r++) {
        if (r == rank) {
            for (int i = 0; i < n_local_rows; i++) {
                printf("%f ", x[i]);
            }
            printf("\n");
        }
        MPI_Barrier(MPI_COMM_WORLD);
        usleep(100000);
    }
    ILU_multiply(ilu, x, res);
    for (int i = first_row; i < last_row; i++)
    {
        if (abs(v_part[i - first_row] - res[i - first_row]) > EPS)
        {
            success = 0;
        }
    }
    int passed;
    MPI_Reduce(&success, &passed, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);
    if (rank == 0)
    {
        if(passed == 0)
        {
            printf("TEST FAILED\n");
        }
        else
        {
            printf("TEST PASSED\n");
        }
    }
    free(v_part);
    free(x);
    free(res);
    return passed;
}

int main(int argc, char* argv[])
{
    assert(argc == 2);

    std::ios_base::sync_with_stdio(false);
    std::cout.rdbuf()->pubsetbuf(nullptr, 0);

    int rank;
    int world_size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int N = 0, nnz = 0;
    int* row = NULL;
    int* col = NULL;
    double* val = NULL;
    if (rank == 0)
    {
        read_matrix(argv[1], &N, &nnz, &row, &col, &val);
    }

    struct ILUFact* ilu;
    ilu = ILU_factorize(N, nnz, row, col, val);
    free(row);
    free(col);
    free(val);

    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);

    double* v1 = (double*) malloc(N * sizeof(double));
    double* v2 = (double*) malloc(N * sizeof(double));
    for (int i = 0; i < N; i++)
    {
        v1[i] = 1;
        v2[i] = i;
    }

    test_vector(ilu, N, v1);
    //test_vector(ilu, N, v2);

    ILU_free(ilu);
    MPI_Finalize();
    return 0;
}
