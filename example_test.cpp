#include <cstdio>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <vector>
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

void print_vectors(double* v_part, double* res, int n_local_rows, int rank, int world_size)
{
    for (int r = 0; r < world_size; r++) {
        if (r == rank) {
            for (int i = 0; i < n_local_rows; i++) {
                printf("v: %f , res: %f\n, diff: %f\n", v_part[i], res[i], abs(v_part[i] - res[i]));
            }
            printf("\n");
        }
        MPI_Barrier(MPI_COMM_WORLD);
        usleep(100000);
    }
}

bool test_vector(struct ILUFact* ilu, int N, double* v, double start_time)
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

    MPI_Barrier(MPI_COMM_WORLD);
    double end_time = MPI_Wtime();
    double local_time = end_time - start_time;
    if (rank == 0) {
        printf("Time after solve: %f\n", local_time);
    }

    ILU_multiply(ilu, x, res);

    for (int i = 0; i < n_local_rows; i++) {
        if (std::abs(v_part[i] - res[i]) > EPS
            || std::isnan(res[i]) || std::isinf(res[i])) {
            success = 0;
        }
    }
    int passed = 1;
    MPI_Reduce(&success, &passed, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Bcast(&passed, 1, MPI_INT, 0, MPI_COMM_WORLD);
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

void fill_row_partition(int N, int world_size, std::vector<int>& sendcounts, std::vector<int>& displs)
{
    sendcounts.resize(world_size);
    displs.resize(world_size);
    for (int r = 0; r < world_size; r++) {
        int first = test_rank_first_row(r, N, world_size);
        int last = test_rank_first_row(r + 1, N, world_size);
        sendcounts[r] = last - first;
        displs[r] = first;
    }
}

bool check_factorization(
    struct ILUFact* ilu,
    int N,
    int nnz,
    const int* row,
    const int* col,
    const double* val
) {
    int rank;
    int world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    std::vector<int> sendcounts;
    std::vector<int> displs;
    fill_row_partition(N, world_size, sendcounts, displs);

    const int n_local_rows = sendcounts[rank];
    const int first_row = displs[rank];

    double* e_part = (double*) calloc(n_local_rows, sizeof(double));
    double* col_part = (double*) malloc(n_local_rows * sizeof(double));
    std::vector<double> gathered_col;
    std::vector<double> A_dense;
    std::vector<double> LU_dense;

    if (rank == 0) {
        gathered_col.resize(N);
        A_dense.assign(N * N, 0.0);
        LU_dense.assign(N * N, 0.0);
        for (int k = 0; k < nnz; k++) {
            A_dense[row[k] * N + col[k]] = val[k];
        }
    }

    for (int j = 0; j < N; j++) {
        for (int i = 0; i < n_local_rows; i++) {
            e_part[i] = (first_row + i == j) ? 1.0 : 0.0;
        }

        ILU_multiply(ilu, e_part, col_part);

        MPI_Gatherv(
            col_part,
            n_local_rows,
            MPI_DOUBLE,
            rank == 0 ? gathered_col.data() : nullptr,
            rank == 0 ? sendcounts.data() : nullptr,
            rank == 0 ? displs.data() : nullptr,
            MPI_DOUBLE,
            0,
            MPI_COMM_WORLD
        );

        if (rank == 0) {
            for (int i = 0; i < N; i++) {
                LU_dense[i * N + j] = gathered_col[i];
            }
        }
    }

    int success = 1;
    if (rank == 0) {
        printf("Original matrix A:\n");
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                printf("%12.6f ", A_dense[i * N + j]);
            }
            printf("\n");
        }

        printf("Recovered matrix (LU * I):\n");
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                printf("%12.6f ", LU_dense[i * N + j]);
            }
            printf("\n");
        }

        double max_diff = 0.0;
        for (int k = 0; k < nnz; k++) {
            int i = row[k];
            int j = col[k];
            double diff = std::abs(A_dense[i * N + j] - LU_dense[i * N + j]);
            max_diff = std::max(max_diff, diff);
            if (diff > EPS || std::isnan(LU_dense[i * N + j]) || std::isinf(LU_dense[i * N + j])) {
                success = 0;
            }
        }

        if (success) {
            printf("FACTORIZATION CHECK PASSED (max_diff=%.6e)\n", max_diff);
        } else {
            printf("FACTORIZATION CHECK FAILED (max_diff=%.6e)\n", max_diff);
        }
    }

    int passed = 0;
    MPI_Bcast(&success, 1, MPI_INT, 0, MPI_COMM_WORLD);
    passed = success;

    free(e_part);
    free(col_part);
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
   
    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();
    if (rank == 0)
    {   
        printf("Starting test\n");
    }



    struct ILUFact* ilu;
    ilu = ILU_factorize(N, nnz, row, col, val);
    
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&nnz, 1, MPI_INT, 0, MPI_COMM_WORLD);
    check_factorization(ilu, N, nnz, row, col, val);

    free(row);
    free(col);
    free(val);

    double* v1 = (double*) malloc(N * sizeof(double));
    double* v2 = (double*) malloc(N * sizeof(double));
    for (int i = 0; i < N; i++)
    {
        v1[i] = 1;
        v2[i] = i;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        printf("Time after factorization: %f\n", MPI_Wtime() - start_time);
    }

    test_vector(ilu, N, v1, start_time);
    test_vector(ilu, N, v2, start_time);

    MPI_Barrier(MPI_COMM_WORLD);
    double end_time = MPI_Wtime();
    double local_time = end_time - start_time;
    if (rank == 0) {
        printf("Test finished\n");
        printf("RESULT: %f\n", local_time);
    }

    ILU_free(ilu);
    MPI_Finalize();
    return 0;
}
