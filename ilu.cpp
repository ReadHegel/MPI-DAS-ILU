#include "ilu.h"

#include <mpi.h>

#include <algorithm>
#include <numeric>
#include <vector>

struct CSRMatrix {
    int num_rows;
    int num_cols;
    int nnz;
    std::vector<int> row_ptr;   // Size: num_rows + 1
    std::vector<int> col_idx;   // Size: nnz
    std::vector<double> val;    // Size: nnz
    std::vector<int> diag_idx;  // num_rows

    static CSRMatrix from_COO(
        int N, int nnz, const int *row, const int *col, const double *val
    ) {
        CSRMatrix mat;

        mat.num_rows = N;
        mat.num_cols = N;
        mat.nnz = nnz;
        mat.row_ptr.resize(N + 1, 0);
        mat.col_idx.resize(nnz);
        mat.val.resize(nnz);

        std::vector<int> indices(N);
        std::iota(indices.begin(), indices.end(), 0);

        std::sort(indices.begin(), indices.end(), [&](int a, int b) {
            if (row[a] != row[b]) {
                return row[a] < row[b];
            }
            return col[a] < col[b];
        });

        for (int i = 0; i < nnz; ++i) {
            int idx = indices[i];
            int r = row[idx];
            mat.row_ptr[r + 1]++;
            mat.col_idx[i] = col[idx];
            mat.val[i] = val[idx];
        }

        return mat;
    }

    void add_mult_row_to_row(
        double alpha,
        const int target_local_row,
        const double *other_val,
        const int *other_col_idx,
        const int other_nnz
    ) {
        int other_i = 0;
        int i = row_ptr[target_local_row];
        int row_end = row_ptr[target_local_row + 1];

        while (other_i < other_nnz && i < row_end) {
            int other_col = other_col_idx[other_i];
            int col = col_idx[i];

            if (other_col == col) {
                val[i] += alpha * other_val[other_i];
                other_i++;
                i++;
            }
            else if (other_col < col) {
                other_i++;
            }
            else {
                i++;
            }
        }
    }

    static CSRMatrix Identity(int N) {
        CSRMatrix I;
        I.num_rows = N;
        I.nnz = N;
        I.row_ptr.resize(N + 1);
        I.col_idx.resize(N);
        I.val.resize(N, 1.0);

        for (int i = 0; i < N; ++i) {
            I.row_ptr[i] = i;
            I.col_idx[i] = i;
        }
        I.row_ptr[N] = N;

        return I;
    }
};

struct ILUFact {
    int N;
    int rank;
    int world_size;

    int global_offset;
    int num_rows_local;

    int num_interior;
    int num_separator;

    int glob_to_loc_row(int global_row) const {
        return global_row - global_offset;
    }

    int loc_to_glob_row(int local_row) const {
        return local_row + global_offset;
    }

    std::vector<int> perm;
    std::vector<int> inv_perm;

    CSRMatrix LU;  // Unified matrix for both L and U

    std::vector<int> send_to_ranks;
    std::vector<int> recv_from_ranks;
};

static void ILU(CSRMatrix &LU, const int num_rows_from_U_to_factorize) {
    for (int i = 1; i < num_rows_from_U_to_factorize; ++i) {
        for (int k = LU.row_ptr[i]; k < LU.row_ptr[i + 1]; ++k) {
            int row_to_subtract = LU.col_idx[k];

            if (row_to_subtract >= i) {
                break;
            }

            int diag_kk_idx = LU.diag_idx[row_to_subtract];
            LU.val[k] = LU.val[k] / LU.val[diag_kk_idx];

            LU.add_mult_row_to_row(
                -LU.val[k],
                i,
                LU.val.data() + diag_kk_idx + 1,
                LU.col_idx.data() + diag_kk_idx + 1,
                LU.row_ptr[row_to_subtract + 1] - LU.row_ptr[row_to_subtract] -
                    1
            );
        }
    }
}

void distribute_data(
    int N, int nnz, int *row, int *col, double *val, struct ILUFact *ilu
) {
    CSRMatrix &recived_matrix = ilu->LU;

    auto get_first_row_of_process = [&](int rank) {
        int rowsPerProcess = N / ilu->world_size;
        int remainder = N % ilu->world_size;
        return rank * rowsPerProcess + (rank < remainder ? rank : remainder);
    };

    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);

    ilu->global_offset = get_first_row_of_process(ilu->rank);
    ilu->num_rows_local =
        get_first_row_of_process(ilu->rank + 1) - ilu->global_offset;
    recived_matrix.num_rows = ilu->num_rows_local;
    recived_matrix.num_cols = N;
    recived_matrix.row_ptr.resize(recived_matrix.num_rows + 1);

    std::vector<int> col_idx_sendcounts;
    std::vector<int> col_idx_displacements;
    CSRMatrix A;

    if (ilu->rank == 0) {
        A = CSRMatrix::from_COO(N, nnz, row, col, val);
        col_idx_sendcounts.resize(ilu->world_size, 0);
        col_idx_displacements.resize(ilu->world_size, 0);

        int first_row = 0;
        int last_row = get_first_row_of_process(1) - 1;

        col_idx_displacements[0] = A.row_ptr[first_row];
        col_idx_sendcounts[0] = A.row_ptr[last_row + 1] - A.row_ptr[first_row];

        recived_matrix.nnz = col_idx_sendcounts[0];
        std::copy(
            A.row_ptr.begin() + first_row,
            A.row_ptr.begin() + last_row + 2,
            recived_matrix.row_ptr.begin()
        );

        for (int r = 1; r < ilu->world_size; ++r) {
            first_row = last_row + 1;
            last_row = get_first_row_of_process(r + 1) - 1;

            col_idx_sendcounts[r] =
                A.row_ptr[last_row + 1] - A.row_ptr[first_row];
            col_idx_displacements[r] = A.row_ptr[first_row];

            int nnz_for_r = col_idx_sendcounts[r];
            MPI_Send(&nnz_for_r, 1, MPI_INT, r, 0, MPI_COMM_WORLD);
            MPI_Send(
                &A.row_ptr[first_row],
                last_row - first_row + 2,
                MPI_INT,
                r,
                0,
                MPI_COMM_WORLD
            );
        }
    }
    else {
        MPI_Recv(
            &recived_matrix.nnz,
            1,
            MPI_INT,
            0,
            0,
            MPI_COMM_WORLD,
            MPI_STATUS_IGNORE
        );
        MPI_Recv(
            recived_matrix.row_ptr.data(),
            recived_matrix.num_rows + 1,
            MPI_INT,
            0,
            0,
            MPI_COMM_WORLD,
            MPI_STATUS_IGNORE
        );
    }

    recived_matrix.col_idx.resize(recived_matrix.nnz);
    recived_matrix.val.resize(recived_matrix.nnz);

    MPI_Scatterv(
        ilu->rank == 0 ? A.col_idx.data() : nullptr,
        col_idx_sendcounts.data(),
        col_idx_displacements.data(),
        MPI_INT,
        recived_matrix.col_idx.data(),
        recived_matrix.nnz,
        MPI_INT,
        0,
        MPI_COMM_WORLD
    );

    MPI_Scatterv(
        ilu->rank == 0 ? A.val.data() : nullptr,
        col_idx_sendcounts.data(),
        col_idx_displacements.data(),
        MPI_DOUBLE,
        recived_matrix.val.data(),
        recived_matrix.nnz,
        MPI_DOUBLE,
        0,
        MPI_COMM_WORLD
    );

    int offset = recived_matrix.row_ptr[0];
    for (auto &rp : recived_matrix.row_ptr) {
        rp -= offset;
    }
}

/**
 * Performs the distributed ILU factorization.
 */
struct ILUFact *ILU_factorize(int N, int nnz, int *row, int *col, double *val) {
    // 3. Podział wierszy na "interior" i "separator"
    // 4. Renumeracja
    // 5. Lokalna faktoryzacja
    struct ILUFact *ilu = new ILUFact();

    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &ilu->rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ilu->world_size);

    distribute_data(N, nnz, row, col, val, ilu);


    return ilu;
}

/**
 * Solves the linear equation LUx = b.
 */
void ILU_solve(struct ILUFact *ilu, double *b, double *res) {
    // 1. Zastosowanie permutacji do wektora b
    // 2. Rozwiązanie układu z macierzą dolnotrójkątną L (w przód)
    // 3. Rozwiązanie układu z macierzą górnotrójkątną U (wstecz)
    // 4. Komunikacja międzyprocesowa w trakcie rozwiązywania
}

/**
 * Multiplies the vector b by LU.
 */
void ILU_multiply(struct ILUFact *ilu, double *b, double *res) {
    // 1. Mnożenie wektora przez macierz U
    // 2. Mnożenie wyniku przez macierz L
    // 3. Obsługa komunikacji między procesami
}

/**
 * Frees the memory allocated by ILU_factorize.
 */
void ILU_free(struct ILUFact *ilu) {
    if (ilu != nullptr) {
        delete ilu;
    }
}
